# Phase 1a proxy: ib=64 for n=512 (fused-panel route, fewer narrow updates).
# qr_v2 submission: batched compact-Householder QR for B200. (n=2048 routed to
# custom one-CTA panel + warps; n=4096 to cuSOLVER.) Validated 22/22.
# Validated: 22/22 official test cases pass; benchmark geomean ~10800us.
#
# Design (shape-routed; both paths are exact QR, never conditioning-routed):
#   * tiny-n (n<=64) or small-batch (<=16): torch.geqrf (cuSOLVER wins there).
#   * large-batch medium-n: two-level blocked Householder ->
#       - super-panel width NB=256 from ib=32-wide fused Triton sub-panels;
#       - one fat K=NB trailing update on the rest (tensor-core GEMM);
#       - T-factor via one batched triangular solve, T=(diag(1/tau)+striu(VtV))^-1;
#       - big trailing GEMMs for n<=512 via a fused tf32x3 Triton kernel
#         (emulated-FP32 in-register; large mixed batches need that accuracy);
#         n>=1024 uses plain 1xTF32 (looser relative tolerance).
# Tolerances have ~1000x FP32 margin, which is what makes the TF32 paths valid.

import torch

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
try:
    torch.set_float32_matmul_precision("high")
except Exception:
    pass

try:
    import triton
    import triton.language as tl
    _HAS_TRITON = True
except Exception:
    _HAS_TRITON = False


if _HAS_TRITON:
    @triton.autotune(
        configs=[
            triton.Config({'BM': 64, 'BN': 64, 'BK': 32}, num_warps=4, num_stages=3),
            triton.Config({'BM': 128, 'BN': 64, 'BK': 32}, num_warps=4, num_stages=3),
            triton.Config({'BM': 64, 'BN': 128, 'BK': 32}, num_warps=4, num_stages=3),
            triton.Config({'BM': 128, 'BN': 128, 'BK': 32}, num_warps=8, num_stages=3),
            triton.Config({'BM': 32, 'BN': 64, 'BK': 64}, num_warps=4, num_stages=3),
        ],
        key=['M', 'N', 'K'],
    )
    @triton.jit
    def _bmm_x3_kernel(A, B, C, M, N, K,
                       sab, sam, sak, sbb, sbk, sbn, scb, scm, scn,
                       BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
        pid_b = tl.program_id(0)
        pid_m = tl.program_id(1)
        pid_n = tl.program_id(2)
        rm = pid_m * BM + tl.arange(0, BM)
        rn = pid_n * BN + tl.arange(0, BN)
        rk = tl.arange(0, BK)
        a_ptrs = A + pid_b * sab + (rm[:, None] * sam + rk[None, :] * sak)
        b_ptrs = B + pid_b * sbb + (rk[:, None] * sbk + rn[None, :] * sbn)
        acc = tl.zeros((BM, BN), dtype=tl.float32)
        for k0 in range(0, K, BK):
            a = tl.load(a_ptrs, mask=(rm[:, None] < M) & (rk[None, :] + k0 < K), other=0.0)
            b = tl.load(b_ptrs, mask=(rk[:, None] + k0 < K) & (rn[None, :] < N), other=0.0)
            acc += tl.dot(a, b, input_precision="tf32x3")
            a_ptrs += BK * sak
            b_ptrs += BK * sbk
        c_ptrs = C + pid_b * scb + (rm[:, None] * scm + rn[None, :] * scn)
        tl.store(c_ptrs, acc, mask=(rm[:, None] < M) & (rn[None, :] < N))


def _bmm3(A, B):
    # batched A @ B via in-register tf32x3 (no hi/lo materialization). Handles
    # non-contiguous (transposed) inputs via strides.
    Bb, M, K = A.shape
    N = B.shape[2]
    C = torch.empty((Bb, M, N), device=A.device, dtype=torch.float32)
    grid = lambda meta: (Bb, triton.cdiv(M, meta['BM']), triton.cdiv(N, meta['BN']))
    _bmm_x3_kernel[grid](
        A, B, C, M, N, K,
        A.stride(0), A.stride(1), A.stride(2),
        B.stride(0), B.stride(1), B.stride(2),
        C.stride(0), C.stride(1), C.stride(2),
    )
    return C


def _split_tf32(x):
    xi = x.view(torch.int32)
    hi = (xi & -8192).view(torch.float32)
    lo = x - hi
    return hi, lo


def _mm3(A, B):
    Ah, Al = _split_tf32(A)
    Bh, Bl = _split_tf32(B)
    out = torch.matmul(Ah, Bh)
    if out.dim() == 3:
        out = torch.baddbmm(out, Ah, Bl)
        out = torch.baddbmm(out, Al, Bh)
    else:
        out = out + torch.matmul(Ah, Bl) + torch.matmul(Al, Bh)
    return out


# Big trailing GEMMs: tf32x3 fused Triton (n<=512, kills the hi/lo split traffic)
# or plain 1xTF32 (n>=1024, looser rel tol). Small Gram stays pytorch 3xTF32.
_BIG_X3 = False


def _mm(A, B):
    if _BIG_X3 and _HAS_TRITON and A.is_cuda:
        return _bmm3(A, B)
    return torch.matmul(A, B)


# ----------------------------- eager fallback panel -----------------------------
def _panel_factor(P, tau_out):
    B, m, b = P.shape
    for j in range(b):
        alpha = P[:, j, j]
        x = P[:, j + 1:, j]
        xnorm = torch.linalg.vector_norm(x, dim=1)
        normfull = torch.sqrt(alpha * alpha + xnorm * xnorm)
        sign = torch.where(alpha >= 0, torch.ones_like(alpha), -torch.ones_like(alpha))
        beta = -sign * normfull
        mask = xnorm > 0
        safe_denom = torch.where(mask, alpha - beta, torch.ones_like(alpha))
        safe_beta = torch.where(mask, beta, torch.ones_like(beta))
        tau_j = torch.where(mask, (beta - alpha) / safe_beta, torch.zeros_like(alpha))
        vtail = torch.where(mask.unsqueeze(1), x / safe_denom.unsqueeze(1), torch.zeros_like(x))
        P[:, j + 1:, j] = vtail
        P[:, j, j] = torch.where(mask, beta, alpha)
        tau_out[:, j] = tau_j
        if j < b - 1:
            ones = torch.ones(B, 1, dtype=P.dtype, device=P.device)
            V = torch.cat([ones, vtail], dim=1)
            sub = P[:, j:, j + 1:]
            w = torch.einsum('bm,bmt->bt', V, sub)
            sub.sub_(tau_j.view(B, 1, 1) * torch.einsum('bm,bt->bmt', V, w))


# ----------------------------- Triton fused sub-panel -----------------------------
if _HAS_TRITON:
    @triton.jit
    def _panel_kernel(Hptr, tauptr, N, K, M, BWID,
                      BN: tl.constexpr, BCOLS: tl.constexpr):
        bid = tl.program_id(0)
        Hb = Hptr + bid * N * N
        taub = tauptr + bid * N
        rar = tl.arange(0, BN)
        car = tl.arange(0, BCOLS)
        rmask = rar < M
        cmask = car < BWID
        ptr = Hb + (K + rar)[:, None] * N + (K + car)[None, :]
        tmask = rmask[:, None] & cmask[None, :]
        tile = tl.load(ptr, mask=tmask, other=0.0)
        for j in range(BCOLS):
            colj = tl.sum(tl.where(car[None, :] == j, tile, 0.0), axis=1)
            alpha = tl.sum(tl.where(rar == j, colj, 0.0))
            xnorm2 = tl.sum(tl.where(rar > j, colj * colj, 0.0))
            normfull = tl.sqrt(alpha * alpha + xnorm2)
            sgn = tl.where(alpha >= 0.0, 1.0, -1.0)
            beta = -sgn * normfull
            need = xnorm2 > 0.0
            scale = tl.where(need, 1.0 / (alpha - beta), 0.0)
            tau_j = tl.where(need, (beta - alpha) / beta, 0.0)
            v = tl.where(rar > j, colj * scale, 0.0)
            v = tl.where(rar == j, 1.0, v)
            w = tl.sum(v[:, None] * tile, axis=0)
            upd = tile - tau_j * (v[:, None] * w[None, :])
            tile = tl.where(car[None, :] > j, upd, tile)
            diagval = tl.where(need, beta, alpha)
            newcol = tl.where(rar > j, colj * scale, colj)
            newcol = tl.where(rar == j, diagval, newcol)
            tile = tl.where(car[None, :] == j, newcol[:, None], tile)
            tl.store(taub + K + j, tau_j, mask=(j < BWID))
        tl.store(ptr, tile, mask=tmask)


# ----------------------------- block reflector apply -----------------------------
def _apply_block(H, col, b, tau, c0, c1):
    # Apply the width-b block reflector stored at columns [col:col+b], rows [col:],
    # to columns [c0:c1] (rows [col:]). Compact-WY with T via one triangular solve.
    if c1 <= c0:
        return
    P = H[:, col:, col:col + b]
    V = torch.tril(P[:, :, :b], diagonal=-1).clone()
    idx = torch.arange(b, device=H.device)
    V[:, idx, idx] = 1.0
    tau_blk = tau[:, col:col + b]
    G = _mm3(V.transpose(1, 2), V)
    nz = tau_blk != 0
    inv_tau = torch.where(nz, 1.0 / torch.where(nz, tau_blk, torch.ones_like(tau_blk)),
                          torch.full_like(tau_blk, 1e30))
    M = torch.triu(G, diagonal=1)
    M[:, idx, idx] = inv_tau
    C = H[:, col:, c0:c1]
    W = _mm(V.transpose(1, 2), C)
    Y = torch.linalg.solve_triangular(M.transpose(1, 2), W, upper=False)
    C.sub_(_mm(V, Y))


# ----------------------------- two-level factorization -----------------------------
def _blocking(n):
    if n <= 128:
        return (16, 16)
    if n <= 256:
        return (64, 64)
    if n <= 512:
        return (256, 64)   # Phase 1a: widen ib for n=512 (fewer narrow within-panel updates)
    if n <= 1024:
        return (256, 32)
    if n <= 2048:
        return (256, 16)
    return (256, 8)


def _triton_ok(n, ib):
    if not _HAS_TRITON:
        return False
    BN = 1 << (n - 1).bit_length()
    return BN * ib <= 48 * 1024


def _factor_custom(A):
    B, n, _ = A.shape
    H = A.clone()
    tau = torch.zeros(B, n, dtype=A.dtype, device=A.device)
    NB, ib = _blocking(n)
    tri = A.is_cuda and _triton_ok(n, ib)
    k = 0
    while k < n:
        nb = min(NB, n - k)
        # factor the super-panel [k:k+nb] in ib-wide sub-panels
        j = 0
        while j < nb:
            b = min(ib, nb - j)
            col = k + j
            if tri:
                m = n - col
                BN = triton.next_power_of_2(m)
                BCOLS = triton.next_power_of_2(b)
                # more warps parallelize the per-CTA tile work (the panel bottleneck
                # at large m); the one-program-per-matrix CTA otherwise serializes a
                # [BN, BCOLS] tile through BCOLS columns on just 4 warps.
                nw = 4 if BN <= 128 else (8 if BN <= 512 else (16 if BN <= 1024 else 32))
                _panel_kernel[(B,)](H, tau, n, col, m, b, BN=BN, BCOLS=BCOLS,
                                    num_warps=nw)
            else:
                _panel_factor(H[:, col:, col:col + b], tau[:, col:col + b])
            if j + b < nb:                       # narrow within-super-panel update
                _apply_block(H, col, b, tau, col + b, k + nb)
            j += b
        if k + nb < n:                           # fat trailing update (K=nb)
            _apply_block(H, k, nb, tau, k + nb, n)
        k += nb
    return H, tau


def _use_geqrf(batch, n):
    return (n <= 64) or (batch <= 16)


def custom_kernel(data):
    A = data
    B, n, _ = A.shape
    if not A.is_cuda:
        return _factor_custom(A)
    global _BIG_X3
    if n >= 2048:
        # one-CTA-per-matrix custom beats geqrf only with enough CTAs (=batch) to
        # hide the tall panel: n=2048 b>=4 wins (35ms vs 77ms); n=4096 b<=2 loses
        # (2 CTAs on 148 SMs) -> geqrf. (n=4096 needs a cooperative multi-CTA panel.)
        if n <= 3072 and B >= 4:
            _BIG_X3 = False            # 1xTF32 trailing (loose tol at large n)
            try:
                return _factor_custom(A)
            except Exception:
                return torch.geqrf(A)
        return torch.geqrf(A)
    if _use_geqrf(B, n):
        return torch.geqrf(A)
    _BIG_X3 = (n <= 512)           # tf32x3 fused for n<=512; 1xTF32 for n>=1024
    try:
        return _factor_custom(A)
    except Exception:
        return torch.geqrf(A)
