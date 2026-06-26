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

    @triton.autotune(
        configs=[
            triton.Config({'BM': 64, 'BN': 64, 'BK': 32}, num_warps=4, num_stages=3),
            triton.Config({'BM': 128, 'BN': 64, 'BK': 32}, num_warps=4, num_stages=3),
            triton.Config({'BM': 64, 'BN': 128, 'BK': 32}, num_warps=4, num_stages=3),
            triton.Config({'BM': 128, 'BN': 128, 'BK': 32}, num_warps=8, num_stages=3),
            triton.Config({'BM': 32, 'BN': 64, 'BK': 64}, num_warps=4, num_stages=3),
        ],
        key=['M', 'N', 'K'],
        restore_value=['C'],   # in-place (C -= A@B): autotune re-runs configs on the
                               # same buffer, so C MUST be restored between trials or the
                               # first call (autotune) over-subtracts -> wrong result.
    )
    @triton.jit
    def _bmm_x3_sub_kernel(A, B, C, M, N, K,
                           sab, sam, sak, sbb, sbk, sbn, scb, scm, scn,
                           BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
        # C -= A @ B (tf32x3), fused subtract epilogue (kills the separate sub_ kernel
        # AND the intermediate matmul output for the n<=512 trailing update).
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
        cmask = (rm[:, None] < M) & (rn[None, :] < N)
        c_ptrs = C + pid_b * scb + (rm[:, None] * scm + rn[None, :] * scn)
        prev = tl.load(c_ptrs, mask=cmask, other=0.0)
        tl.store(c_ptrs, prev - acc, mask=cmask)


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


def _bmm3_sub(A, B, C):
    # C -= A @ B in-place (tf32x3), fused. C is a strided view into H.
    Bb, M, K = A.shape
    N = B.shape[2]
    grid = lambda meta: (Bb, triton.cdiv(M, meta['BM']), triton.cdiv(N, meta['BN']))
    _bmm_x3_sub_kernel[grid](
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


def _mm_sub(A, B, C):
    # C -= A @ B, fused: tf32x3 subtract-epilogue kernel for n<=512 (keeps precision),
    # cuBLAS baddbmm for n>=1024 (1xTF32) -- both avoid a separate matmul output + sub_.
    if _BIG_X3 and _HAS_TRITON and C.is_cuda:
        _bmm3_sub(A, B, C)
    else:
        C.baddbmm_(A, B, beta=1.0, alpha=-1.0)


def _gram(A, B):
    # T-factor Gram VtV (profiled at 15% of n=512 / 29% of n=1024 factor time). Was
    # unconditionally the hi/lo-split _mm3; routed like the trailing instead: fused
    # tf32x3 for n<=512 (the bake-off showed fused beats the 3xcuBLAS split), plain
    # 1xTF32 for n>=1024 (loose tol; V is well-conditioned unit reflectors, |.|~O(1)).
    # Modal apples-to-apples: geomean 7793 -> 6907 (1.13x), 22/22, mixed margins
    # unchanged at the 2.0x safe floor (the 1xTF32 trailing already pinned n>=1024).
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


# ----------------------------- IMPLICIT-V apply (DLARFB; V never materialized) -----
# V is the unit-lower-trapezoidal panel: V[r,i] = 0 (r<i), 1 (r==i), H[col+r,col+i] (r>i).
# Load V on-the-fly from H with a triangular mask in each GEMM -> kill the torch.tril
# materialization (a standalone memory-bound op) + the glue. tf32x3 throughout.
if _HAS_TRITON:
    _IV_CFGS = [
        triton.Config({'BM': 64, 'BN': 64, 'BK': 32}, num_warps=4, num_stages=3),
        triton.Config({'BM': 128, 'BN': 64, 'BK': 32}, num_warps=4, num_stages=3),
        triton.Config({'BM': 64, 'BN': 128, 'BK': 32}, num_warps=4, num_stages=3),
        triton.Config({'BM': 128, 'BN': 128, 'BK': 32}, num_warps=8, num_stages=3),
        triton.Config({'BM': 32, 'BN': 64, 'BK': 64}, num_warps=4, num_stages=3),
    ]

    @triton.autotune(configs=_IV_CFGS, key=['m', 'Wd'])
    @triton.jit
    def _iv_VtC(Hp, Wp, shb, shr, shc, swb, swi, swj, col, c0, m, b, Wd,
                BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
        # W[i,j] = sum_r V[r,i] * C[r,j];  C = H[:,col:,c0:].
        pb = tl.program_id(0); pi = tl.program_id(1); pj = tl.program_id(2)
        ri = pi * BM + tl.arange(0, BM); rj = pj * BN + tl.arange(0, BN); rk = tl.arange(0, BK)
        Hb = Hp + pb * shb
        acc = tl.zeros((BM, BN), tl.float32)
        for k0 in range(0, m, BK):
            r = k0 + rk
            ah = tl.load(Hb + (col + r[None, :]) * shr + (col + ri[:, None]) * shc,
                         mask=(r[None, :] < m) & (ri[:, None] < b), other=0.0)
            a = tl.where(r[None, :] < ri[:, None], 0.0,
                         tl.where(r[None, :] == ri[:, None], 1.0, ah))
            bb = tl.load(Hb + (col + r[:, None]) * shr + (c0 + rj[None, :]) * shc,
                         mask=(r[:, None] < m) & (rj[None, :] < Wd), other=0.0)
            acc += tl.dot(a, bb, input_precision="tf32x3")
        tl.store(Wp + pb * swb + ri[:, None] * swi + rj[None, :] * swj, acc,
                 mask=(ri[:, None] < b) & (rj[None, :] < Wd))

    @triton.autotune(configs=_IV_CFGS, key=['m', 'b'])
    @triton.jit
    def _iv_VtV(Hp, Gp, shb, shr, shc, sgb, sgi, sgj, col, m, b,
                BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
        # G[i,j] = sum_r V[r,i] * V[r,j]  (both operands implicit).
        pb = tl.program_id(0); pi = tl.program_id(1); pj = tl.program_id(2)
        ri = pi * BM + tl.arange(0, BM); rj = pj * BN + tl.arange(0, BN); rk = tl.arange(0, BK)
        Hb = Hp + pb * shb
        acc = tl.zeros((BM, BN), tl.float32)
        for k0 in range(0, m, BK):
            r = k0 + rk
            ah = tl.load(Hb + (col + r[None, :]) * shr + (col + ri[:, None]) * shc,
                         mask=(r[None, :] < m) & (ri[:, None] < b), other=0.0)
            a = tl.where(r[None, :] < ri[:, None], 0.0,
                         tl.where(r[None, :] == ri[:, None], 1.0, ah))
            bh = tl.load(Hb + (col + r[:, None]) * shr + (col + rj[None, :]) * shc,
                         mask=(r[:, None] < m) & (rj[None, :] < b), other=0.0)
            bv = tl.where(r[:, None] < rj[None, :], 0.0,
                          tl.where(r[:, None] == rj[None, :], 1.0, bh))
            acc += tl.dot(a, bv, input_precision="tf32x3")
        tl.store(Gp + pb * sgb + ri[:, None] * sgi + rj[None, :] * sgj, acc,
                 mask=(ri[:, None] < b) & (rj[None, :] < b))

    @triton.autotune(configs=_IV_CFGS, key=['m', 'Wd'], restore_value=['Cp'])
    @triton.jit
    def _iv_VYsub(Hp, Cp, Yp, shb, shr, shc, scb, scr, scc, syb, syi, syj, col, m, b, Wd,
                  BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
        # C[r,j] -= sum_i V[r,i] * Y[i,j], in place.  Cp = H[:,col:,c0:] view.
        pb = tl.program_id(0); pr = tl.program_id(1); pj = tl.program_id(2)
        rr = pr * BM + tl.arange(0, BM); rj = pj * BN + tl.arange(0, BN); rk = tl.arange(0, BK)
        Hb = Hp + pb * shb
        acc = tl.zeros((BM, BN), tl.float32)
        for k0 in range(0, b, BK):
            i = k0 + rk
            ah = tl.load(Hb + (col + rr[:, None]) * shr + (col + i[None, :]) * shc,
                         mask=(rr[:, None] < m) & (i[None, :] < b), other=0.0)
            a = tl.where(rr[:, None] < i[None, :], 0.0,
                         tl.where(rr[:, None] == i[None, :], 1.0, ah))
            bb = tl.load(Yp + pb * syb + i[:, None] * syi + rj[None, :] * syj,
                         mask=(i[:, None] < b) & (rj[None, :] < Wd), other=0.0)
            acc += tl.dot(a, bb, input_precision="tf32x3")
        cptr = Cp + pb * scb + rr[:, None] * scr + rj[None, :] * scc
        cmask = (rr[:, None] < m) & (rj[None, :] < Wd)
        tl.store(cptr, tl.load(cptr, mask=cmask, other=0.0) - acc, mask=cmask)


def _apply_block_implicit(H, col, b, tau, c0, c1):
    B, N, _ = H.shape
    m = N - col; Wd = c1 - c0
    shb, shr, shc = H.stride()
    G = torch.empty((B, b, b), device=H.device, dtype=torch.float32)
    _iv_VtV[lambda M: (B, triton.cdiv(b, M['BM']), triton.cdiv(b, M['BN']))](
        H, G, shb, shr, shc, *G.stride(), col, m, b)
    tau_blk = tau[:, col:col + b]
    nz = tau_blk != 0
    inv_tau = torch.where(nz, 1.0 / torch.where(nz, tau_blk, torch.ones_like(tau_blk)),
                          torch.full_like(tau_blk, 1e30))
    Mm = torch.triu(G, diagonal=1)
    Mm.diagonal(dim1=-2, dim2=-1).copy_(inv_tau)
    W = torch.empty((B, b, Wd), device=H.device, dtype=torch.float32)
    _iv_VtC[lambda M: (B, triton.cdiv(b, M['BM']), triton.cdiv(Wd, M['BN']))](
        H, W, shb, shr, shc, *W.stride(), col, c0, m, b, Wd)
    Y = torch.linalg.solve_triangular(Mm.transpose(1, 2), W, upper=False)
    C = H[:, col:, c0:c1]
    _iv_VYsub[lambda M: (B, triton.cdiv(m, M['BM']), triton.cdiv(Wd, M['BN']))](
        H, C, Y, shb, shr, shc, *C.stride(), *Y.stride(), col, m, b, Wd)


# ----------------------------- block reflector apply -----------------------------
def _apply_block(H, col, b, tau, c0, c1):
    # n<=512 (tf32x3) -> IMPLICIT-V DLARFB (no torch.tril). n>=1024 -> explicit (1xTF32).
    if c1 <= c0:
        return
    if _BIG_X3 and _HAS_TRITON and H.is_cuda:
        _apply_block_implicit(H, col, b, tau, c0, c1)
        return
    P = H[:, col:, col:col + b]
    V = torch.tril(P[:, :, :b], diagonal=-1)
    V.diagonal(dim1=-2, dim2=-1).fill_(1.0)
    tau_blk = tau[:, col:col + b]
    G = _gram(V.transpose(1, 2), V)
    nz = tau_blk != 0
    inv_tau = torch.where(nz, 1.0 / torch.where(nz, tau_blk, torch.ones_like(tau_blk)),
                          torch.full_like(tau_blk, 1e30))
    M = torch.triu(G, diagonal=1)
    M.diagonal(dim1=-2, dim2=-1).copy_(inv_tau)
    C = H[:, col:, c0:c1]
    W = _mm(V.transpose(1, 2), C)
    Y = torch.linalg.solve_triangular(M.transpose(1, 2), W, upper=False)
    _mm_sub(V, Y, C)   # C -= V @ Y, fused (no separate matmul output + sub_ kernel)


# ----------------------------- two-level factorization -----------------------------
def _blocking(n):
    if n <= 128:
        return (16, 16)
    if n <= 256:
        return (64, 64)
    if n <= 512:
        return (128, 64)   # NB=128/ib=64 probe (ib stays 64 throughout -> precision floor untouched)
    if n <= 1024:
        return (256, 128)  # ib_max; _max_ib caps at 32 where m large, 64/128 where m small
    if n <= 2048:
        return (256, 128)  # ib_max; _max_ib caps at 16 where m large, growing as m shrinks
    return (256, 8)


def _max_ib(m, ib_max):
    # Adaptive sub-panel width: largest ib whose RESIDENT panel tile
    # next_pow2(m)*next_pow2(ib) stays under the same 48*1024-fp32-element SRAM gate
    # the kernel respects (mirrors its BCOLS=next_pow2(b) rounding, so never overflows).
    # At the top of the matrix (m=n) this returns today's ib; as m halves marching
    # down, the byte budget lets ib double for FREE -> fewer + fatter (K=64/128)
    # narrow updates (n=2048 narrow-apply count 120->78). Modal: n=2048 27.3->23.9ms
    # (1.14x), n=1024 8.8->8.5ms, geomean 6907->6771; n=512 untouched (precision safe).
    BN = 1 << (m - 1).bit_length()
    ib = ib_max
    while ib > 1 and BN * (1 << (ib - 1).bit_length()) > 48 * 1024:
        ib //= 2
    return ib


def _triton_ok(n, ib):
    if not _HAS_TRITON:
        return False
    BN = 1 << (n - 1).bit_length()
    return BN * ib <= 48 * 1024


def _factor_custom(A):
    B, n, _ = A.shape
    H = A.clone()
    tau = torch.zeros(B, n, dtype=A.dtype, device=A.device)
    NB, ib_max = _blocking(n)
    tri = A.is_cuda and _HAS_TRITON
    k = 0
    while k < n:
        nb = min(NB, n - k)
        # factor the super-panel [k:k+nb] in ADAPTIVE-width sub-panels (ib grows as
        # remaining height m shrinks, under the same SRAM byte budget)
        j = 0
        while j < nb:
            col = k + j
            m = n - col
            b = min(_max_ib(m, ib_max), nb - j)   # fatter ib where remaining m is small
            if tri:
                BN = triton.next_power_of_2(m)
                BCOLS = triton.next_power_of_2(b)
                # The panel is LATENCY-bound (sequential reflector reductions), so warps
                # past 8 don't help and HURT (microbench_panel.py: first sub-panel best
                # nw=8 for BN=512/1024/2048; nw=16/32 were 8-13% slower, nw=4 6x slower).
                nw = 4 if BN <= 128 else 8
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
