# Two-level blocked Householder QR for the large-batch custom path.
#   * super-panel width NB (e.g. 128) factored from ib-wide (e.g. 32) Triton
#     sub-panels with narrow within-super-panel updates;
#   * then ONE fat K=NB trailing update on the rest -> efficient tensor-core GEMM.
# Plus: geqrf routing for small-batch/tiny-n, 3xTF32 (baddbmm) emulated-fp32 GEMMs.

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
    W = _mm3(V.transpose(1, 2), C)
    Y = torch.linalg.solve_triangular(M.transpose(1, 2), W, upper=False)
    C.sub_(_mm3(V, Y))


# ----------------------------- two-level factorization -----------------------------
def _blocking(n):
    # (NB super-panel width, ib sub-panel width). NB==ib -> plain one-level blocking.
    if n <= 128:
        return (16, 16)
    if n <= 256:
        return (64, 64)
    return (128, 32)


def _triton_ok(n, ib):
    if not _HAS_TRITON:
        return False
    BN = 1 << (n - 1).bit_length()
    return n <= 1024 and BN * ib <= 64 * 1024


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
                _panel_kernel[(B,)](H, tau, n, col, m, b, BN=BN, BCOLS=BCOLS)
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
    if _use_geqrf(B, n):
        return torch.geqrf(A)
    try:
        return _factor_custom(A)
    except Exception:
        return torch.geqrf(A)
