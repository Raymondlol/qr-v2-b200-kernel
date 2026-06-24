# Adds a LARGE-N path to the hybrid: blocked Householder where each narrow panel
# is factored by cuSOLVER (torch.geqrf on the tall-skinny panel slice) and the
# wide trailing update runs on tensor cores via 3xTF32. For small-batch large-n,
# the trailing GEMM is the bulk FLOP, so moving it off cuSOLVER's FP32 path onto
# tensor cores should beat full geqrf (which does everything in FP32).
#
# Routing (shape-only, both paths exact QR):
#   * tiny n               -> full geqrf
#   * large batch, n<=1024 -> Triton-panel custom (cand_D)
#   * small batch, large n -> geqrf-panel blocked custom (this file)

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
    return torch.matmul(Ah, Bh) + torch.matmul(Ah, Bl) + torch.matmul(Al, Bh)


# ----------------------------- Triton fused panel -----------------------------
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


def _panel_factor_triton(H, k, b, tau):
    B, n, _ = H.shape
    m = n - k
    BN = triton.next_power_of_2(m)
    BCOLS = triton.next_power_of_2(b)
    _panel_kernel[(B,)](H, tau, n, k, m, b, BN=BN, BCOLS=BCOLS)


# ----------------------------- shared trailing update -----------------------------
def _build_V(P, b):
    V = torch.tril(P[:, :, :b], diagonal=-1).clone()
    idx = torch.arange(b, device=P.device)
    V[:, idx, idx] = 1.0
    return V


def _trailing_update(H, k, b, tau, V=None):
    P = H[:, k:, k:k + b]
    if V is None:
        V = _build_V(P, b)
    tau_blk = tau[:, k:k + b]
    G = _mm3(V.transpose(1, 2), V)
    nz = tau_blk != 0
    inv_tau = torch.where(nz, 1.0 / torch.where(nz, tau_blk, torch.ones_like(tau_blk)),
                          torch.full_like(tau_blk, 1e30))
    M = torch.triu(G, diagonal=1)
    idx = torch.arange(b, device=H.device)
    M[:, idx, idx] = inv_tau
    C = H[:, k:, k + b:]
    W = _mm3(V.transpose(1, 2), C)
    Y = torch.linalg.solve_triangular(M.transpose(1, 2), W, upper=False)
    C.sub_(_mm3(V, Y))


# ----------------------------- Triton-panel custom (large batch) -----------------------------
def _block_size_smalln(n):
    if n <= 128:
        return 16
    if n <= 512:
        return 64
    return 32


def _triton_ok(n, block):
    if not _HAS_TRITON:
        return False
    BN = 1 << (n - 1).bit_length()
    return n <= 1024 and BN * block <= 64 * 1024


def _factor_triton_custom(A):
    B, n, _ = A.shape
    H = A.clone()
    tau = torch.zeros(B, n, dtype=A.dtype, device=A.device)
    block = _block_size_smalln(n)
    tri = A.is_cuda and _triton_ok(n, block)
    k = 0
    while k < n:
        b = min(block, n - k)
        if tri:
            _panel_factor_triton(H, k, b, tau)
        else:
            _factor_geqrf_panel_inplace(H, tau, k, b)
        if k + b < n:
            _trailing_update(H, k, b, tau)
        k += b
    return H, tau


# ----------------------------- geqrf-panel blocked (small batch large n) -----------------------------
def _factor_geqrf_panel_inplace(H, tau, k, b):
    # Factor panel H[:, k:, k:k+b] with cuSOLVER and write compact form back.
    P = H[:, k:, k:k + b]
    hp, tp = torch.geqrf(P)
    H[:, k:, k:k + b] = hp
    tau[:, k:k + b] = tp


def _block_size_largen(n):
    if n >= 4096:
        return 128
    if n >= 2048:
        return 128
    return 64


def _factor_geqrf_blocked(A):
    B, n, _ = A.shape
    H = A.clone()
    tau = torch.zeros(B, n, dtype=A.dtype, device=A.device)
    block = _block_size_largen(n)
    k = 0
    while k < n:
        b = min(block, n - k)
        _factor_geqrf_panel_inplace(H, tau, k, b)
        if k + b < n:
            _trailing_update(H, k, b, tau)
        k += b
    return H, tau


# ----------------------------- routing -----------------------------
def custom_kernel(data):
    A = data
    B, n, _ = A.shape
    if not A.is_cuda:
        return _factor_geqrf_blocked(A)
    if n <= 64:
        return torch.geqrf(A)
    if B <= 16 and n >= 1024:          # small batch, large n -> geqrf-panel blocked
        try:
            return _factor_geqrf_blocked(A)
        except Exception:
            return torch.geqrf(A)
    if B <= 16:                         # small batch, smaller n -> full geqrf
        return torch.geqrf(A)
    try:
        return _factor_triton_custom(A)  # large batch -> Triton-panel custom
    except Exception:
        return torch.geqrf(A)
