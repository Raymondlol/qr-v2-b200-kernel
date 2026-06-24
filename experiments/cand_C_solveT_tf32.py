# Batched square Householder QR (compact-WY).
#   * Fused Triton panel kernel (1 launch / panel)        -> low dispatch overhead
#   * T-factor via a single batched triangular solve       -> kills the dlarft loop
#   * TF32 tensor cores on the trailing-update GEMMs        -> bulk FLOP on tensor cores
# Eager path (Triton off) is CPU-verifiable against the real checker.

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


# =========================== eager reference (verified) ===========================

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


# =========================== fused Triton panel kernel ===========================

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


# =============================== trailing update (solve-T) ===============================

def _build_V(P, b):
    V = torch.tril(P[:, :, :b], diagonal=-1).clone()
    idx = torch.arange(b, device=P.device)
    V[:, idx, idx] = 1.0
    return V


def _trailing_update(H, k, b, tau):
    # Apply Q_block^T = I - V T^T V^T to the trailing columns, with
    # T = (diag(1/tau) + striu(V^T V))^{-1}  -> one triangular solve, no T formed.
    P = H[:, k:, k:k + b]
    V = _build_V(P, b)                              # (B, m, b) unit-lower
    tau_blk = tau[:, k:k + b]                        # (B, b)
    G = torch.matmul(V.transpose(1, 2), V)          # (B, b, b) Gram
    nz = tau_blk != 0
    inv_tau = torch.where(nz, 1.0 / torch.where(nz, tau_blk, torch.ones_like(tau_blk)),
                          torch.full_like(tau_blk, 1e30))
    M = torch.triu(G, diagonal=1)
    idx = torch.arange(b, device=H.device)
    M[:, idx, idx] = inv_tau
    C = H[:, k:, k + b:]
    W = torch.matmul(V.transpose(1, 2), C)          # (B, b, t)
    Y = torch.linalg.solve_triangular(M.transpose(1, 2), W, upper=False)  # M^T Y = W
    C.sub_(torch.matmul(V, Y))


# =============================== full factorization ===============================

def _block_size(n):
    if n <= 256:
        return 16
    if n <= 1024:
        return 32
    return 64


def _triton_ok(n, block):
    if not _HAS_TRITON:
        return False
    BN = 1 << (n - 1).bit_length()
    return n <= 1024 and BN * block <= 48 * 1024


def _factor_into(A, use_triton):
    B, n, _ = A.shape
    H = A.clone()
    tau = torch.zeros(B, n, dtype=A.dtype, device=A.device)
    block = _block_size(n)
    tri = use_triton and A.is_cuda and _triton_ok(n, block)
    k = 0
    while k < n:
        b = min(block, n - k)
        if tri:
            _panel_factor_triton(H, k, b, tau)
        else:
            _panel_factor(H[:, k:, k:k + b], tau[:, k:k + b])
        if k + b < n:
            _trailing_update(H, k, b, tau)
        k += b
    return H, tau


def custom_kernel(data):
    try:
        return _factor_into(data, use_triton=True)
    except Exception:
        return _factor_into(data, use_triton=False)
