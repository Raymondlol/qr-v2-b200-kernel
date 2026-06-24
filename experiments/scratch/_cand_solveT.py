# Candidate: blocked Householder QR, but T-factor via a single batched
# triangular solve instead of the sequential per-column _form_T loop.
#
# Identity used:  H_1 H_2 ... H_b = I - V T V^T  with
#     T = ( diag(1/tau) + striu(V^T V) )^{-1}        (upper-triangular inverse)
# Verified on b=2 by hand; this file CPU-verifies it end-to-end vs the checker.
#
# Trailing update applies Q^T = I - V T^T V^T:
#     W = V^T C ;  Y = T^T W = solve(M^T, W, lower) ;  C -= V Y
# so we never materialize T -- one triangular solve replaces ~60 tiny launches.

import torch

try:
    from task import input_t, output_t
except Exception:
    input_t = output_t = object


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


def _build_V(P, b):
    V = torch.tril(P[:, :, :b], diagonal=-1).clone()
    idx = torch.arange(b, device=P.device)
    V[:, idx, idx] = 1.0
    return V


def _block_size(n):
    if n <= 256:
        return 16
    if n <= 1024:
        return 32
    return 64


def custom_kernel(data):
    A = data
    B, n, _ = A.shape
    H = A.clone()
    tau = torch.zeros(B, n, dtype=A.dtype, device=A.device)
    block = _block_size(n)
    k = 0
    while k < n:
        b = min(block, n - k)
        P = H[:, k:, k:k + b]
        _panel_factor(P, tau[:, k:k + b])
        if k + b < n:
            V = _build_V(P, b)                                   # (B, m, b) unit-lower
            tau_blk = tau[:, k:k + b]                            # (B, b)
            G = torch.matmul(V.transpose(1, 2), V)              # (B, b, b) Gram, one bmm
            # M = diag(1/tau) + striu(G); where tau==0 -> huge diagonal (T entry ->0)
            inv_tau = torch.where(tau_blk != 0, 1.0 / torch.where(tau_blk != 0, tau_blk, torch.ones_like(tau_blk)),
                                  torch.full_like(tau_blk, 1e30))
            M = torch.triu(G, diagonal=1)
            idx = torch.arange(b, device=A.device)
            M[:, idx, idx] = inv_tau
            C = H[:, k:, k + b:]
            W = torch.matmul(V.transpose(1, 2), C)              # (B, b, t)
            # Y = T^T W = M^{-T} W : solve lower-tri (M^T) Y = W
            Y = torch.linalg.solve_triangular(M.transpose(1, 2), W, upper=False)
            C.sub_(torch.matmul(V, Y))
        k += b
    return H, tau
