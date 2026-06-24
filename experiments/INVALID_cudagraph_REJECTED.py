# Batched square Householder QR (compact-WY) with CUDA-graph capture.
# Robustness-first numerics (unchanged from the verified FP32 version); the graph
# wrapper targets the diagnosed bottleneck: per-column CPU dispatch in the panel loop.
#
# Diagnosis recap (from leaderboard run): runtime was ~constant 0.21 ms/column,
# independent of batch and conditioning -> dispatch-bound, not FLOP-bound. CUDA
# graphs collapse the ~10*n eager launches per call into a single replay.
# Expectation: a few-x speedup (CPU dispatch removed), but the ~10*n small GPU
# kernels remain -> this does NOT reach the leaderboard band. It is a clean
# one-variable test of "is dispatch the bottleneck?". The real fix is fusing the
# panel into one kernel (Triton).

import torch

try:
    from task import input_t, output_t
except Exception:
    input_t = output_t = object


# ---------------- numerics (identical math to the verified version) ----------------

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
        vtail = torch.where(mask.unsqueeze(1), x / safe_denom.unsqueeze(1),
                            torch.zeros_like(x))
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


def _form_T(V, tau):
    B, m, b = V.shape
    T = torch.zeros(B, b, b, dtype=V.dtype, device=V.device)
    T[:, 0, 0] = tau[:, 0]
    for i in range(1, b):
        Vi = V[:, :, i]
        Vprev = V[:, :, 0:i]
        t = -tau[:, i:i + 1] * torch.einsum('bmi,bm->bi', Vprev, Vi)
        T[:, 0:i, i] = torch.einsum('bij,bj->bi', T[:, 0:i, 0:i], t)
        T[:, i, i] = tau[:, i]
    return T


def _block_size(n):
    if n <= 256:
        return 16
    if n <= 1024:
        return 32
    return 64


def _factor_into(A_src, H_buf, tau_buf, block):
    # Reads A_src, writes the (H, tau) compact factor into H_buf / tau_buf in place.
    # Fully shape-static given n and block -> CUDA-graph capturable.
    H_buf.copy_(A_src)
    tau_buf.zero_()
    n = H_buf.shape[1]
    k = 0
    while k < n:
        b = min(block, n - k)
        P = H_buf[:, k:, k:k + b]
        _panel_factor(P, tau_buf[:, k:k + b])
        if k + b < n:
            V = _build_V(P, b)
            T = _form_T(V, tau_buf[:, k:k + b])
            C = H_buf[:, k:, k + b:]
            VtC = torch.matmul(V.transpose(1, 2), C)
            TtVtC = torch.matmul(T.transpose(1, 2), VtC)
            C.sub_(torch.matmul(V, TtVtC))
        k += b


# ---------------- CUDA-graph cache (one graph per (B, n) shape) ----------------

class _ShapeGraph:
    def __init__(self, A_like, block):
        B, n, _ = A_like.shape
        dev, dt = A_like.device, A_like.dtype
        self.static_A = torch.empty(B, n, n, dtype=dt, device=dev)
        self.static_H = torch.empty(B, n, n, dtype=dt, device=dev)
        self.static_tau = torch.empty(B, n, dtype=dt, device=dev)
        self.block = block
        # Warmup on the DEFAULT stream (no user-created stream -> platform-safe).
        # This initializes cuBLAS handles / lazy state before capture.
        for _ in range(3):
            _factor_into(self.static_A, self.static_H, self.static_tau, block)
        torch.cuda.synchronize()
        # Capture. torch.cuda.graph() manages capture internally; we create no stream.
        self.graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph):
            _factor_into(self.static_A, self.static_H, self.static_tau, block)

    def run(self, A):
        self.static_A.copy_(A)           # eager: 1 launch
        self.graph.replay()              # eager: 1 launch (replays the whole factorization)
        return self.static_H.clone(), self.static_tau.clone()


_CACHE = {}


def custom_kernel(data: input_t) -> output_t:
    A = data
    B, n, _ = A.shape
    block = _block_size(n)
    if not A.is_cuda:                    # eager fallback (also the CPU correctness path)
        H = torch.empty_like(A)
        tau = torch.empty(B, n, dtype=A.dtype, device=A.device)
        _factor_into(A, H, tau, block)
        return H, tau
    key = (B, n, A.dtype)
    if key not in _CACHE:
        try:
            _CACHE[key] = _ShapeGraph(A, block)   # warmup + capture (once per shape)
        except Exception:
            _CACHE[key] = None                    # capture failed -> degrade to eager
    g = _CACHE[key]
    if g is None:
        H = torch.empty_like(A)
        tau = torch.empty(B, n, dtype=A.dtype, device=A.device)
        _factor_into(A, H, tau, block)
        return H, tau
    return g.run(A)