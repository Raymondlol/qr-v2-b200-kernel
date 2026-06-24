"""Probe whether a CUSTOM CholeskyQR could be fast: isolate the two slow torch
primitives (cholesky 4742us, trsm 5822us for n=512 b=640) and test tensor-core
alternatives. The Gram (718us) and any A@X GEMM are already fast.

Tests for n=512 b=640 (median us):
  - Gram A^T A (tf32 GEMM)                  [known ~718]
  - torch.linalg.cholesky (cuSOLVER)        [known ~4742]
  - torch trsm  Q = A R^-1                   [known ~5822]
  - trsm-as-inverse+GEMM:  Rinv (trsm vs I) then A@Rinv (tf32x3-ish GEMM)
  - A @ Rinv  GEMM-only  (the 'apply' cost if we had Rinv cheaply)
  - recursive/blocked cholesky in pure torch (does blocking beat cuSOLVER?)
"""
import torch
torch.backends.cuda.matmul.allow_tf32 = True


def clear_l2():
    torch.empty((32, 1024, 1024), dtype=torch.int64, device="cuda").fill_(0)


def t(fn, iters=40):
    for _ in range(5):
        try: fn()
        except Exception as e: return f"ERR:{type(e).__name__}"
    torch.cuda.synchronize(); ts = []
    for _ in range(iters):
        clear_l2(); torch.cuda.synchronize()
        s = torch.cuda.Event(enable_timing=True); e = torch.cuda.Event(enable_timing=True)
        s.record(); fn(); e.record(); torch.cuda.synchronize(); ts.append(s.elapsed_time(e))
    ts.sort(); return f"{ts[len(ts)//2]*1000:9.1f}"


def blocked_chol(G, bs=128):
    # right-looking blocked Cholesky (trailing update = GEMM = tensor-core)
    B, n, _ = G.shape
    R = G.clone()
    for k in range(0, n, bs):
        e = min(k + bs, n)
        Rkk = torch.linalg.cholesky(R[:, k:e, k:e], upper=True)
        R[:, k:e, k:e] = Rkk
        if e < n:
            # R[k:e, e:] = Rkk^-T @ G[k:e, e:]
            R[:, k:e, e:] = torch.linalg.solve_triangular(
                Rkk.transpose(-1, -2), R[:, k:e, e:], upper=False)
            # trailing -= R[k:e,e:]^T @ R[k:e,e:]  (GEMM)
            R[:, e:, e:] -= R[:, k:e, e:].transpose(-1, -2) @ R[:, k:e, e:]
    return torch.triu(R)


def main():
    print("dev", torch.cuda.get_device_name(0))
    B, n = 640, 512
    A = torch.randn(B, n, n, device="cuda")
    eye = torch.eye(n, device="cuda").expand(B, n, n).contiguous()
    shift = n * 1e-3
    G = A.transpose(-1, -2) @ A + shift * eye
    R = torch.linalg.cholesky(G, upper=True)

    print(f"\nn=512 b=640 (median us):")
    print(f"  Gram A^T A                : {t(lambda: A.transpose(-1,-2)@A)}")
    print(f"  torch.cholesky            : {t(lambda: torch.linalg.cholesky(G, upper=True))}")
    print(f"  torch trsm  A R^-1         : {t(lambda: torch.linalg.solve_triangular(R, A, upper=True, left=False))}")
    print(f"  Rinv = trsm(R, I)          : {t(lambda: torch.linalg.solve_triangular(R, eye, upper=True))}")
    Rinv = torch.linalg.solve_triangular(R, eye, upper=True)
    print(f"  A @ Rinv  (GEMM only)      : {t(lambda: A @ Rinv)}")
    print(f"  blocked_chol bs=128        : {t(lambda: blocked_chol(G, 128))}")
    print(f"  blocked_chol bs=64         : {t(lambda: blocked_chol(G, 64))}")

    # what a custom CholeskyQR2 *could* cost if trsm = Rinv+GEMM and chol stays torch:
    def cqr2_inv():
        G1 = A.transpose(-1,-2)@A + shift*eye
        R1 = torch.linalg.cholesky(G1, upper=True)
        Q1 = A @ torch.linalg.solve_triangular(R1, eye, upper=True)
        G2 = Q1.transpose(-1,-2)@Q1 + shift*eye
        R2 = torch.linalg.cholesky(G2, upper=True)
        Q = Q1 @ torch.linalg.solve_triangular(R2, eye, upper=True)
        return Q, R2@R1
    print(f"  CQR2 (trsm->inv+GEMM)      : {t(cqr2_inv)}   <- vs naive-trsm CQR2 ~22000, our Householder ~14000")


if __name__ == "__main__":
    main()
