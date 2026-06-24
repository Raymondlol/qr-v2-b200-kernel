"""Decisive speed test for the CholeskyQR hypothesis. Add a shift so Cholesky never
fails, and break the time into Gram / Cholesky / triangular-solve so we see whether
the batched CHOLESKY itself batches well (or is geqrf-slow). Compare to our ~14ms
(n=512) / ~10ms (n=1024) Householder path.
"""
import torch
torch.backends.cuda.matmul.allow_tf32 = True


def clear_l2():
    torch.empty((32, 1024, 1024), dtype=torch.int64, device="cuda").fill_(0)


def time_fn(fn, iters=50):
    for _ in range(5): fn()
    torch.cuda.synchronize()
    ts = []
    for _ in range(iters):
        clear_l2(); torch.cuda.synchronize()
        s = torch.cuda.Event(enable_timing=True); e = torch.cuda.Event(enable_timing=True)
        s.record(); fn(); e.record(); torch.cuda.synchronize()
        ts.append(s.elapsed_time(e))
    ts.sort(); return ts[len(ts)//2] * 1000


def main():
    print("dev", torch.cuda.get_device_name(0), "| tf32", torch.backends.cuda.matmul.allow_tf32)
    print(f"\n{'shape':14s} {'gram':>8s} {'chol':>8s} {'trsm':>8s} {'CQR2 total':>11s}  (median us)")
    for (B, n) in [(640, 512), (60, 1024), (8, 2048), (2, 4096)]:
        A = torch.randn(B, n, n, device="cuda")
        eye = torch.eye(n, device="cuda")
        shift = (n * 1e-3)  # keep Gram safely PD for the timing
        def gram(): return A.transpose(-1, -2) @ A + shift * eye
        G = gram()
        def chol(): return torch.linalg.cholesky(G, upper=True)
        R = chol()
        def trsm(): return torch.linalg.solve_triangular(R, A, upper=True, left=False)
        def cqr2():
            G1 = A.transpose(-1, -2) @ A + shift * eye
            R1 = torch.linalg.cholesky(G1, upper=True)
            Q1 = torch.linalg.solve_triangular(R1, A, upper=True, left=False)
            G2 = Q1.transpose(-1, -2) @ Q1 + shift * eye
            R2 = torch.linalg.cholesky(G2, upper=True)
            Q = torch.linalg.solve_triangular(R2, Q1, upper=True, left=False)
            return Q, R2 @ R1
        tg = time_fn(gram); tc = time_fn(chol); tt = time_fn(trsm); ttot = time_fn(cqr2)
        print(f"b={B:4d} n={n:5d} {tg:8.1f} {tc:8.1f} {tt:8.1f} {ttot:11.1f}")

    # also: geqrf reference (what we currently route large-n to) for n=2048/4096
    print("\n--- torch.geqrf (reference, for comparison) ---")
    for (B, n) in [(640, 512), (8, 2048), (2, 4096)]:
        A = torch.randn(B, n, n, device="cuda")
        print(f"geqrf b={B:4d} n={n:5d}: {time_fn(lambda: torch.geqrf(A), iters=20):10.1f} us")


if __name__ == "__main__":
    main()
