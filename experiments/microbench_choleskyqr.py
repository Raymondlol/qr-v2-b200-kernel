"""Test the CholeskyQR hypothesis: is a GEMM-heavy CholeskyQR2 ~10x faster than our
Householder path on n=512 b=640, and which stress cases survive numerically?

Speed: CholeskyQR2 is all batched GEMM + batched Cholesky + triangular solve (tensor
cores, fills GPU). Compare to our ~14ms. Correctness here only checks Q,R quality
(||QtQ-I||, ||A-QR||/||A||) -- the flat (H,tau) reconstruction is a separate step.
"""
import sys, torch
sys.path.insert(0, "harness")
import reference

torch.backends.cuda.matmul.allow_tf32 = True


def choleskyqr2(A, gram_dtype=torch.float32):
    # CholeskyQR2: two CholeskyQR passes (orthogonality ~eps if cond(A) < ~1/sqrt(eps)).
    G = (A.transpose(-1, -2).to(gram_dtype) @ A.to(gram_dtype)).to(torch.float32)
    R1 = torch.linalg.cholesky(G, upper=True)
    Q1 = torch.linalg.solve_triangular(R1, A, upper=True, left=False)
    G2 = Q1.transpose(-1, -2) @ Q1
    R2 = torch.linalg.cholesky(G2, upper=True)
    Q = torch.linalg.solve_triangular(R2, Q1, upper=True, left=False)
    R = R2 @ R1
    return Q, R


def clear_l2():
    torch.empty((32, 1024, 1024), dtype=torch.int64, device="cuda").fill_(0)


def time_fn(fn, A, iters=60):
    for _ in range(5):
        try: fn(A)
        except Exception: return float('nan')
    torch.cuda.synchronize()
    ts = []
    for _ in range(iters):
        clear_l2(); torch.cuda.synchronize()
        s = torch.cuda.Event(enable_timing=True); e = torch.cuda.Event(enable_timing=True)
        s.record(); fn(A); e.record(); torch.cuda.synchronize()
        ts.append(s.elapsed_time(e))
    ts.sort(); return ts[len(ts)//2] * 1000  # median us


def main():
    print("dev", torch.cuda.get_device_name(0))
    # --- SPEED: n=512 b=640 random (the dominant case; our Householder ~14ms) ---
    print("\n=== SPEED (median us) ===")
    for (B, n) in [(640, 512), (60, 1024), (8, 2048), (2, 4096)]:
        A = torch.randn(B, n, n, device="cuda")
        t = time_fn(choleskyqr2, A)
        print(f"CholeskyQR2  b={B:4d} n={n:5d}:  {t:10.1f} us")

    # --- NUMERICS: which stress cases survive CholeskyQR2? ---
    print("\n=== NUMERICS (n=512, b=32) : does chol succeed? ||QtQ-I||, ||A-QR||/||A|| ===")
    cases = [
        dict(batch=32, n=512, cond=2, seed=1, case="dense"),
        dict(batch=32, n=512, cond=4, seed=2, case="dense"),
        dict(batch=32, n=512, cond=0, seed=3, case="rankdef"),
        dict(batch=32, n=512, cond=0, seed=4, case="clustered"),
        dict(batch=32, n=512, cond=0, seed=5, case="nearcollinear"),
        dict(batch=32, n=512, cond=0, seed=6, case="band"),
        dict(batch=32, n=512, cond=2, seed=7, case="mixed"),
    ]
    n = 512
    factor_gate = 20 * n * torch.finfo(torch.float32).eps
    orth_gate = 100 * n * torch.finfo(torch.float32).eps
    print(f"gates: factor_rtol={factor_gate:.2e}  orth_rtol={orth_gate:.2e}")
    for tc in cases:
        A = reference.generate_input(**tc)
        for gd, tag in [(torch.float32, "g32"), (torch.float64, "g64")]:
            try:
                Q, R = choleskyqr2(A, gram_dtype=gd)
                Ad, Qd, Rd = A.double(), Q.double(), R.double()
                I = torch.eye(n, device=A.device, dtype=torch.float64)
                orth = torch.linalg.matrix_norm(Qd.transpose(-1,-2)@Qd - I, ord=1, dim=(-2,-1)).amax()
                recon = torch.linalg.matrix_norm(Qd@Rd - Ad, ord=1, dim=(-2,-1)).amax()
                anorm = torch.linalg.matrix_norm(Ad, ord=1, dim=(-2,-1)).amax()
                o = (orth).item(); r = (recon/anorm.clamp_min(1e-30)).item()
                ok = "PASS" if (o < orth_gate and r < factor_gate) else "FAIL"
                print(f"  {tc['case']:14s} {tag}: {ok}  orth={o:.2e}  recon/||A||={r:.2e}")
            except Exception as ex:
                print(f"  {tc['case']:14s} {tag}: CHOL FAILED ({type(ex).__name__})")


if __name__ == "__main__":
    main()
