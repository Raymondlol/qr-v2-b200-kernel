"""Decisive B200 test of the CholeskyQR floor-breaker for the DENSE large-n cases
(n=2048 b=8, n=4096 b=2) where torch.geqrf collapses (only 2-8 panels on 148 SMs).

Measures the FULL competition-valid path end to end:
    Gram (fp32 or tf32x3)  ->  shifted CholeskyQR (2 or 3 passes)
                            ->  Householder reconstruction to flat (H,tau)
and compares total to torch.geqrf, verifying the output passes the OFFICIAL
harness/reference.py check_implementation (factor + orthogonality residuals).

KEY EMPIRICAL FINDING (CPU, reference.generate_input case='dense', cond=1):
  These "dense" inputs are NOT uniformly well-conditioned. Square randn has
  cond(A) growing ~n with a heavy tail: measured max cond(A) ~ 8.1e5 at n=2048,
  ~9.2e4 at n=4096. Gram SQUARES it: cond(G) ~ 6.5e11 / 8.5e9. So:
    * plain TF32 Gram (10-bit) loses PD -> torch.linalg.cholesky _LinAlgError
      (the original microbench nan bug).
    * even an EXACT fp32 Gram is non-PD at these cond(G): UNSHIFTED CholeskyQR
      raises _LinAlgError at n>=256 on the bad seeds. A Demmel shift
      s = 11*n*eps*max_diag(G) is REQUIRED on pass 1.
    * tf32x3 Gram needs 3 passes AND still failed at n=1024 in CPU tests; fp32
      Gram + shifted CQR2/CQR3 is the only path that reliably PASSES. So the
      reviewer's "dense => no shift / no CQR3 needed" premise is FALSE here, and
      the cheap tf32x3-CQR2 flop estimate is NOT the safe configuration.

RECONSTRUCTION (Q,R -> flat H,tau), verified vs reference.check_implementation:
  Q from CholeskyQR is orthonormal with A = Q @ R. geqrf(Q) yields strict-lower
  reflectors + tau that factor Qhat = Q @ diag(s), s = sign(diag(triu(geqrf(Q)))).
  Sign-correct R as Rout = diag(s) @ R so householder_product(H,tau) @ triu(H)=A.
  This geqrf(Q) recon is HONEST and drop-in correct, but it is the SAME geqrf
  primitive, so it is the PESSIMISTIC upper bound on reconstruction cost (a custom
  GEMM-trailing unpivoted GETRFNP / ORHR_COL would be ~n^3/3, but needs a
  hand-written kernel). If geqrf(Q) recon alone is as slow as geqrf(A), the
  floor-breaker is dead without that extra custom kernel -- the script exposes it.
"""
import sys
sys.path.insert(0, "harness")
import torch
import reference as ref


def clear_l2():
    torch.empty((32, 1024, 1024), dtype=torch.int64, device="cuda").fill_(0)


def time_fn(fn, iters):
    for _ in range(3):
        fn()
    torch.cuda.synchronize()
    ts = []
    for _ in range(iters):
        clear_l2()
        torch.cuda.synchronize()
        s = torch.cuda.Event(enable_timing=True)
        e = torch.cuda.Event(enable_timing=True)
        s.record(); fn(); e.record()
        torch.cuda.synchronize()
        ts.append(s.elapsed_time(e))
    ts.sort()
    return ts[len(ts) // 2] * 1000.0  # median, microseconds


def gram_fp32(X):
    prev = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False  # exact fp32, never plain TF32
    G = X.transpose(-1, -2) @ X
    torch.backends.cuda.matmul.allow_tf32 = prev
    return G


def gram_tf32x3(X):
    # 3-pass split-precision A^T A (~fp32 accuracy) on tensor cores.
    hi = (X * 4096.0).round() / 4096.0
    lo = X - hi
    prev = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = True
    G = (hi.transpose(-1, -2) @ hi
         + hi.transpose(-1, -2) @ lo
         + lo.transpose(-1, -2) @ hi)
    torch.backends.cuda.matmul.allow_tf32 = prev
    return G


def shifted_cqr(A, gram, npass):
    # Shifted CholeskyQR: pass 1 shifted (Demmel) to guarantee PD at high cond(G),
    # remaining passes plain to drive orthogonality to ~eps. Returns (Q, R_full).
    n = A.shape[-1]
    eye = torch.eye(n, device=A.device, dtype=A.dtype)
    eps = torch.finfo(A.dtype).eps
    Q, Racc = A, None
    for p in range(npass):
        G = gram(Q)
        if p == 0:
            tr = torch.diagonal(G, dim1=-2, dim2=-1).amax(dim=-1)
            G = G + (11.0 * n * eps * tr).view(-1, 1, 1) * eye
        R = torch.linalg.cholesky(G, upper=True)
        Q = torch.linalg.solve_triangular(R, Q, upper=True, left=False)
        Racc = R if Racc is None else R @ Racc
    return Q, Racc


def reconstruct(Q, R):
    Hq, tau = torch.geqrf(Q)
    s = torch.sign(torch.diagonal(torch.triu(Hq), dim1=-2, dim2=-1))
    s = torch.where(s == 0, torch.ones_like(s), s)
    H = torch.tril(Hq, diagonal=-1) + torch.triu(s.unsqueeze(-1) * R)
    return H, tau


def full_path(A, gram, npass):
    return reconstruct(*shifted_cqr(A, gram, npass))


def run(batch, n, seed, iters, gram, npass, label):
    A = ref.generate_input(batch=batch, n=n, cond=1, seed=seed, case="dense")
    try:
        H, tau = full_path(A, gram, npass)
        ok, msg = ref.check_implementation(A, (H, tau))
    except Exception as exc:
        ok, msg = False, f"EXC {type(exc).__name__}"
    sf = msg.split("scaled_factor_residual=")[1].split(";")[0] if "scaled_factor_residual=" in msg else "-"
    so = msg.split("scaled_orthogonality_residual=")[1].split(";")[0] if "scaled_orthogonality_residual=" in msg else "-"

    print(f"\n=== b={batch} n={n}  [{label} Gram, shifted CQR{npass}] ===")
    print(f"  check_implementation : {'PASS' if ok else 'FAIL'}  factor={sf} orth={so}")
    if not ok and msg.startswith("EXC"):
        print(f"  (skipping timing: {msg})")
        return
    t_cqr = time_fn(lambda: shifted_cqr(A, gram, npass), iters)
    t_recon = time_fn(lambda: reconstruct(*shifted_cqr(A, gram, npass)), iters) - t_cqr
    t_total = time_fn(lambda: full_path(A, gram, npass), iters)
    t_geqrf = time_fn(lambda: torch.geqrf(A), iters)
    print(f"  CQR{npass}                 : {t_cqr:10.1f} us")
    print(f"  reconstruction       : {t_recon:10.1f} us")
    print(f"  CholeskyQR-path TOTAL: {t_total:10.1f} us")
    print(f"  torch.geqrf          : {t_geqrf:10.1f} us")
    print(f"  speedup vs geqrf     : {t_geqrf / t_total:9.2f}x")


def main():
    print("dev", torch.cuda.get_device_name(0))
    print("Safe path = fp32 Gram + shifted CQR3 (PASSES). tf32x3/CQR2 shown for "
          "speed-vs-safety; geqrf(Q) recon is the pessimistic upper bound.")
    for label, gram, npass in (("fp32", gram_fp32, 3),
                               ("fp32", gram_fp32, 2),
                               ("tf32x3", gram_tf32x3, 3)):
        run(8, 2048, seed=1, iters=20, gram=gram, npass=npass, label=label)
        run(2, 4096, seed=2, iters=15, gram=gram, npass=npass, label=label)


if __name__ == "__main__":
    main()
