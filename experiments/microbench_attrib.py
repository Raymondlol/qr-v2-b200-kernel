"""ATTRIBUTION: the real submission gives mixed@640 sfr~10.7 (margin 1.9x) with
allow_tf32=True, but ~0.025 (margin 800x) with allow_tf32=False. Same Triton
kernels. The only allow_tf32-gated torch-native op in the n<=512 path is
torch.linalg.solve_triangular (Y = solve(M^T, W)). If forcing JUST that solve to
fp32 recovers ~0.025, then the documented 'tf32x3 = 1.9x safe floor' is actually a
TF32 TRIANGULAR SOLVE artifact -- and fixing it (cheap; the solve is small) unlocks
huge precision headroom, making sub-tf32 trailing (1xTF32 / fp8) trivially safe.

Run: modal run modal_microbench.py --script microbench_attrib.py
"""
import sys, re
import torch
sys.path.insert(0, "/work"); sys.path.insert(0, "/work/harness")
import reference
import _real_submission as sub

torch.set_grad_enabled(False)
_orig_solve = torch.linalg.solve_triangular

def sfr_of(data, out):
    good, msg = reference.check_implementation(data, out)
    m = re.search(r"scaled_factor_residual=([0-9.eE+-]+)", msg)
    return good, (float(m.group(1)) if m else float("nan"))

def measure(b, n, seed):
    data = reference.generate_input(batch=b, n=n, cond=2, seed=seed, case="mixed").cuda()
    H, tau = sub.custom_kernel(data.clone())
    return sfr_of(data, (H, tau))

def solve_fp32(*a, **k):
    # force fp32 accumulation inside the triangular solve only
    prev = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False
    try:
        return _orig_solve(*a, **k)
    finally:
        torch.backends.cuda.matmul.allow_tf32 = prev

def solve_fp64(A, B, **k):
    return _orig_solve(A.double(), B.double(), **k).to(B.dtype)

def main():
    print("torch", torch.__version__, "| dev", torch.cuda.get_device_name(0))
    print("=" * 84)
    print("ATTRIBUTION of the mixed@640 ~1.9x floor (real submission.py custom_kernel)")
    print("=" * 84)
    seeds = [32530, 1, 2024]
    cases = [(512, 640, "n512 b640"), (1024, 60, "n1024 b60")]

    def run_variant(name, setup, teardown):
        for n, b, lab in cases:
            sfrs = []
            for s in seeds:
                setup()
                try:
                    good, sfr = measure(b, n, s)
                    sfrs.append(sfr if good else float("nan"))
                except Exception as e:
                    sfrs.append(float("nan"))
                teardown()
            worst = max([x for x in sfrs if x == x], default=float("nan"))
            mar = 20.0 / worst if worst == worst and worst > 0 else float("nan")
            cells = " ".join(f"{x:8.3g}" for x in sfrs)
            print(f"{name:42s} {lab:11s} {cells}  margin={mar:8.2f}", flush=True)

    def noop(): pass
    # A: real default (allow_tf32=True, real solve)
    def setA():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.linalg.solve_triangular = _orig_solve
    run_variant("A. real (tf32 on, real solve) [=10.7?]", setA, noop)
    print()
    # B: allow_tf32=False everywhere
    def setB():
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.linalg.solve_triangular = _orig_solve
    run_variant("B. allow_tf32=False all [=0.025?]", setB, noop)
    print()
    # C: tf32 ON globally, but solve forced fp32  <-- THE attribution test
    def setC():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.linalg.solve_triangular = solve_fp32
    run_variant("C. tf32 ON, solve->fp32 (attribution)", setC, noop)
    print()
    # D: tf32 ON globally, solve forced fp64
    def setD():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.linalg.solve_triangular = solve_fp64
    run_variant("D. tf32 ON, solve->fp64", setD, noop)

if __name__ == "__main__":
    main()
