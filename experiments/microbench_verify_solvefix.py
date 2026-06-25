"""VERIFY the solve-fix win across the precision-stress benchmark cases (real
pipeline). The attribution proved the n=512 mixed@640 1.9x 'floor' was a tf32
triangular-solve artifact. Now check, on mixed/rankdef/clustered/nearrank at the
binding shapes:
  A. baseline (real: tf32x3 n<=512 / 1xTF32 n>=1024, tf32 solve)
  B. solve->fp32, keep tf32x3 trailing (n<=512)            [safety: no regress, unlock margin]
  C. solve->fp32 + 1xTF32 trailing EVERYWHERE (_BIG_X3=False) [SPEED: 1267us->139us trailing]
If C holds >=2x on all stress cases, 1xTF32 trailing + fp32 solve is a big deployable win.

Run: modal run modal_microbench.py --script microbench_verify_solvefix.py
"""
import sys, re
import torch
sys.path.insert(0, "/work"); sys.path.insert(0, "/work/harness")
import reference
import _real_submission as sub

torch.set_grad_enabled(False)
_orig_solve = torch.linalg.solve_triangular

def solve_fp32(*a, **k):
    prev = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False
    try: return _orig_solve(*a, **k)
    finally: torch.backends.cuda.matmul.allow_tf32 = prev

def sfr_of(data, out):
    good, msg = reference.check_implementation(data, out)
    m = re.search(r"scaled_factor_residual=([0-9.eE+-]+)", msg)
    return good, (float(m.group(1)) if m else float("nan")), msg

# the 5 precision-stress benchmark cases at the binding shapes (gpu_bench seeds)
CASES = [
    (640, 512, 2, "mixed",    770001),
    (640, 512, 0, "rankdef",  770003),
    (640, 512, 0, "clustered",770004),
    (60, 1024, 2, "mixed",    770002),
    (60, 1024, 0, "nearrank", 770005),
]

def factor(data, n, big_x3, solve):
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.linalg.solve_triangular = solve
    sub._BIG_X3 = big_x3
    return sub._factor_custom(data.clone())

def main():
    print("torch", torch.__version__, "| dev", torch.cuda.get_device_name(0))
    print("=" * 96)
    print("Real pipeline. margin=20/sfr (worst-of-batch). SAFE=margin>=2.0. gate fails if margin<1.")
    print("=" * 96)
    variants = [
        ("A baseline (tf32x3<=512, tf32 solve)", lambda n: (n <= 512), _orig_solve),
        ("B solve->fp32 (tf32x3 kept)",          lambda n: (n <= 512), solve_fp32),
        ("C solve->fp32 + 1xTF32 trailing ALL",  lambda n: False,      solve_fp32),
    ]
    hdr = f"{'case':26s}"
    for vn, _, _ in variants:
        hdr += f" | {vn[:22]:>22s}"
    print(hdr)
    for b, n, cond, case, seed in CASES:
        data = reference.generate_input(batch=b, n=n, cond=cond, seed=seed, case=case).cuda()
        row = f"{case+f' n{n} b{b}':26s}"
        for vn, big_fn, solve in variants:
            try:
                H, tau = factor(data, n, big_fn(n), solve)
                good, sfr, _ = sfr_of(data, (H, tau))
                mar = 20.0 / sfr if (good and sfr > 0) else float("nan")
                flag = "SAFE" if (mar == mar and mar >= 2.0) else ("pass" if (good and sfr < 20) else "FAIL")
                row += f" | {sfr:8.3g} m{mar:6.1f} {flag:4s}"
            except Exception as e:
                row += f" | ERR {type(e).__name__:17s}"
        print(row, flush=True)

if __name__ == "__main__":
    main()
