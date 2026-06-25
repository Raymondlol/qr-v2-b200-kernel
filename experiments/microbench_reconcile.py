"""RECONCILIATION: does the REAL submission.py (Triton panel + fused tf32x3 kernel)
give scaled_factor_residual ~0.05 (my eager harness) or ~10 (the documented 2.0x
margin) at mixed n=512 batch=640? Runs the real submission AND my eager emulated
tf32x3, on the SAME inputs, several seeds -> isolates Triton-kernel vs emulation.

If real ~= 0.05 -> the documented 'tf32x3 = 2.0x safe floor' is stale; huge
precision headroom exists; sub-tf32x3 (incl fp8) is wide open.
If real ~= 10   -> the real Triton kernels are noisier than emulation; trust ~10.

Run: modal run modal_microbench.py --script microbench_reconcile.py
"""
import sys, re
import torch
sys.path.insert(0, "/work"); sys.path.insert(0, "/work/harness")
import reference
import _real_submission as sub

torch.set_grad_enabled(False)


def sfr_of(data, output):
    good, msg = reference.check_implementation(data, output)
    m = re.search(r"scaled_factor_residual=([0-9.eE+-]+)", msg)
    sfr = float(m.group(1)) if m else float("nan")
    return good, sfr, msg


def main():
    print("torch", torch.__version__, "| dev", torch.cuda.get_device_name(0))
    print("=" * 80)
    print("REAL submission.py at the documented binding cases (worst-of-batch sfr, gate=20)")
    print("=" * 80)
    cases = [
        ("n512 mixed b640", 640, 512),
        ("n1024 mixed b60", 60, 1024),
    ]
    for seed in [32530, 1, 7, 2024, 99991]:
        for name, b, n in cases:
            data = reference.generate_input(batch=b, n=n, cond=2, seed=seed, case="mixed").cuda()
            # real submission (Triton panel + fused tf32x3 for n<=512 / 1xTF32 for n>=1024)
            try:
                H, tau = sub.custom_kernel(data.clone())
                good, sfr, _ = sfr_of(data, (H, tau))
                tag = "PASS" if good else "FAIL"
            except Exception as e:
                good, sfr, tag = False, float("nan"), "ERR:" + type(e).__name__
            mar = 20.0 / sfr if sfr > 0 else float("inf")
            print(f"  seed={seed:6d} {name:16s} REAL submission -> sfr={sfr:9.4g} "
                  f"margin={mar:8.2f} {tag}", flush=True)
    print()
    print("=" * 80)
    print("CROSS-CHECK: torch.geqrf (reference) sfr at the same cases (the gold floor)")
    print("=" * 80)
    for seed in [32530, 1]:
        for name, b, n in cases:
            data = reference.generate_input(batch=b, n=n, cond=2, seed=seed, case="mixed").cuda()
            H, tau = torch.geqrf(data)
            good, sfr, _ = sfr_of(data, (H, tau))
            print(f"  seed={seed:6d} {name:16s} geqrf -> sfr={sfr:9.4g} "
                  f"{'PASS' if good else 'FAIL'}", flush=True)


if __name__ == "__main__":
    main()
