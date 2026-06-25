"""Characterize the reseed-DQ risk of cand_solvefix_1x (1xTF32 trailing + fp32 solve)
at the binding mixed@640 case across many seeds. Gate = worst-of-640 sfr < 20.
If max sfr across seeds stays well below 20, the +7.1% geomean is a defensible
aggressive submission; if it spikes near/over 20, it's too risky.

Run: modal run modal_microbench.py --script microbench_1x_reseed.py
"""
import sys, re
import torch
sys.path.insert(0, "/work"); sys.path.insert(0, "/work/harness")
import reference
import cand_solvefix_1x as c1x
import cand_solvefix as csafe

torch.set_grad_enabled(False)

def sfr_of(data, out):
    good, msg = reference.check_implementation(data, out)
    m = re.search(r"scaled_factor_residual=([0-9.eE+-]+)", msg)
    return good, (float(m.group(1)) if m else float("nan"))

def sweep(mod, label):
    print(f"\n--- {label} ---")
    print(f"{'case':16s} " + " ".join(f"s{i}" for i in range(12)) + "   MAX  min-margin")
    for n, b, cs in [(512, 640, "mixed"), (512, 640, "rankdef"), (512, 640, "clustered"),
                     (1024, 60, "mixed")]:
        sfrs = []
        for i in range(12):
            seed = 100000 + i * 37 + (0 if cs == "mixed" else 1000)
            cond = 2 if cs in ("mixed",) else 0
            data = reference.generate_input(batch=b, n=n, cond=cond, seed=seed, case=cs).cuda()
            try:
                H, tau = mod.custom_kernel(data.clone())
                good, sfr = sfr_of(data, (H, tau))
                sfrs.append(sfr if good else float("nan"))
            except Exception as e:
                sfrs.append(float("nan"))
        mx = max([x for x in sfrs if x == x], default=float("nan"))
        mar = 20.0 / mx if mx == mx and mx > 0 else float("nan")
        cells = " ".join(f"{x:4.1f}" if x == x else " nan" for x in sfrs)
        flag = "SAFE" if mar >= 2.0 else ("RISKY" if mar >= 1.0 else "FAIL")
        print(f"{cs+f' n{n}':16s} {cells}   {mx:5.1f}  m{mar:5.2f} {flag}", flush=True)

def main():
    print("torch", torch.__version__, "| dev", torch.cuda.get_device_name(0))
    print("=" * 96)
    print("Reseed-risk: worst-of-batch sfr across 12 seeds. gate=20. SAFE>=2x margin, RISKY 1-2x, FAIL<1x")
    print("=" * 96)
    sweep(c1x, "cand_solvefix_1x (1xTF32 trailing + fp32 solve) — the +7.1% candidate")
    sweep(csafe, "cand_solvefix (tf32x3 + fp32 solve) — the safe reference")

if __name__ == "__main__":
    main()
