"""Validate a candidate: (1) correctness on all 12 benchmark shapes (good + gate),
(2) mixed@640 / rankdef / clustered / n1024 worst-of-batch margin across seeds.
Edit CAND below. Run: modal run modal_microbench.py --script microbench_validate.py
"""
import sys, re
import torch
sys.path.insert(0, "/work"); sys.path.insert(0, "/work/harness")
import reference

CAND = "cand_fp16x3_tsolvecol"
mod = __import__(CAND)
torch.set_grad_enabled(False)

def sfr_of(data, out):
    good, msg = reference.check_implementation(data, out)
    m = re.search(r"scaled_factor_residual=([0-9.eE+-]+)", msg)
    return good, (float(m.group(1)) if m else float("nan")), msg

# all 12 benchmark shapes (gpu_bench)
BENCH = [
    (20,32,1,"dense",43214),(40,176,1,"dense",423011),(40,352,1,"dense",123456),
    (640,512,2,"dense",1029),(60,1024,2,"dense",75342),(8,2048,1,"dense",224466),
    (2,4096,1,"dense",32412),(640,512,2,"mixed",770001),(60,1024,2,"mixed",770002),
    (640,512,0,"rankdef",770003),(640,512,0,"clustered",770004),(60,1024,0,"nearrank",770005),
]

def main():
    print("torch", torch.__version__, "| dev", torch.cuda.get_device_name(0), "| CAND =", CAND)
    print("=" * 90)
    print("(1) CORRECTNESS on 12 benchmark shapes (good=checker pass, sfr=worst-of-batch)")
    print("=" * 90)
    npass = 0
    for b, n, cond, case, seed in BENCH:
        data = reference.generate_input(batch=b, n=n, cond=cond, seed=seed, case=case).cuda()
        try:
            H, tau = mod.custom_kernel(data.clone())
            good, sfr, msg = sfr_of(data, (H, tau))
            npass += int(good)
            mar = 20.0/sfr if (sfr==sfr and sfr>0) else float('inf')
            print(f"  {'OK ' if good else 'BAD'} b{b:<4d} n{n:<5d} {case:10s} sfr={sfr:8.3g} m={mar:7.1f}"
                  + ("" if good else "  <-- "+msg[:60]), flush=True)
        except Exception as e:
            print(f"  ERR b{b} n{n} {case}: {type(e).__name__}: {str(e)[:60]}", flush=True)
    print(f"  => {npass}/12 benchmark shapes pass")
    print()
    print("=" * 90)
    print("(2) RESEED MARGIN at the binding cases (worst-of-batch sfr, 8 seeds; gate=20, SAFE>=2x)")
    print("=" * 90)
    for n, b, cond, case in [(512,640,2,"mixed"),(512,640,0,"rankdef"),(512,640,0,"clustered"),(1024,60,2,"mixed")]:
        sfrs=[]
        for i in range(8):
            seed=200000+i*53
            data=reference.generate_input(batch=b,n=n,cond=cond,seed=seed,case=case).cuda()
            try:
                H,tau=mod.custom_kernel(data.clone()); good,sfr,_=sfr_of(data,(H,tau))
                sfrs.append(sfr if good else float('nan'))
            except Exception: sfrs.append(float('nan'))
        mx=max([x for x in sfrs if x==x],default=float('nan'))
        mar=20.0/mx if (mx==mx and mx>0) else float('nan')
        flag="SAFE" if mar>=2.0 else ("RISKY" if mar>=1.0 else "FAIL")
        print(f"  {case+f' n{n}':18s} sfr=[{' '.join(f'{x:.1f}' for x in sfrs)}] MAX={mx:.2f} m={mar:.2f} {flag}",flush=True)

if __name__ == "__main__":
    main()
