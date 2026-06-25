"""Detailed op-level bottleneck map of the CURRENT submission (gram + adaptive-ib),
via torch.profiler CUDA activity -> ground-truth per-kernel time (no forced-sync
artifacts). Tells us EXACTLY which op dominates n=512 b=640 and n=1024 b=60:
the panel Triton kernel, the fused tf32x3 GEMM, cuBLAS matmul, cuSOLVER
solve_triangular, or the elementwise glue (tril/clone/where/index_put/triu/sub).
Then aggregates kernels into categories. No banned substrings.
"""
import sys, re
sys.path.insert(0, "harness")
import torch
import reference as ref
import sub_profile as S
from torch.profiler import profile, ProfilerActivity

torch.backends.cuda.matmul.allow_tf32 = True


def categorize(name):
    n = name.lower()
    if "panel" in n: return "PANEL (triton)"
    if "bmm_x3" in n or "_x3" in n: return "tf32x3 GEMM (triton, n<=512)"
    if any(s in n for s in ("sgemm","gemm","cutlass","ampere","sm90","sm100","volta","tensorop","cublas")): return "cuBLAS matmul"
    if any(s in n for s in ("trsm","triangular","solve","trotri","cusolver","potrf","getrf","larf","ormqr","geqrf")): return "cuSOLVER (solve/geqrf)"
    if any(s in n for s in ("tril","triu","copy","clone","fill","where","index","sub","add","mul","elementwise","cast","stride","contiguous","zero","arange","cat")): return "elementwise glue"
    return "other:" + name[:40]


def run(batch, n, big, reps=10):
    S._BIG_X3 = big
    A = ref.generate_input(batch=batch, n=n, cond=2, seed=999, case="dense")
    for _ in range(5):  # warmup (JIT + autotune + sustained clocks)
        S.custom_kernel(A.clone())
    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        for _ in range(reps):
            S.custom_kernel(A.clone())
        torch.cuda.synchronize()
    evs = prof.key_averages()
    rows = []
    for e in evs:
        ct = getattr(e, "self_device_time_total", 0) or getattr(e, "self_cuda_time_total", 0)
        if ct > 0:
            rows.append((e.key, ct / reps))  # us per call
    rows.sort(key=lambda x: -x[1])
    total = sum(c for _, c in rows)
    print(f"\n================ n={n} b={batch} (BIG_X3={big})  total CUDA {total:.0f} us/call ================")
    cat = {}
    for k, c in rows:
        cat[categorize(k)] = cat.get(categorize(k), 0) + c
    print("--- by CATEGORY (us/call, % of CUDA time) ---")
    for k in sorted(cat, key=lambda x: -cat[x]):
        print(f"  {k:34s} {cat[k]:9.1f}  {100*cat[k]/total:5.1f}%")
    print("--- top 18 individual kernels ---")
    for k, c in rows[:18]:
        print(f"  {c:9.1f} us {100*c/total:5.1f}%  {k[:70]}")


def main():
    print("dev", torch.cuda.get_device_name(0), "torch", torch.__version__)
    run(640, 512, True)
    run(60, 1024, False)
    run(8, 2048, False)


if __name__ == "__main__":
    main()
