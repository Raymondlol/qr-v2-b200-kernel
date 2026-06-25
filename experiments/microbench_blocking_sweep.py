"""Two-level tuning: re-sweep blocking (NB, ib_max) now that adaptive-ib + glue-fusion
changed the narrow-update/trailing tradeoff. Old finding "NB=512 worse than 256" predates
those. Bigger NB -> bigger-K fat trailing (higher GEMM efficiency: bake-off K=128->21%,
K=256->34% cuBLAS) at the cost of more narrow updates (now cheaper). Monkeypatches the
CURRENT submission's _blocking and times custom_kernel on the real benchmark shapes.
No banned substrings.
"""
import sys
sys.path.insert(0, "harness")
import torch
import reference as ref
import sub_now as S

torch.backends.cuda.matmul.allow_tf32 = True


def clear_l2(): torch.empty((32, 1024, 1024), dtype=torch.int64, device="cuda").fill_(0)


def time_fn(fn, it=20):
    for _ in range(4):
        try: fn()
        except Exception as e: return ("ERR:" + type(e).__name__)
    torch.cuda.synchronize(); ts = []
    for _ in range(it):
        clear_l2(); torch.cuda.synchronize()
        s = torch.cuda.Event(enable_timing=True); e = torch.cuda.Event(enable_timing=True)
        s.record(); fn(); e.record(); torch.cuda.synchronize(); ts.append(s.elapsed_time(e))
    ts.sort(); return ts[len(ts) // 2] * 1000


def bench_case(B, n, seed, grid):
    A = ref.generate_input(batch=B, n=n, cond=2, seed=seed, case="dense")
    cur_NB, cur_ib = S._blocking(n)
    print(f"\n=== n={n} b={B}  (current _blocking = NB{cur_NB}/ib{cur_ib}) ===")
    print(f"{'NB':>5} {'ib_max':>7} {'us/call':>10}  {'vs cur':>8}")
    base = None
    for NB, ib in grid:
        S._blocking = (lambda NB, ib: (lambda nn: (NB, ib)))(NB, ib)  # force this config
        t = time_fn(lambda: S.custom_kernel(A.clone()))
        if isinstance(t, str):
            print(f"{NB:>5} {ib:>7} {t:>10}")
            continue
        if NB == cur_NB and ib == cur_ib:
            base = t
        tag = ""
        print(f"{NB:>5} {ib:>7} {t:>10.1f}")
    # second pass to print vs-current ratios once base known
    if base:
        print(f"  (current config = {base:.1f} us; lower is better)")


def main():
    print("dev", torch.cuda.get_device_name(0), "torch", torch.__version__, "triton")
    # n=512: NB sweep (cur 128/64). bigger NB -> bigger-K C-=V@Y trailing.
    bench_case(640, 512, 1029, [(128, 64), (128, 128), (192, 64), (256, 64),
                                (256, 128), (384, 128), (512, 128)])
    # n=1024: cur 256/128 (ib capped at 32 by SRAM at large m). sweep NB.
    bench_case(60, 1024, 75342, [(128, 128), (256, 128), (384, 128), (512, 128)])
    # n=2048: cur 256/128.
    bench_case(8, 2048, 224466, [(128, 128), (256, 128), (384, 128), (512, 128)])


if __name__ == "__main__":
    main()
