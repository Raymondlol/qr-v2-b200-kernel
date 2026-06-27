"""DEEP-DIVE 2 — launch-gap + per-n regime probe (B200).

The roofline (deepdive_roofline.py) says small-n is memory-side + tiny absolute work => likely
LAUNCH-bound. Confirm directly: for each ranked case, measure wall-clock vs SUM of GPU kernel
self-times. gap = wall - sum(gpu) = time the GPU sat IDLE waiting for CPU to dispatch the next
of ~55 kernels/iter. Large gap% => launch/dispatch-bound => the lever is FEWER launches (which is
why implicit-V won small-n). Small gap% => GPU-bound (roofline applies).

Also reports the per-category split (panel/tf32x3/solve/glue) at each n, to see if glue% rises at
small-n. No banned substrings.
"""
import pathlib, modal

HERE = pathlib.Path(__file__).parent
image = (modal.Image.debian_slim(python_version="3.12")
         .pip_install("numpy")
         .pip_install("torch", index_url="https://download.pytorch.org/whl/cu128")
         .add_local_file(str(HERE / "submission.py"), remote_path="/work/submission.py", copy=True))
app = modal.App("qr-deepdive-launchgap")


@app.function(gpu="B200", image=image, timeout=900)
def run():
    import sys, statistics
    sys.path.insert(0, "/work")
    import torch
    import submission

    dev = "cuda"
    print("dev", torch.cuda.get_device_name(0), "torch", torch.__version__)

    def clear_l2():
        torch.empty((32, 1024, 1024), dtype=torch.int64, device=dev).fill_(0)

    def make(n, b):
        torch.manual_seed(1234)
        return torch.randn(b, n, n, device=dev, dtype=torch.float32)

    def wall_us(fn, it=30):
        for _ in range(8):
            fn()
        torch.cuda.synchronize()
        ts = []
        for _ in range(it):
            clear_l2(); torch.cuda.synchronize()
            s = torch.cuda.Event(enable_timing=True); e = torch.cuda.Event(enable_timing=True)
            s.record(); fn(); e.record(); torch.cuda.synchronize()
            ts.append(s.elapsed_time(e))
        ts.sort()
        return ts[len(ts) // 2] * 1000.0

    def categorize(name):
        n = name.lower()
        if "_panel_kernel" in n: return "panel"
        if "_bmm_x3" in n: return "tf32x3"
        if "trsm" in n or "getrf" in n or "geqrf" in n or "ormqr" in n: return "solve"
        if "triu_tril" in n or "memcpy" in n or "elementwise" in n or "fill" in n or \
           "reciprocal" in n or "where" in n or "arange" in n or "copy" in n or "offsetpointer" in n:
            return "glue"
        if "cutlass" in n or "gemm" in n or "ampere" in n or "sgemm" in n: return "cublas"
        return "other"

    def gpu_self_us(fn, iters=10):
        # sum of CUDA-side self time per iter + per-category, via kineto (gVisor-safe)
        from torch.profiler import profile, ProfilerActivity
        for _ in range(5):
            fn()
        torch.cuda.synchronize()
        with profile(activities=[ProfilerActivity.CUDA]) as prof:
            for _ in range(iters):
                fn()
            torch.cuda.synchronize()
        tot = 0.0
        bycat = {}
        nlaunch = 0
        for ev in prof.key_averages():
            # self_device_time_total is microseconds across all `count` calls
            sd = getattr(ev, "self_device_time_total", 0) or 0
            if sd <= 0:
                continue
            tot += sd
            bycat[categorize(ev.key)] = bycat.get(categorize(ev.key), 0.0) + sd
            nlaunch += getattr(ev, "count", 0)
        tot /= iters
        bycat = {k: v / iters for k, v in bycat.items()}
        return tot, bycat, nlaunch / iters

    cases = [(176, 40), (352, 40), (512, 640), (1024, 60)]
    print(f"\n{'case':>12} {'wall_us':>9} {'gpu_us':>9} {'gap_us':>8} {'gap%':>6} {'launch/it':>9}  | category split (% of gpu)")
    print("-" * 110)
    for n, b in cases:
        A = make(n, b)
        fn = lambda A=A: submission.custom_kernel(A)
        try:
            wall = wall_us(fn)
            gpu, bycat, nl = gpu_self_us(fn)
        except Exception as ex:
            import traceback; print(f"  n={n} b={b} FAIL\n" + traceback.format_exc()[-1500:]); continue
        gap = wall - gpu
        gappct = 100.0 * gap / wall
        cats = " ".join(f"{k}:{100*v/gpu:.0f}" for k, v in sorted(bycat.items(), key=lambda x: -x[1]))
        print(f"  n={n:>4} b={b:>4} {wall:>9.1f} {gpu:>9.1f} {gap:>8.1f} {gappct:>5.1f}% {nl:>9.1f}  | {cats}")
    print("\nREAD: large gap% (GPU idle waiting on CPU dispatch) => LAUNCH-bound => fewer-launches is the lever")
    print("      (the regime where implicit-V's op/launch cut won on Modal). small gap% => GPU-bound (roofline applies).")


@app.local_entrypoint()
def main():
    print(run.remote())
