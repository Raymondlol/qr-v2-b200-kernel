"""profile_workload.py — minimal, profiler-targetable run of the REAL qr kernel.

Designed to be wrapped by an external profiler (ncu / nsys). Keeps the launched
kernel count small and bounds the region of interest with cudaProfiler start/stop
so a profiler can `--capture-range cudaProfilerApi` to skip warmup.

Env vars:
  QR_SUB   submission file to load (default submission.py in cwd)
  QR_N     matrix size       (default 512)
  QR_B     batch             (default 640)
  QR_COND  conditioning code (default 2)
  QR_ITERS timed iterations  (default 1)
  QR_WARM  warmup iterations (default 5)
"""
import os, sys, importlib.util
import torch
import reference

torch.backends.cuda.matmul.allow_tf32 = True
try:
    torch.set_float32_matmul_precision("high")
except Exception:
    pass


def load_kernel(path):
    import task
    sys.modules.setdefault("task", task)
    spec = importlib.util.spec_from_file_location("uut", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.custom_kernel


def main():
    sub = os.environ.get("QR_SUB", "submission.py")
    n = int(os.environ.get("QR_N", 512))
    b = int(os.environ.get("QR_B", 640))
    cond = int(os.environ.get("QR_COND", 2))
    iters = int(os.environ.get("QR_ITERS", 1))
    warm = int(os.environ.get("QR_WARM", 5))

    k = load_kernel(sub)
    A = reference.generate_input(batch=b, n=n, cond=cond, seed=999)
    print(f"[workload] {sub} n={n} b={b} cond={cond} warm={warm} iters={iters}", flush=True)

    for _ in range(warm):
        k(A.clone())
    torch.cuda.synchronize()

    torch.cuda.profiler.start()
    for _ in range(iters):
        k(A.clone())
    torch.cuda.synchronize()
    torch.cuda.profiler.stop()
    print("[workload] done", flush=True)


if __name__ == "__main__":
    main()
