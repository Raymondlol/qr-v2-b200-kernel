# harness/ — correctness + timing

**Mixed provenance — see `../NOTICE` before reusing anything here.**

Upstream (GPU MODE `kernelbot` / `reference-kernels`, vendored for reproducibility):

| File | What it is |
|---|---|
| `eval.py` | The competition's own evaluation driver (timing loop, L2 clear, recheck). |
| `reference.py` | The `qr_v2` task: input generation, the reference QR, and the correctness gates (factor residual ≤ 20·n·eps, orthogonality ≤ 100·n·eps, measured in FP64). |
| `task.py` | 10-line py3.8 shim for upstream's `TypedDict`-based version. |
| `utils.py` | Seeding + L2-cache clear (upstream, itself adapted from Liger-Kernel). |

Mine:

| File | What it does |
|---|---|
| `check_local.py` | Free, instant CPU gate. Runs the **real** `reference.py` checker against a submission's `custom_kernel`. Validates the eager/torch path and the `(H, tau)` contract; Triton paths do not run on CPU. |
| `gpu_bench.py` | Standalone B200 benchmark faithful to `eval.py`'s leaderboard timing (CUDA events, L2 clear, batch count, `recheck=True`) plus a full warmup pass. Prints per-case µs + the geomean the leaderboard ranks on. |
| `lab.py` | The experiment lab, run *inside* one Modal container: baseline + N candidates timed **interleaved** within each rep, R reps → per-case paired delta ± stderr. This is what makes A/B results trustworthy; see `../docs/METHODOLOGY.md`. |
| `profile_workload.py` | Minimal profiler-targetable run (bounded with `cudaProfiler` start/stop) for ncu/nsys. |
