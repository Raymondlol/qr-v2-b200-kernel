# Profiling on Modal — what works, what's blocked, and the NCU substitute

> **Session 2026-06-26, branch `ncu-profiler-modal`.** Goal: get NVIDIA Nsight Compute
> (`ncu`) working on Modal B200 so we can see occupancy / warp-stall reasons / memory
> throughput the way the leaders profile. **Verdict: ncu (and nsys) are IMPOSSIBLE on
> Modal — gVisor sandbox blocks them. A static + kineto substitute was built instead and
> recovers most of the signal.** Raw run: `results/ncu_substitute_analysis_2026-06-26.txt`.

## TL;DR
- **`ncu` cannot profile on Modal.** Not a permission or version bug — Modal runs under
  **gVisor** (`uname` = `4.19.0-gvisor`) and the GPU performance-counter device interface
  (`/dev/nvidia-caps`, `/proc/driver/nvidia/capabilities`) is **not exposed** by gVisor's
  nvproxy. ncu's counter library fails `LibraryNotLoaded` on **every** metric set, across
  **both** ncu 2025.1.1 (CUDA 12.8) and ncu 2025.3.1 (CUDA 13.0, matched to driver 580).
- **`nsys` is blocked too**, by a *different* gVisor limit: the GPU-tick→wall-clock
  converter throws `ConvertGpuTicksToSyncNs InternalErrorException`, so no `.nsys-rep` is
  ever produced (the trace collects, post-processing crashes).
- **What DOES work on Modal:** (a) Triton **compile-time** resources (n_regs / n_spills /
  shared / occupancy) — no GPU profiling needed; (b) **SASS** instruction mix via
  `cuobjdump` on `kernel.asm['cubin']`; (c) **torch.profiler / kineto** per-kernel timeline
  (in-process CUPTI activity records — no global clock sync, so it survives gVisor).
- **To run REAL ncu** you need a **non-gVisor (bare-metal / full-VM) GPU host**.
- **Tool built:** `modal_kernel_analysis.py` — one B200 container that dumps all of the
  above for `submission.py` at the dominant n=512 b=640 case. Re-runnable any time.

## Evidence (two probes, `modal_ncu_probe.py` + `modal_ncu_probe2b.py`)
| check | result |
|---|---|
| user | `uid=0(root)` |
| `RmProfilingAdminOnly` | **0** → the classic `ERR_NVGPUCTRPERM` perm gate is OPEN (not our blocker) |
| `CapEff` | lacks `CAP_SYS_ADMIN` — moot, since the perm gate is already open |
| `uname -a` | **`Linux modal 4.19.0-gvisor`** → gVisor sandbox |
| `/dev/nvidia*` | only `nvidia-uvm`, `nvidiaN`, `nvidiactl` — **no `/dev/nvidia-caps`** |
| `/proc/driver/nvidia/capabilities` | **absent** (the perfmon capability interface) |
| ncu 2025.1.1 (CUDA 12.8) | `Failed to initialize the profiler: LibraryNotLoaded` |
| ncu 2025.3.1 (CUDA 13.0, driver-matched) | **identical `LibraryNotLoaded`** → version-independent |
| nsys 2025.3.2 | collects, then `ConvertGpuTicksToSyncNs InternalErrorException`, no report |

The two NVIDIA tools fail through two independent gVisor gaps (perfmon device access for
ncu; GPU/CPU clock-domain sync for nsys). Both are host/sandbox decisions Modal controls —
nothing we set inside the container changes them.

## The substitute: `modal_kernel_analysis.py`
Recovers the NCU-class signal we actually wanted, via gVisor-safe mechanisms:
1. **Per-kernel static resources** — runs the real kernels, walks live Triton
   `CompiledKernel` objects (`n_regs`, `n_spills`, `metadata.shared/num_warps`), writes each
   `asm['cubin']` to disk, computes analytic B200 occupancy. This IS the register-budget
   signal the design-A "panel must be ≤~64 regs to co-reside" crux depends on.
2. **SASS instruction mix** — `cuobjdump -sass` on the panel cubin, opcode histogram +
   families (FP-math / warp-reduce / shared-mem / barrier / control-flow). Quantifies the
   "panel is latency-bound by the serial reflector reduction" claim.
3. **kineto per-kernel timeline** — `torch.profiler` op-level + per-kernel durations and
   launch counts (the working stand-in for the dead nsys timeline).

## Findings for `submission.py` @ n=512 b=640 (the dominant case, 4 of 12)

### Per-kernel resources (everything is REGISTER-limited; occupancy 12.5–25%)
| kernel | warps | regs | spills | smem | occ | limiter |
|---|---|---|---|---|---|---|
| `_panel_kernel` (tall, first sub-panels) | 8 | **255** | **18** | 2048 | **0.125** | regs |
| `_panel_kernel` (other configs) | 4–8 | 141–179 | 0 | 1024–2048 | 0.125–0.188 | regs |
| `_bmm_x3_kernel` (trailing tf32x3) | 4–8 | 108–200 | 0 | 49K–98K | 0.125–0.25 | regs |
| `_bmm_x3_sub_kernel` (within-panel trailing) | 4–8 | 128–221 | 0 | 49K–98K | 0.125–0.25 | regs |

- **Every kernel is register-bound to low occupancy (12.5–25%).** This is fine at n≤512
  because batch=640 CTAs over 148 SMs hides latency *across* CTAs — but it is exactly why
  the panel **cannot co-reside** with a trailing-GEMM worker on one SM (the design-A crux):
  at 255 regs/thread the panel already consumes the whole 64K register file for its 8 warps.
- The heaviest panel config **spills (18 regs)** — confirms the docs' "255 regs" figure and
  shows the production panel is at the register ceiling. (The `gluon-regfile-panel` branch's
  MAGMA panel at **108 regs / 0 spills** is the fix that unlocks co-residence.)

### Panel SASS instruction mix — latency-bound, NOT FMA-bound (now quantified)
Consistent across all `_panel_kernel` configs (here: the 255-reg config, 1997 instrs):
- **FP-math ~25%** (FFMA 6%, FADD 11%, FMUL 7%, FSEL 14%…) — only a quarter is actual math.
- **warp-reduce (SHFL) ~8%** (154 SHFL) — the reflector norm/dot reductions, a serial chain.
- shared-mem ~2%, barrier ~0.7% (13 `BAR`), global-mem ~1%.
- The remaining ~63% "other" is predication/addressing (ISETP, LOP3, IMAD, LEA, P2R,
  CS2R, `@!P` predicated ops) — the per-column branchless Householder masking.

→ The panel spends most of its instructions on **reductions + predication + addressing**,
not FMAs. That is the SASS-level signature of a **latency-bound, reduction-serial** kernel —
exactly the docs' characterization, now measured (and only obtainable because `cuobjdump`
works under gVisor while `ncu` does not).

### kineto timeline breakdown (corroborates the op-level profile)
`panel 43.4% · tf32x3-trailing 27.5% · cuSOLVER trsm(solve) 14.6% · glue 13.6% · cuBLAS 0.8%`
(self GPU time/iter, b=640). Matches the prior `panel 41 / gram 15 / trailing 26 / solve 18`.
- Biggest single kernels: `_panel_kernel` (43%), `_bmm_x3_kernel` + `_bmm_x3_sub_kernel`
  (trailing, 27%), `batch_trsm_left_kernel` (the T-solve, 15%).
- **glue is a non-trivial 13.6%** — mostly `triu_tril_kernel` (≈751µs across 2 variants) +
  device-to-device memcpy (427µs) + small elementwise. A possible *separate* deployable
  cleanup target (reduce the torch-level triu/tril/copy orchestration), independent of the
  panel overlap engine.

## How this connects to the optimization
The register/occupancy numbers are the concrete baseline for the `gluon-regfile-panel`
overlap effort: they show *why* the production panel can't co-reside (255 regs, spills) and
confirm the regfile panel (108 regs / 0 spills) is the right lever. The SASS confirms the
panel is latency-bound, so hiding it behind the trailing (not speeding up its FMAs) is the
correct strategy. See `[[qr-v2-gluon-warp-specialize]]` / the `gluon-regfile-panel` branch.

## If real NCU is needed (recommendation)
Use a **non-gVisor GPU host** where the perfmon device interface and clock sync are intact:
- Bare-metal / full-VM B200 or H100 (e.g. Lambda Cloud, RunPod *bare-metal*, Datacrunch,
  Crusoe, a Vast.ai bare host, or a local box). Confirm `ls /dev/nvidia-caps` is non-empty
  and `cat /proc/driver/nvidia/params | grep Prof` shows `RmProfilingAdminOnly: 0` (or run
  ncu as root). Then `ncu --set full` gives occupancy / stall reasons / memory throughput /
  tensor-core utilization directly.
- The competition **eval** infra is likely similarly locked down; profiling is a dev-time
  activity on your own host, not on the eval box.

## Re-run the substitute
```
/Users/raymond/Downloads/SubPY/.modalenv/bin/modal run modal_kernel_analysis.py
```
Probes (diagnostics only): `modal_ncu_probe.py`, `modal_ncu_probe2b.py`.
