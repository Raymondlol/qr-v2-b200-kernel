# HANDOVER → CuTe-DSL port session (archive of the 2026-06-27 session)

> Read this, then `CLAUDE.md` + auto-memory. ⚠️ **CLAUDE.md still says "eval has no nvcc" — that is
> now FACTUALLY FALSE (see §1). Do not trust the no-nvcc banner.** Comp ends **2026-06-30**.

## §0 — STATE (one line)
Official best = **V10 = 5343µs** (`milestones/submissionV10_fused_modal4913.py` on `persistent-engine`,
id 838292; main/V9 = 5791 untouched). Leader 1292. This session: confirmed V10 official, **falsified**
the Gluon tcgen05 per-matrix engine (TMEM-64-row wall), and **discovered the eval has nvcc+CUTLASS+
cute-dsl** → reopens raw-CUDA/CUTLASS/cute-dsl. Next: port the warp-spec persistent engine to cute-dsl.

## §1 — ★ THE UNLOCK (verified, foundational): the eval HAS nvcc + CUTLASS + cute-dsl
I WebFetch'd live `gpu-mode/kernelbot/main:src/runners/modal_runner.py`. The eval image is:
**`nvidia/cuda:12.9.1-devel-ubuntu24.04`** (⇒ **nvcc present**) + pip **`nvidia-cutlass-dsl==4.5.2`** +
`cuda-python[all]==13.0` + `cuda-core[cu13]` + torch 2.12 (+triton) + tinygrad + helion; AND
`git clone --branch v4.5.1 NVIDIA/cutlass /opt/cutlass` (`CUTLASS_PATH` set). `run_eval.py` runs `.py`
via python3 and `.cu` via nvcc. **Submissions CANNOT pip-install** (static image) but CAN use everything
pre-installed + shell to nvcc + the CUTLASS headers.
- **The project's "eval has no nvcc → raw-CUDA/PTX/CUTLASS undeployable" was a MODAL-HARNESS ARTIFACT:**
  our `modal_microbench.py` image = `debian_slim+torch+triton` (no nvcc/cutlass — my own probe
  `modal_cutedsl_test.py` confirmed THAT image lacks them), but that is OUR test image, NOT the eval.
- **=> deployable now: raw CUDA `.cu`, CUTLASS C++, AND cute-dsl.** All three.

### ★ FIRST THING TO DO (do not build on an unverified assumption):
Submit `experiments/probe_eval_env.py` (ready, in this branch) — it prints `which nvcc` + `import
cutlass`/`cuda` to stdout+stderr (lands in the gpumode "Debug Info" column), then falls back to
`torch.geqrf` (correct, 22/22, low score, zero risk). Confirm the **qr_v2 board** uses the CUDA-devel
image (I verified the GENERAL kernelbot main; the board *could* override). One submission, decisive.

## §2 — FA4 is the blueprint (cross-verified: Tri Dao blog + Colfax + Modal teardown)
FlashAttention-4 (Blackwell) = **warp-specialized persistent tcgen05 engine, written ENTIRELY in
cute-dsl (Python) + inline PTX — no C++/nvcc.** It proves a top-tier Blackwell warp-spec kernel is
deployable in pure Python-DSL (exactly our deployability situation). Architecture (our design-B target):
- **~5 specialized warp roles** via smem-barrier state machine: 1 Load warp (TMA, async, register-light),
  1 MMA warp (`tcgen05.mma`, single-thread launch = low reg pressure), N "compute" warpgroups (FA4: 8
  softmax), 1 dedicated Correction warpgroup (off the critical path), 1-2 Epilogue warps.
- **Persistent:** `StaticPersistentTileScheduler` (1 CTA/SM, dynamic tile→SM scheduling).
- **Pipelining:** ping-pong tiles; the next matmul issues BEFORE the current non-matmul (softmax) work
  finishes → tensor cores continuously fed. *This is exactly "hide the panel behind the GEMM".*
- **★ THE register-wall escape = accumulators in TMEM (not registers)** — the headline FA3→FA4 change.
  "On Blackwell accumulators live in TMEM, making it practical to keep multiple MMAs in flight while
  CUDA cores do the element-wise work." This is the answer to OUR design-A wall (128-reg worker can't
  co-reside): park the MMA accumulator in TMEM, use TMA for loads → the warp-spec overlap fits.
- **2-SM / 2-CTA UMMA mode** (one MMA across a CTA-pair).

## §3 — cute-dsl vs Gluon (the honest, decisive comparison)
- **Both** compile to the SAME PTX/SASS → **SAME hardware constraints.** cute-dsl is NOT a hardware or
  numerical unlock.
- **Gluon HAS the primitives:** `warp_specialize` (with `worker_num_regs` → dynamic `setmaxnreg` reg
  realloc), TMA (`tma.async_load`/`TensorDescriptor`), `tcgen05_mma` + `allocate_tensor_memory` (TMEM
  acc), `two_ctas` 2-SM tiles, persistent grid (manual), `clc`. The leader's **1558µs is PURE GLUON** —
  proof the engine is Gluon-expressible.
- **What cute-dsl ADDS = turnkey machinery Gluon lacks:** pre-built deadlock-hardened **pipeline-ring
  classes** (`PipelineTmaUmma` TMA→UMMA, `PipelineUmmaAsync` UMMA→async-accumulator, `PipelineTmaAsync`,
  `MbarrierArray`, `PipelineState`), **persistent tile schedulers** (`StaticPersistentTileScheduler`,
  `ClcDynamicPersistentTileScheduler`), explicit `warpgroup_reg_alloc/dealloc`, hardware `NamedBarrier`
  (16 slots), `elect_one`, `TmemAllocator`. In Gluon you HAND-ROLL all of this from raw mbarriers + manual
  loops. **=> cute-dsl makes design-B much EASIER + less deadlock-prone to BUILD, NOT easier to WIN.**

## §4 — ★ SET EXPECTATIONS (the wall cute-dsl does NOT remove)
- **tcgen05 TMEM 64-row accumulator minimum** (M∈{64,128} MMA tiles; TMEM always 128 lanes). This is
  HARDWARE — identical PTX from both DSLs. It is what walled our Gluon engine: it forces a ~180-190 reg
  floor on the tcgen05 apply → 1 CTA/SM (vs V10's plain `tl.dot` apply 168 regs/3 CTAs/SM), and pads any
  narrow NB→64 (2-4× waste). **cute-dsl hits it identically.**
- **★ FA4 dodges this because attention has FAT tiles (M=128, big K); our qr n≤512 trailing is K-POOR
  (K=16-64) → ~25% peak roofline ceiling.** So porting FA4's structure does NOT automatically beat the
  K-poor ceiling. **The only win available is the OVERLAP (hide the latency-bound panel behind the
  trailing GEMM) — NOT GEMM efficiency.** Best case ≈ the trailing-apply-alone time = **8552µs = 0.87×
  V10** (measured this session). So a *successful* cute-dsl port lands ~0.87× V10 on n=512, IF the
  overlap fully hides the panel — a real but bounded win, and still far from 1292 (which needs the panel
  hidden AND the K-poor GEMM lifted, the latter being roofline-bounded).

## §5 — the QR problem + reusable assets (don't rebuild)
- **Contract:** `custom_kernel(data:[B,n,n]) -> (H, tau)` flat compact-Householder (reflectors in
  strict-lower(H), R=triu(H), tau). 12 cases (gpu_bench.py), geomean; n=512 b640 ×4 dominate. recheck=True.
- **Algorithm spec (design-neutral, START HERE):** `experiments/cpu_blocked_qr_spec.py` +
  `cpu_nb_qr_spec.py` (right-looking ib=16 blocked Householder + LARFT, validated vs geqrf to 1e-15).
- **Branch `engine-moonshot`** = the Gluon engine R&D + the CORRECT overlap prototype:
  - `experiments/m4_overlap.py` = **CORRECT look-ahead warp_specialize(panel[k+1]‖far-trailing[k])**
    prototype (grid=1 H=3.86e-6). Spills (register co-residence) → 4.8× — port THIS structure to cute-dsl
    with TMEM-acc + the pipeline classes to kill the spill.
  - `gfused.py` (NB=64 monolithic), `m2_narrowapply.py` (`_narrow_apply`, correct), `stage1d_overlap.py`
    (a working `warp_specialize(panel‖trailing)`), `stage0_regfile_panel.py` (rowmagma panel).
  - `docs/MOONSHOT_PLAN_M3.md` + `MOONSHOT_LOG.md` = full engine post-mortem (the TMEM-64 wall, the lean
    paths, why each failed). `MOONSHOT_BRIEF.md` = the gates/discipline. Reproducible snapshot tag
    `engine-moonshot-base-v0`.
- **Branch `persistent-engine`** = V10 (`experiments/cand_fused.py`), the profiling
  (`modal_profile_fused.py`: SASS shows V10 is scalar/occupancy-bound, HMMA only 3.7%), the precision
  probe (`fused_qr_prec.py`: tf32x3 margin 587, tf32x1 fails+only 1.3% faster → precision DEAD for V10;
  tf32x2 valid for the engine when GEMM is the critical path).

## §6 — HARD GATES (every iter)
1. **`grep -niE "stream|graph" <submission>` MUST be empty** (static scan). ⚠️ cute-dsl: `cutlass.pipeline`
   / CLC scheduler names MAY surface those substrings in imported strings — RE-GREP every cute submission.
2. relerr<1e-4 vs geqrf (22/22). **mixed@640 worst-of-640 margin ≥1.83 probed separately** (tf32x3 ok;
   tf32x2 likely ok). tf32x3 trailing for n≤512; n≥1024 may use 1×TF32.
3. Gate EVERY win on a real gpumode submission (Modal ≈ official −4%; but Modal image ≠ eval image — see §1).
4. Never touch `main`(=V9) or `cand_fused.py`(=V10). B200 = the user's $, be deliberate.

## §7 — what THIS session did (archive)
V10 confirmed official 5343 (−7.7% vs V9). Built M1-M5 fused Gluon engine + the M4 correct overlap
prototype; **falsified the per-matrix tcgen05 engine** (best correct = NB=32 19758µs = 2.0× V10; root
cause = TMEM-64-row → 1 CTA/SM). Profiled V10 (scalar/occupancy-bound). Closed precision (dead for V10),
pipelining (slower), n=1024-fused (60-CTA underutil), as orthogonal dead-ends. **Discovered + verified
the eval has nvcc+CUTLASS+cute-dsl** (the big one). Probe + this handover committed on
`claude/festive-ptolemy-6a9bd0`. CuTe-DSL deploys; it's easier engineering for design-B, bounded ~0.87×
V10 by the K-poor wall, NOT a 1292 silver bullet — but worth porting because it makes the only-remaining
lever (overlap) tractable AND reopens raw CUDA/CUTLASS C++.
