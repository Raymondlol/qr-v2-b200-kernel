# HANDOVER → next session: the research-grade swing (Gluon warp-spec panel/trailing OVERLAP)

> **▶▶ NEXT WORK = `docs/SMEM_PANEL_PLAN.md` on branch `gluon-smem-panel`.** The overlap direction is
> de-risked down to ONE remaining step: a SMEM-resident streaming panel (frees registers so the
> 16-warp panel‖trailing config fits). Read the plan's Stage A gate first. This doc below is the full
> Phase 0/1/1.5 diagnosis the plan rests on.

**Read this first, then `CLAUDE.md` + `docs/FP8_SESSION_PROGRESS.md` (the strategic map) + `docs/DEAD_ENDS.md` (do-not-retry).**

> ## ✅ PHASE 0 + PHASE 1 = DONE, BOTH GO (2026-06-26). Design A chosen (flips the old "lean B").
> Evidence: `results/phase0_ws_overlap.txt`, `results/phase1_ws_realscale.txt`, `results/phase1_trailing_penalty.txt`. Scripts: `experiments/gluon_ws_overlap.py`, `gluon_trailing_penalty.py`, `gluon_lowlevel_dump.py` (the canonical async tcgen05 recipe), `gluon_ws_dump.py`.
> - **`gl.warp_specialize` EXISTS** in stock Gluon (the old recon checked `tl.warp_specialize` → wrong). `gl.warp_specialize([(default_fn,args),(worker_fn,args)], [worker_warps],[worker_regs])`. Partitions must NOT return tensors (flatten-return needs `_semantic`) → write outputs directly. `gl.warp_id` absent — but not needed.
> - **Async tcgen05 recipe** (from `tl_dot_blackwell`): smem operands via `get_shared_memory_mma_operand` → `allocate_tensor_memory` → `fence_async_shared` → mbarrier alloc/init → `tcgen05_mma(..., mbarriers=[bar])` (**MUST pass mbarriers or it's SYNCHRONOUS**) → `mbarrier.wait`.
> - **Overlap is real**: reduction (CUDA cores) hides behind async tcgen05 (tensor cores). eff 0.97–1.00 when MMA-dominated; eff 0.80–0.84 at grid=640 in the panel-dominated n=512 regime. Non-warp-spec single-partition = ZERO overlap (gl.sum is a CTA-collective that waits for the issuing warp).
> - **Single-CTA trailing penalty at n=512 = only 1.12×** (tf32x3-equiv, tcgen05; the old 1.71× was tl.dot). 640 matrices fill 148 SMs → no occupancy penalty. (n=1024 = 2.31× → shape-route A to n≤512 only.)
> - **Design-A net estimate ~1.26–1.3× on n=512 ≈ +9% geomean** (on the PROXY).
>
> ## ⛔ PHASE 1.5 DE-RISK (2026-06-26) = DESIGN A IS NO-GO. Do not build the in-CTA fused kernel.
> Ported the REAL `_panel_kernel` to Gluon (`experiments/gluon_panel.py`, bit-faithful relerr ~1e-7) and co-hosted it with a real tcgen05 trailing in ONE CTA via warp_specialize (`experiments/gluon_panel_trailing.py`, `results/phase1.5_panel_coresidence_NOGO.txt`). At every real co-resident panel height (384/256/128, grid=640) **WS is 1.2–1.4× SLOWER than sequential (eff NEGATIVE)**. The panel keeps its [M,64] tile resident (~128 regs/thread); co-residing it with the trailing worker blows registers/occupancy + warp_specialize overhead (~485µs) > the hide-able trailing. **The Phase-0/1 LIGHTWEIGHT proxy masked this.** Lesson: validate overlap with the REAL register footprint.
> ## 🔬 FULL DIAGNOSIS (deeper probing, `gluon_panel_salvage.py` + `gluon_panel_async.py`): design A needs TWO fixes
> The regression has two independent causes, BOTH must be solved:
> 1. **Worker must use LOW-LEVEL async tcgen05, NOT `tl_dot`** — `tl_dot`'s CTA-wide barrier couples the warp_specialize partitions → they serialize (eff 0.06). Low-level `tcgen05_mma(mbarriers=[private_bar])` does NOT couple → overlaps. **BUILT** (`gluon_panel_async.py::_async_trail_part`). Worker also needs 8w/128r (4w/64r starves it 3.3×).
> 2. **Panel must be LOW-REGISTER (smem-resident)** — real register-resident panel ([M,64] tile ~96–128 regs) + async worker = 16 warps blows the 64K reg file → WS up to 2.7× SLOWER at M=384; only M=128 goes positive. PROOF the fix works: a LOW-reg reduction default || async worker at 16 warps overlaps at **eff 0.64–0.73** (`results/phase1.5_16warp_async.txt`).
> - **Verdict: design A IS salvageable, but needs a SMEM-RESIDENT STREAMING PANEL rewrite** (reduction + rank-1 update stream row-blocks through smem, one block in registers) — multi-hour, residual risk = smem-panel speed. **Realistic payoff now ~+6% geomean** (eff~0.65 × trailing 26%), which EQUALS the zero-risk recursive-blocking fallback. So the smem-panel build is no longer clearly better EV.
> - **Reusable BUILT+verified assets:** `gluon_panel.py` (panel port), `gluon_panel_async.py` (async tiled tcgen05 trailing worker + the no-coupling lesson), `gl.warp_specialize` mechanics, the full diagnostic chain. **Best official unchanged: V5 tf32x3 5915µs (net speed gain this session = 0).**

## TL;DR
- **State:** `submission.py` = V5 tf32x3 = **5915µs official (confirmed best)**. Leader **1292µs (~4.6×)**. All cheap/deployable levers exhausted. Net last session = 0 speed (fp16x3 reverted as official wash) but produced the full "what's dead & why" map.
- **The gap is the PANEL** = **~41% of n=512 runtime, latency-bound by the sequential reflector-reduction chain** (Householder's math; profiled in `experiments/profile_phases.py`: panel 41 / gram 15 / trailing 26 / solve 18). The trailing GEMM (26%) is already well-optimized; precision is settled (tf32x3; fp8/fp4 speed-dead; fp16x3 wash).
- **The one un-dead, research-grade lever** (survived the red-team): **hide the latency-bound panel behind the tensor-core trailing GEMM via a fused Gluon warp-specialized kernel** (panel = CUDA cores, trailing = tcgen05 tensor cores = different HW units → overlappable). This is the leader's likely approach. **Payoff ~5% geomean realistic, ~10-15% ceiling. Multi-day, uncertain.**

## Why this and not the others
- fp8/fp4/mxfp/nvfp = **speed-dead** (shape mismatch, dedicated tcgen05 cores verified used — not an API problem). Don't.
- 1×TF32 = reseed-DQ. fp16x3 = official wash (fp32-acc = tf32 rate). recursive-blocking = ~6% but Amdahl-capped, doesn't touch the panel (good DEPLOYABLE FALLBACK, see below).
- CholeskyQR n=4096 = ~3%, one case, but reconstruction (ORHR_COL) re-imports a sequential panel + ill-conditioned dense (cond up to 8e6) needs fp32-Gram+shift → red-team rated low-EV. (Documented alt, not the primary.)
- Only the panel/trailing overlap attacks the real 41% bottleneck on the dominant n=512 cases (4 of 12).

## The central risk you MUST de-risk early (the tension)
Overlap requires FUSION (single kernel — separate kernels serialize, and streams are BANNED). Two granularities:
- **(A) per-matrix fused look-ahead** (one CTA/matrix does panel(k+1) ‖ trailing-rest(k)): SIMPLE but the trailing becomes single-CTA = **1.4–3.8× slower** (`microbench_trailing.py`; n512 fat 1.71×). The overlap gain (hide ≤26%) likely does NOT recover a 1.71× trailing penalty → **(A) probably LOSES. Do not start by building the full (A).**
- **(B) persistent inter-matrix pipelining** (a fixed grid of CTAs pipelines panel-of-matrix-i with trailing-of-matrix-j, keeping batched-trailing efficiency): VIABLE, the leader's likely design, but harder.
The staged TODO below MEASURES this tension before committing to the big build.

---

## STAGED TODO (each phase is a GO/NO-GO gate — abandon cheaply if it fails)

### Phase 0 — Primitive co-host validation (~½ day, ~cents). **THE biggest unknown.**
**Question:** can a Gluon kernel run a CUDA-core reduction (the panel's `gl.sum` reflector chain) CONCURRENTLY with a `tcgen05_mma` (the trailing) in ONE grep-clean kernel, and do they actually OVERLAP?
1. Branch from **main** (`git checkout -b gluon-overlap main`). Copy the scaffolding from the parked branch: `git checkout gluon-tcgen05 -- experiments/gluon_gate2.py experiments/gluon_api_recon.py experiments/gluon_pattern_dump.py`. (Do NOT branch from gluon-tcgen05 — it predates `harness/lab.py` + `modal_lab.py`.)
2. Confirm the **low-level async** tcgen05 API works (gate2 used the high-level `tl_dot` helper which BLOCKS — useless for overlap). You need: `allocate_tensor_memory` (TMEM), the ASYNC `tcgen05_mma` (issue, don't wait), `tcgen05_commit` + `mbarrier` (wait), `tcgen05_copy` (TMEM→regs). API surface is in `recon_tlx_gluon.py` output (captured 2026-06-26): present = `tcgen05_mma/tcgen05_commit/tcgen05_copy/allocate_tensor_memory/mbarrier/async_copy/tma`; **ABSENT = `warp_specialize`/`async_task`** → you must hand-roll warp specialization with warp-id branching + mbarrier.
3. Build a MINIMAL kernel: warp-group-0 issues an async `tcgen05_mma` loop (trailing-like), warp-group-1 runs a `gl.sum` reduction loop (panel-like) WHILE the mma is in flight, then mbarrier-sync. Toggle sequential-vs-overlapped; time both on B200 (`modal_microbench.py`).
- **GO** (overlap observed, e.g. total ≈ max(mma,reduce) not sum, ≥1.3× vs sequential; grep-clean) → Phase 1.
- **NO-GO** (async tcgen05 can't co-host a reduction / no overlap / API too limited) → **the direction is infeasible in stock Gluon. STOP. Fall back to recursive-blocking (deployable ~6%, see Fallback).**

### Phase 1 — Overlap ceiling vs the trailing-penalty tension (~1 day). **GO/NO-GO on design (A) vs (B).**
1. Pin the single-CTA-trailing penalty for n=512 from `microbench_trailing.py` (≈1.71× fat). Compute: does hiding the panel (41%) behind a 1.71×-slower trailing net-win? (Rough: trailing 26%→44% under penalty, hide ≤26% of panel → ~break-even. So (A) is marginal-to-losing.)
2. Build a minimal ONE-super-panel-step fused kernel at the REAL n=512 shape, in the granularity Phase-1.1 favors (likely **(B) persistent**, since (A) loses). Compare wall-time to the current sequential (separate `_panel_kernel` + trailing).
- **GO** (fused step nets >1.1× vs sequential) → Phase 2.
- **NO-GO** ((A) loses AND (B) prototype doesn't net-win) → **STOP, recursive-blocking fallback.**

### Phase 2 — Full fused factorization (multi-day, THE build).
- Port `submission.py::_panel_kernel` (the reflector reduction loop) into Gluon as the panel warp-group; the trailing via async `tcgen05_mma`; mbarrier coordination; the look-ahead split (do the look-ahead column block [k+nb:k+2nb] of trailing(k) first, then panel(k+1) ‖ trailing-rest(k)).
- Build incrementally: correct on ONE matrix vs `torch.geqrf` → batched 640 → then 22/22.
- Keep panel reductions FP32 (low-precision panel corrupts reflectors — proven dead). Keep the T-factor solve fp32-safe (the solve-tf32 artifact — see [[qr-v2-solve-tf32-floor-artifact]]).

### Phase 3 — Integrate, validate, submit (~1 day).
- Shape-route: the Gluon overlap kernel for **n≤512** (where panel = 41%); keep the existing path for n≥1024 / large-n.
- Gate: (a) 22/22 via `modal run modal_lab.py --mode correctness`; (b) **mixed@640 worst-of-640 margin ≥ 1.83×** probed SEPARATELY (the lab gate is batch-16 and MISSES it — use the `microbench_validate.py` / `microbench_1x_reseed.py` pattern); (c) `modal_lab.py --mode compare` geomean; (d) `grep -niE "stream|graph" submission.py` empty.
- **SUBMIT to gpumode to confirm official** — Modal codegen gains can fail to transfer (the fp16x3 lesson: Modal +3.6% → official wash). Do NOT trust a Modal-only win.

---

## Fallback ladder (so the session always lands something)
1. Phase 0 NO-GO → **recursive blocking** (K-rich trailing, `docs/NEXT_STEPS.md` §1a): ~6% on the trailing, deployable, NO Gluon, no new paradigm. Ship it.
2. Phase 1 NO-GO → recursive blocking + optionally the **solve→fp32 robustness fix** (`experiments/cand_fp16x3_solvefix.py` pattern on V5: margin 1.9×→800× at ~+1%; only for reseed-DQ insurance).
3. Phase 2 too hard → ship whatever Phase 0/1 banked + recursive blocking.

## Starting assets (exact)
- Scaffolding: parked branch `gluon-tcgen05` → `experiments/gluon_gate2.py` (Gluon batched GEMM: `from triton.experimental.gluon import language as gl`, `default_blocked_layout`, `SliceLayout`, batched grid launch), `gluon_api_recon.py`, `gluon_pattern_dump.py`. Commit `22d2550`.
- Capability (captured this session, `recon_tlx_gluon.py` on B200 triton 3.6.0): Gluon `nvidia.blackwell` exposes `tcgen05_mma`, `tcgen05_mma_scaled`, `tcgen05_commit`, `tcgen05_copy`, `allocate_tensor_memory`, `tensor_memory_descriptor`, `mbarrier`, `async_copy`, `tma`, `fence_async_shared`. NO high-level `warp_specialize`/`async_task`.
- Panel to port: `submission.py` `_panel_kernel` (lines ~215-260) + `_apply_block` (compact-WY trailing).
- Profile anchors: `experiments/profile_phases.py` (panel 41 / gram 15 / trailing 26 / solve 18 at n=512 b=640).
- Test harness: `modal_lab.py` (correctness/compare/profile) + `modal_microbench.py --script <x>` (single B200 script). Modal path: `/Users/raymond/Downloads/SubPY/.modalenv/bin/modal`.

## Honest risk register
- No high-level `warp_specialize` → hand-rolled mbarrier warp-spec is the core difficulty and the Phase-0 risk.
- The single-CTA-trailing penalty may force design (B) persistent (harder than (A)).
- Pure-Gluon may have a ceiling above the leader's 1292 (they may use something beyond stock Gluon; the docs are medium-confidence on this).
- Payoff is bounded ~5-15% geomean even if it fully works — this is a research investment, not a sure thing. **The GO/NO-GO gates at Phase 0/1 are there so you spend ≤1.5 days before knowing.**

## Do-NOT-retry (waste of time — see `docs/DEAD_ENDS.md`)
fp8/fp4/mxfp/nvfp trailing; 1×TF32; fp16x3; raw-PTX/load_inline (no nvcc); streams/cooperative-launch (rule); conditioning-routing/early-stop (reward-hack DQ); lowprec-panel; mega-kernel single-CTA trailing; naive (library-primitive) CholeskyQR; TSQR→flat reconstruction; per-column panel streamline.
