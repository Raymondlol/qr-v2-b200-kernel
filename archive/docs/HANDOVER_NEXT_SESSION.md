# HANDOVER → next session (final state after the 2026-06-26 research-grade session)

> **▶ READ THIS FIRST, then `CLAUDE.md` + `docs/DEAD_ENDS.md`. Memory files `the Gluon warp-specialization autopsy in `docs/DEAD_ENDS.md`` + `the atomic grid-barrier result in `docs/DEAD_ENDS.md`` mirror this. The autonomous next-session plan is `docs/NEXT_SESSION_PROMPT.md`.**

## STATE
`submission.py` = **V5 tf32x3 = 5915µs official (best, UNCHANGED, grep-clean)**. Leader **1292µs (4.6×)**. Net session speed gain = **0**; the value is the COMPLETE measured map of what's dead and why (below). Branch this work lives on: `gluon-smem-panel` → merged to `main`.

## What the 2026-06-26 session CLOSED (every overlap + large-n + deployable lever, measured-dead)
- **Gluon warp-spec OVERLAP (design A = in-CTA panel‖trailing): DEAD.** Mechanism validated (`gl.warp_specialize` EXISTS in stock Gluon; async tcgen05 overlaps eff 0.6–1.0; single-CTA trailing penalty at n=512 only 1.12×; the worker MUST use low-level async `tcgen05_mma(mbarriers=[bar])` NOT `tl_dot` which CTA-barrier-couples → eff 0.06). BUT the panel can't co-reside: the register panel blows the 16-warp/64K budget; the smem-streaming panel (built, bit-faithful) is **2.56–6.74× too slow** (register gate ⟂ speed gate = mutually exclusive). See DEAD_ENDS "Gluon design-A overlap".
- **Deployable trailing levers: DEAD.** §1a 3-level recursive (K-rich) blocking = **0.976×** (NB-256 widening eats the win); §1b 3×cuBLAS trailing = **0.866×**. The fused tf32x3 Triton kernel is already optimal.
- **CholeskyQR (incl the conditioning-routing idea): DEAD.** Floor-breaker `_LinAlgError` at n=2048/4096 (cond(G)~6.5e11 — can't even factor); slower than our Householder where valid (n=512 22ms vs 14ms, because the panel is already batch-hidden there); conditioning-routing is the v2-forbidden reward-hack (reseed-DQ).
- **Multi-CTA cooperative panel: barrier WORKS, panel use DEAD.** Hand-rolled sense-reversing atomic grid barrier (`tl.atomic_add`+spin) is grep-clean / correct / reusable / **~2.6µs** — overturns "grid-sync=banned" (the ban is only `cudaLaunchCooperativeKernel`). BUT the panel's per-column work (~0.66µs) < barrier (~2.6µs) → cooperative panel **7–8× SLOWER**. Granularity-dead. See `the atomic grid-barrier result in `docs/DEAD_ENDS.md``.
- **TLX (the leaders' filename route): INFEASIBLE for us.** NOT on Modal (Meta triton fork; no pip pkg, only a from-source build); eval has no nvcc for raw PTX. We can't develop/debug it (`modal_tlx_probe.py`).

## Why the leaders are 4.6× ahead (first-hand, this session)
SAME algorithm (blocked Householder, flat output). The gap = an integrated **warp-specialized PERSISTENT tcgen05 engine**: a peak-fed trailing GEMM (TMA producer warps + multi-stage pipeline + 2-SM tiles) WITH the latency-bound panel hidden behind it. the then-leader (1558µs, plain `submission.py`) proves the pure-Gluon route EXISTS — not toolchain-blocked, just multi-week expert engineering depth we haven't built.

## THE remaining crux (everything else is dead): a panel that is BOTH fast AND co-residable
design A died on exactly this. Two unexplored cracks:
1. **MAGMA-style register-file panel** (one thread/row, in-register geqr2): fast like the register panel BUT low-enough register to co-reside with the async trailing worker. **NOT YET TESTED — this is the cheapest next crux experiment (~½ day): build it, measure n_regs + speed vs `gluon_panel.py`. GO if regs ≤ ~64/thread AND speed ≈ register panel → design A revived. NO-GO → only design B remains.**
2. **design B (inter-CTA persistent)**: panel-CTA ‖ trailing-CTA co-resident on an SM via occupancy (no register sharing). The validated atomic barrier provides the coarse-grained per-matrix coordination. Major multi-week build.

## To reach the leader-class engine (if the crux passes): NEVER-USED primitives still needed
TMA async bulk-load (producer warps + tensor descriptors + 128B swizzle), multi-stage software pipeline (N-buffer producer/consumer mbarrier protocol), 2-SM cooperative tiles (`two_ctas`), PLUS the crux panel, PLUS weeks of per-shape tuning. ALL present in the recon'd Gluon Blackwell API (not toolchain-blocked) — but deep.

## Reusable BUILT + verified assets (do NOT rebuild)
- `experiments/gluon_panel.py` — bit-faithful Gluon Householder panel port (correctness reference).
- `experiments/gluon_panel_async.py::_async_trail_part` — low-level async tcgen05 tiled trailing worker (+ the no-`tl_dot`-in-worker lesson).
- `experiments/gluon_lowlevel_dump.py` / `gluon_ws_dump.py` / `gluon_smem_api.py` — the async tcgen05 recipe + `gl.warp_specialize` mechanics + smem-descriptor API (slice/index/load/store; runtime-`range`+`.index` lowers regs, `static_range` does NOT).
- `experiments/tri_grid_barrier.py` — validated atomic grid barrier (~2.6µs) for design-B coarse coordination.
- `experiments/gluon_panel_smem.py` — smem-streaming panel (too slow, but the low-reg streaming trick is reusable).

## Recommendation
Deployable space is measured-exhausted; **V5 5915 is the clean best**. The only path to materially close the leader gap is the multi-week warp-spec persistent engine, gated on the register-file-panel crux (#1). If NOT investing multi-weeks → ship V5 (optionally the solve→fp32 reseed-DQ robustness variant, `experiments/cand_fp16x3_solvefix.py` pattern on V5). **The autonomous plan in `docs/NEXT_SESSION_PROMPT.md` starts with the ≤½-day crux gate so a cheap test decides whether the multi-week build is worth it.**

---
**⚠️ Everything BELOW this line is the staged Gluon-overlap plan from the START of the 2026-06-26 session — now SUPERSEDED by the findings above (design A is dead). Kept only for historical detail / asset pointers.**

## TL;DR (HISTORICAL — superseded)
- **State:** `submission.py` = V5 tf32x3 = **5915µs official (confirmed best)**. Leader **1292µs (~4.6×)**.
- **The gap is the PANEL** = **~41% of n=512 runtime, latency-bound by the sequential reflector-reduction chain** (profiled: panel 41 / gram 15 / trailing 26 / solve 18).
- **The (then-)un-dead lever:** hide the panel behind the trailing via a fused Gluon warp-spec kernel. **→ NOW MEASURED DEAD (design A); see the top of this doc.**

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
- Keep panel reductions FP32 (low-precision panel corrupts reflectors — proven dead). Keep the T-factor solve fp32-safe (the solve-tf32 artifact — see the tf32-solve artifact finding in `docs/DEAD_ENDS.md`).

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
- Test harness: `modal_lab.py` (correctness/compare/profile) + `modal_microbench.py --script <x>` (single B200 script). Modal path: `modal`.

## Honest risk register
- No high-level `warp_specialize` → hand-rolled mbarrier warp-spec is the core difficulty and the Phase-0 risk.
- The single-CTA-trailing penalty may force design (B) persistent (harder than (A)).
- Pure-Gluon may have a ceiling above the leader's 1292 (they may use something beyond stock Gluon; the docs are medium-confidence on this).
- Payoff is bounded ~5-15% geomean even if it fully works — this is a research investment, not a sure thing. **The GO/NO-GO gates at Phase 0/1 are there so you spend ≤1.5 days before knowing.**

## Do-NOT-retry (waste of time — see `docs/DEAD_ENDS.md`)
fp8/fp4/mxfp/nvfp trailing; 1×TF32; fp16x3; raw-PTX/load_inline (no nvcc); streams/cooperative-launch (rule); conditioning-routing/early-stop (reward-hack DQ); lowprec-panel; mega-kernel single-CTA trailing; naive (library-primitive) CholeskyQR; TSQR→flat reconstruction; per-column panel streamline.
