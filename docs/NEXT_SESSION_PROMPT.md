# Next-session autonomous startup prompt (qr_v2, B200)

Paste the block below to start the next session. It is written so the agent can run **largely
autonomously** through GO/NO-GO gates without hand-holding. (User-facing reports in Chinese; thinking/code/execution English.)

---

You are continuing the GPU MODE **qr_v2** batched-Householder-QR kernel competition (B200, geomean of
12 cases, deadline 2026-06-30). **Read in order before doing anything:** `CLAUDE.md` (auto-loaded) →
`docs/HANDOVER_NEXT_SESSION.md` (FINAL STATE after the 2026-06-26 research session) → `docs/DEAD_ENDS.md`
(do-not-retry). The auto-memory mirrors the durable facts.

## Where things stand (do NOT re-derive or re-explore)
- `submission.py` = **V5 tf32x3 = 5915µs official = the confirmed best, UNCHANGED.** Leader 1292µs (4.6×).
- The 2026-06-26 session **measured-DEAD every overlap + large-n + deployable lever**: Gluon warp-spec
  OVERLAP design A (panel can't co-reside — register panel blows the 16-warp budget, smem panel
  2.56–6.74× too slow); CholeskyQR (LinAlgError + conditioning-routing is the v2-forbidden reward-hack);
  multi-CTA cooperative panel (atomic barrier WORKS ~2.6µs but per-column granularity is 7–8× too coarse);
  recursive-blocking (0.976×); cuBLAS-trailing (0.866×); TLX (infeasible on Modal). Full proof in DEAD_ENDS.
- **The leaders' edge is engineering, not algorithm**: an integrated warp-specialized PERSISTENT tcgen05
  engine (TMA producer warps + multi-stage pipeline + 2-SM tiles) with the latency-bound panel hidden
  behind the trailing. gau.nernst (1558µs, plain submission.py) proves the pure-Gluon route exists.

## Reusable BUILT + verified assets (do NOT rebuild — read them)
`experiments/gluon_panel.py` (bit-faithful Gluon Householder panel = correctness ref),
`gluon_panel_async.py::_async_trail_part` (low-level async tcgen05 tiled trailing worker; worker MUST NOT
use `tl_dot` — it CTA-barrier-couples), `gluon_lowlevel_dump.py`/`gluon_ws_dump.py`/`gluon_smem_api.py`
(async tcgen05 recipe + `gl.warp_specialize` mechanics + smem-descriptor API: runtime-`range`+`.index`
lowers regs, `static_range` does NOT; partitions must WRITE outputs not return tensors), `tri_grid_barrier.py`
(validated atomic grid barrier for design-B coordination), `gluon_panel_smem.py` (the low-reg streaming trick).

## YOUR MANDATE: the one remaining crux, then decide the multi-week question
Everything dead reduces to ONE unsolved problem: **a panel that is BOTH fast AND low-register enough to
co-reside with the trailing.** Drive this staged plan autonomously; each stage is a GO/NO-GO gate.

### STAGE 0 (the crux, ~½ day) — MAGMA-style register-file panel. START HERE.
Build a register-file Householder panel: one thread owns one (or a few) ROWS of the [M, ib] tile in
registers (MAGMA `geqr2`-style), so the per-column norm/dot are warp/block reductions over the
thread-distributed rows, and the rank-1 update is each thread updating its own rows — instead of the
current full-tile-in-registers layout (~128 regs/thread) OR the smem layout (too slow). Goal: a layout
that is **fast (≈ the register-resident `gluon_panel.py`) AND low-register (≤ ~64 regs/thread)** so it
fits with the 8w/128r async trailing worker in 16 warps.
- Verify **bit-faithful** vs `gluon_panel.py::panel_ref` (relerr < 1e-4) at M ∈ {512,384,256,128}, b=64.
- Report compiled `n_regs`/`n_spills` and panel time vs `gluon_panel.py` at grid=640, M=512.
- **GATE: GO if regs ≤ ~64/thread AND time ≤ ~1.3× the register panel → STAGE 1 (design A revived).
  NO-GO (can't get both) → the in-CTA panel is fundamentally un-co-residable → STAGE 2 (design B) or ship V5.**

### STAGE 1 (if Stage 0 GO, ~2–3 days) — design-A overlap with the register-file panel
`gl.warp_specialize([(regfile_panel, ...), (_async_trail_part, ...)], [8],[128])` at the real n=512 shapes,
grid=640. Verify eff > 0.5 (panel‖trailing nets >1.1× vs sequential). Then: the look-ahead schedule
(trailing(k) look-ahead block first, then panel(k+1)‖trailing-rest(k)), fold gram+solve (keep solve
fp32-safe), shape-route to n≤512 ONLY, build incrementally (1 matrix vs geqrf → 640 → 22/22), gate
mixed@640 worst-of-640 margin ≥ 1.83 SEPARATELY, then SUBMIT to confirm official.

### STAGE 2 (only if Stage 0 NO-GO + user commits multi-weeks) — design B persistent inter-CTA engine
Separate panel-CTAs ‖ trailing-CTAs co-resident on an SM via occupancy (no register sharing). Use the
validated `tri_grid_barrier.py` atomic barrier for coarse per-matrix coordination. This ALSO needs the
peak-fed trailing engine (never-used primitives: TMA async bulk-load + producer warps, multi-stage
software pipeline with producer/consumer mbarriers, 2-SM `two_ctas` tiles — all in the Gluon Blackwell
API). Major multi-week build; learn the primitives incrementally with standalone microbenches first.

### FALLBACK: ship V5 (5915µs) — the deployable space is measured-exhausted; this is the clean default.
Optionally the solve→fp32 reseed-DQ robustness variant (`experiments/cand_fp16x3_solvefix.py` pattern on V5).

## DISCIPLINE (hard rules)
1. **`grep -niE "stream|graph" submission.py` MUST be empty before any submit** (naive substring scan;
   even comments count — the atomic barrier is grep-clean but don't write "stream"/"graph" in comments).
2. **Measure on B200, gate every "win" on a REAL gpumode submission** — Modal codegen gains can fail to
   transfer (the fp16x3 lesson: Modal +3.6% → official wash). Treat Modal as ~4% optimistic.
3. **mixed@640 worst-of-640 margin** is the binding precision constraint (the lab's batch-16 gate MISSES
   it — probe separately, `microbench_1x_reseed.py` pattern). Keep ≥ ~1.83×. Panel reductions stay FP32.
4. **No conditioning-routing / early-stop / output-caching** (v2-forbidden reward-hacks; reseed-DQ).
5. **Be deliberate with B200 $** (user's money, ~cents/run) — prefer same-run apples-to-apples; PROFILE
   before testing candidates; gate cheaply before multi-day builds.
6. **Each phase is a GO/NO-GO gate ≤1–1.5 days** — don't blindly multi-day build; report the gate verdict.
7. **Do NOT re-explore DEAD_ENDS** (fp8/fp4/mxfp/1×TF32/fp16x3/raw-PTX/TLX/streams/conditioning-routing/
   lowprec-panel/mega-single-CTA-trailing/naive-CholeskyQR/TSQR-recon/cooperative-panel/recursive-blocking/
   cuBLAS-trailing/smem-streaming-panel — all measured-dead).

## ENV
Mac, no local GPU; B200 via Modal (absolute path): `/Users/raymond/Downloads/SubPY/.modalenv/bin/modal`.
- Single script: `… modal run modal_microbench.py --script experiments/<x>.py`
- Lab compare: `… modal run modal_lab.py --mode compare --subs "submission.py,experiments/<cand>.py"`
- Correctness gate: `… modal run modal_lab.py --mode correctness --subs "experiments/<cand>.py"`

**First action: read the 3 docs, then build the Stage 0 register-file panel and run the gate.** Report
the gate verdict (Chinese) and proceed or stop per the result.
