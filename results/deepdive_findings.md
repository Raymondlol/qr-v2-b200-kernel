# Deep-dive into the profiling findings (branch `profiling-deepdive`, off main; submission.py UNTOUCHED = V5)

Pursued the two predictive angles from `docs/PROFILING.md` §8 — **roofline-by-shapes** (free) and
**launch-gap** (1 B200 run) — then validated the implied lever. Three concrete results:

## 1. Roofline-by-shapes (`experiments/deepdive_roofline.py`, free, exact shape arithmetic)
Replicates `_factor_custom` to enumerate every GEMM (M,N,K) + AI = useful-FLOP/HBM-byte vs the
B200 tf32x3 ridge ~20.6. flop-weighted AI rises monotonically with n:
| n | GEMM launches | flop-wt AI | regime |
|---|---|---|---|
| 176 | 6 | **12.4** | fully MEMORY-side (6/6 GEMMs < ridge), 0.01 GFLOP total — tiny |
| 352 | 15 | 20.8 | mixed (10/15 mem) |
| 512 | 21 | **24.3** | COMPUTE-side (heavy fat-trailing GEMMs AI 24–40 dominate flops) |
| 1024 | 63 | 33.4 | compute-side |
**→ PREDICTIVE: n=512's heavy GEMMs are compute-bound → a traffic cut (implicit-V) cannot help
there → predicts the V7 official regression. The traffic/memory lever lives in small-n.**

## 2. Launch-gap (`modal_deepdive_launchgap.py`, B200; wall-clock vs Σ GPU-kernel self-time)
gap = wall − GPU-busy = time the GPU sat IDLE waiting for CPU to dispatch the next of ~40–150 kernels:
| case | wall µs | GPU µs | **idle gap** | launches/it | dominant |
|---|---|---|---|---|---|
| n=176 b=40 | 752 | 388 | **48.4%** | 41 | panel 65 / glue 15 |
| n=352 b=40 | 2010 | 1213 | **39.7%** | 106 | panel 57 / glue 14 |
| n=512 b=640 | 12265 | 12037 | **1.9%** | 148 | panel 44 / tf32x3 28 |
| n=1024 b=60 | 7504 | 5951 | **20.7%** | 442 | panel 49 / glue 18 |
**→ DECISIVE: small-n burns 40–48% of wall-clock GPU-IDLE on kernel dispatch; n=1024 21%; n=512 only
1.9% (GPU-saturated). The deployable lever for the geomean-weighted small cases is FEWER KERNEL
LAUNCHES — a STRUCTURAL reduction that transfers across HW (unlike codegen/precision tricks).**

## 3. Validation: implicit-V routed to the launch-bound regime only (n≤352)
`experiments/cand_implicitV_smalln.py` (V5 + implicit-V apply gated to `H.shape[1] <= 352`; n≥512
stays byte-identical V5). Lab compare vs V5 (same container, 12 reps):
  geomean **1.009× (+0.86%)**; significant wins land EXACTLY on **n=176 dense −4.2%, n=352 dense −3.2%**;
  **n=512 flat** (routed away → no regression, the inverse of V7 which regressed there).
This confirms the thesis: implicit-V's value was always a LAUNCH cut on launch-bound cases; routing
it there (not n=512) flips net-regression → small clean win, and is mechanism-transferable.

## Verdict / where the remaining deployable EV is
- **n=512 (the 4 big cases): GPU-saturated/compute-bound (1.9% gap, AI 24).** No launch/traffic lever;
  only the panel-overlap engine (design-A/B, measured-bounded) or the leader's warp-spec engine.
- **small-n + n=1024 (launch-bound, 20–48% idle): the ONLY remaining deployable territory.** Lever =
  cut kernel LAUNCHES. implicit-V captures a sliver (~2 launches/apply → +0.86%). The fuller lever:
  fuse the per-apply glue (the inv_τ chain where/reciprocal/where/fill + triu/diag-copy = ~6 launches/
  apply, + the DtoD memcpy) into 1 kernel, and/or implicit-V for n=1024's 1×TF32 path. Bounded (~2–5
  small/medium cases) but transfer-reliable. **MUST still confirm on a real gpumode submission** —
  Modal's CPU-dispatch may inflate launch-bound gains (gVisor); but a launch cut can't REGRESS V5
  (small-n only, n=512 untouched).
- The profiler is now PREDICTIVE: roofline (compute-vs-memory) + launch-gap (GPU-vs-dispatch) together
  would have flagged V7's n=512 failure BEFORE building it.

submission.py UNCHANGED = V5 5915µs official. All artifacts on branch `profiling-deepdive`.
