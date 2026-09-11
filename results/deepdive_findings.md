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

## 4. ★ PAYOFF — fused M-builder (option b): Modal +5.2%, bit-identical, transfer-reliable
`experiments/cand_gluefuse.py` (= V5 + `_build_M`): one Triton kernel builds `M = triu(G,1) +
diag(1/tau)` (b<=128, one CTA/matrix), replacing V5's ~7-launch elementwise chain (eq, reciprocal,
where x2, full, triu, diag-copy). **Universal, BIT-IDENTICAL (22/22, margin preserved exactly).**

3-way lab compare (same container, 12 reps), baseline = V5:
  cand_gluefuse (M-fusion)      geomean **0.9478x = +5.2%**  | implicitV->small-n  0.9940x = +0.6%
  per-case (M-fusion): n176 -12.8%, n352 -10.2%, n512 -2.6% (all 4), n1024 -5.8..-6.8%,
                       n2048 -9.0% (low batch=8, occupancy-bound), n4096 0% (geqrf path).

WHY it beats the launch-gap prediction (helps even GPU-bound n=512): it cuts BOTH ~6 kernel
LAUNCHES/apply AND ~6 redundant HBM passes over the [b,b] G/M tensors (each torch glue op re-read/
wrote them). So it wins both the launch-bound regime (small-n, low-batch n=2048) AND the
glue-traffic at n=512. The small-n implicit-V (+0.6%) is dwarfed by it.

WHY it should TRANSFER (unlike fp16x3 / implicit-V-n512): purely STRUCTURAL (fewer dispatches +
fewer bytes moved), BIT-IDENTICAL result, zero precision/ALU trade. Asymmetric: ~0 downside (one
tiny kernel replacing seven), +3-5% likely official. **The cleanest deployable win found in the
project. Promoted to submission.py ON THIS BRANCH (main untouched = V5). MUST confirm via gpumode.**

Next (untested, further glue): fuse V=tril+fill into 1 kernel (2->1, bit-identical); fold _build_M
into the gram epilogue (8->1); a launch-cut for n=1024's 1xTF32 path. All bounded vs the M-fusion +5.2%.

## 5. ★ option (b) — deeper glue fusion: +2.25% MORE on top of V8 (~+7.4% cumulative vs V5)
`experiments/cand_gluefuse2.py` adds two more bit-identical fusions on top of V8's `_build_M`:
- `_build_V`: V = strict-lower(P)+unit-diag in ONE launch (replaces torch.tril + diagonal.fill_, 2->1).
- `_build_Mt`: builds M^T (LOWER-tri) DIRECTLY, so `solve_triangular(M_T, W, upper=False)` needs no
  `.transpose(1,2)` — which cuSOLVER was materializing as a per-apply `direct_copy`. 7->1 AND kills the copy.

3-way lab (baseline = branch submission.py = V8/M-fusion): cand_gluefuse2 **0.9775x = +2.25% vs V8**,
per-case n176 -3.3 / n352 -3.0 / **n512 -3.2 (all 4)** / n1024 -2.4 / n2048 -0.2 / n4096 0. 22/22.
The n=512 gain is the M^T-no-transpose removing the cuSOLVER copy (real op+traffic, even GPU-bound).
**Cumulative vs V5: 0.9478 (M-fusion) x 0.9775 (deeper) ~= 0.9265 -> ~+7.4% Modal**, ALL structural
+ bit-identical (margin preserved). Promoted to branch submission.py (= "V9"); V8 archived per request.

ARCHIVES: milestones/submissionV8_gluefuse_modal5402.py (A = M-fusion, +5.2%) + a standalone copy at
<local scratch>; milestones/submissionV9_gluefuse2_modal5298.py (the (b)
result, ~+7.4%). main UNTOUCHED = V5 5915 official. NEXT: gpumode-confirm the cumulative win (bit-identical
-> zero correctness/margin risk; structural -> should transfer far better than the fp16x3/implicit-V artifacts).
