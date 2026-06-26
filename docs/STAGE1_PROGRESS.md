# Stage 1 (design-A fused factorization) — PROGRESS + BUILD PLAN

> Branch `gluon-regfile-panel`. Continues after Stage 0 reopened design A (`docs/DEAD_ENDS.md`
> top banner). submission.py UNCHANGED = V5 tf32x3 5915µs. This doc = the validated foundations,
> the (favorable) economics, and the precise remaining build, so the full assembly can proceed.

## What Stage 1 has VALIDATED (all GO, committed, each cheaply gated on B200)

| # | piece | result | file |
|---|---|---|---|
| 0 | MAGMA row-distributed **ib=16 panel** | 108 regs / **0 spills** / 0.40× the spill-bound default | `stage0_regfile_panel.py` |
| 0.5 | `warp_specialize(panel‖async worker)` co-resides | compiles+correct, overlap **eff 0.42–0.58, 1.18–1.26×** | `stage0b_coresident.py` |
| 1A | REAL **tf32x3 async tcgen05** trailing GEMM | correct 2.5e-6, **1.11×** `_bmm3` (BM=64, smem-capped) | `stage1a_tf32x3_async.py` |
| 1B | compact-WY apply via async tcgen05 | correct 3e-6; K-loop = reg-accumulate per chunk | `stage1b_compactwy.py` |
| 1C2 | the apply is **2 GEMMs not 3** (VT=V@T^T) | async 2-GEMM apply **competitive/faster** (0.76–1.1×), NOT 1.4× | `stage1c2_2gemm_apply.py` |

### Key corrections discovered while building
- The default register panel **spills 310×** (255 regs); the **MAGMA row-distributed layout**
  (each thread owns COMPLETE rows → in-place rank-1 update, no transient full-tile temporary)
  kills the spills. Narrow **ib=16** brings regs to 108 → co-residable with a 128-reg worker
  (108+128=236 ≤ 256/thread-pair in the 16-warp / 64K budget).
- The async tf32x3 split is naive 3-term (~19–20 bit, relerr 2.5e-6) vs Triton's
  `input_precision="tf32x3"` (~22 bit, 2e-7). A 4th term (Al@Bl) does NOT help (split-limited).
  **Likely absorbed by the solve→fp32 robustness fix (800× margin); re-check at the mixed@640 gate.**
- The K-loop must **register-accumulate per chunk** (each chunk computes its full product into TMEM,
  added to a reg accumulator). Cross-chunk TMEM `use_acc` across re-init'd mbarriers = garbage (0.875).
- The compact-WY apply is **2 GEMMs** (`W1=V^T@C`; `C-=VT@W1` with `VT=V@T^T` precomputed once),
  NOT 3. The W1 global round-trip is ~3% (cheap) → the resident-W1 smem fusion is unnecessary
  (and OOMs at 262KB > 228KB anyway from the tf32x3 hi/lo operand doubling).

## Economics (REVISED — favorable)
With the apply ~1.0× (1C2) and the panel‖trailing overlap eff 0.42–0.58 (0.5):
`designA(n512) = (panel+apply) − eff·((panel+apply) − max(panel,apply)) ≈ 1.2–1.33×`.
Profile panel 41 / gram 15 / trailing 26 / solve 18 → overlapping panel‖trailing hides the
latency-bound panel behind the trailing → **~+7–9% geomean → ~5200–5350 Modal** (V5 = 5678 Modal).
**sub-5000 (= ~12% over V5) is at/just beyond design-A's ceiling** — would need eff≳0.6 AND clean
integration AND ideally extending overlap to n=1024 (penalty 2.31× there → unfavorable, see memory).
Honest: design A is the one live lever and is worth building; the leader's 4.6× (1292µs) is a
multi-week warp-spec persistent engine, out of scope.

## THE remaining build (multi-session) — precise plan
The one gating new piece = an **in-kernel super-panel factor** (the panel partition's body), then
fuse + look-ahead + validate. Reuse everything above.

1. **`_superpanel_factor` (the crux build).** One CTA factors a [m, NB=128] super-panel as 8 × ib=16
   sub-panels. Per sub-panel j: (a) rowmagma factor cols [j:j+16] (HAVE, `_panel_rowmagma`); (b) the
   **T-factor T16** for those 16 reflectors via **LARFT** (in-kernel; gram16 = V16^T@V16 is a tiny
   GEMM, then the sequential 16-col T recurrence — the hardest new code, ~20 lines, one-warp); (c)
   the **blocked within-apply** of block [j:j+16] to super-panel cols [j+16:128] (the 2-GEMM async
   apply, HAVE — `stage1b/1c2`). Blocked (not unblocked) is REQUIRED: unblocked within-applies make
   the panel partition too slow → can't beat V5. Gate: correct vs `torch.geqrf` on a [m,128] block.
2. **Worker = far-trailing 2-GEMM apply** (HAVE, 1C2): `W1=V^T@C_far`, `C_far -= VT@W1` over the
   columns [k+2·NB : n]. Both async tcgen05 (no `tl_dot` — couples partitions).
3. **Fuse:** `gl.warp_specialize([(_superpanel_factor, next-block args), (far_trail_worker, cur-block
   args)], [8], [128])`. Panel = 8 warps (CUDA-core reductions + small within-apply GEMMs); worker =
   8 warps (big far-trailing GEMMs). NOTE the new contention risk: the panel's within-apply GEMMs now
   use tensor cores too (Stage 0.5's eff was a PURE-reduction panel) — re-measure eff here; if the
   within-apply contention tanks it, move within-applies to CUDA-core/streamed or shrink NB.
4. **Look-ahead schedule** (Python loop in `_factor_custom`, n≤512 only): factor super-panel 0
   (existing path); then per k: torch-apply block(k) to the look-ahead cols [k+NB:k+2NB] (so panel
   (k+1) can start) → fused{ panel(k+1) ‖ far-trailing(k) on [k+2NB:] } → torch gram+T(k+1).
   Keep gram + T-solve as torch (fast cuBLAS/cuSOLVER) OFF the overlap critical path.
5. **Validate + submit:** 1-matrix vs geqrf → batch 640 → 22/22 (`modal_lab.py --mode correctness`);
   **mixed@640 worst-of-640 margin ≥ 1.83 probed SEPARATELY** (lab batch-16 gate misses it — use the
   solve→fp32 fix if the async tf32x3 ~19-bit trailing erodes margin); `modal_lab.py --mode compare`
   geomean; `grep -niE "stream|graph" submission.py` empty; **SUBMIT to gpumode** (Modal gains may
   not transfer — fp16x3 lesson).

## Reusable assets BUILT this stage (do NOT rebuild)
`experiments/stage0_regfile_panel.py::_panel_rowmagma` (the co-residable ib=16 panel),
`stage0b_coresident.py` (the warp_specialize co-residence harness),
`stage1a_tf32x3_async.py::_tf32x3_gemm` + `stage1b_compactwy.py::_tf32xN_gemm` (K-looped async
tcgen05 tf32x3 GEMM, reg-accumulate; SUB epilogue), `stage1c2_2gemm_apply.py` (the 2-GEMM apply).
Worker template: `gluon_panel_async.py::_async_trail_part`.

## Risk register for the remaining build
- **In-kernel LARFT (T16)** is the hardest new code (sequential recurrence in SIMT). Fallback: compute
  T per super-panel with a tiny separate kernel/torch BEFORE the fused kernel (the within-apply then
  reads a precomputed T) — costs a small serial step but de-risks the partition body.
- **Tensor-core contention** panel-within-applies vs worker far-trailing (untested; Stage 0.5 panel was
  pure-reduction). If eff drops, shrink the within-apply share (smaller NB) or stream within-applies.
- **Precision** (async ~19-bit trailing) vs mixed@640 — gate separately, lean on solve→fp32.
- **Modal→official transfer** — confirm any win with a real gpumode submission.
