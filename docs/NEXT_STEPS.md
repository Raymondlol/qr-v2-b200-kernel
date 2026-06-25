# Next steps — the evidence-based path forward (updated after the full investigation)

## Immediate
**Submit `submission.py`** (m4 panel-warps + fused-panel ib=64/NB=128 for n=512 + route-n=2048) to confirm the official number (~8000–8200 est, from 8580). `grep -niE "stream|graph" submission.py` → must be empty.

## Where we are
Best confirmed **8580 µs official** (~rank 90); leader **1332 µs**. We exhaustively mapped the landscape this session (see `DEAD_ENDS.md`, `HOW_LEADERS_ARE_FAST.md`). The verdict: **we are on the correct algorithm** (blocked compact-WY Householder, native flat (H,τ)); the ~6× gap is **kernel engineering**, and the *safe, cheap* levers are now exhausted.

## What's left, ranked by EV (all are hard / uncertain)

### 1. Warp-specialized tensor-core GEMM engine for the tf32x3 trailing — THE gap, biggest lift
We run the trailing at **~1.5 % of TF32 peak** (`torch.matmul` / basic Triton `tl.dot`; expanded autotune did NOT help). The leaders run a `tcgen05.mma` warp-specialized persistent kernel (TMA producer / MMA / epilogue warps, 128 B swizzle, 2-SM M256 tiles) — gau.nernst-class **raw PTX**, or **TLX** (Triton Low-level eXtensions, the `triton_tlx` board filename) as the Triton-native route. This is the bulk of the 6×. Risk: a **pure-Triton stack may have a ceiling above 1332 µs** (the >600 µs CUDA-over-Triton gap on the board). Effort: days, research-grade. **This is the only path to truly contend.**

### 2. n=4096 cooperative multi-CTA panel — breaks the last geqrf-floor case (~9 %)
n=4096 b=2 = 52 ms via geqrf (only 2 one-CTA panels). A cooperative multi-CTA panel (atomic-counter spin-barriers, single stream — Triton has no clean grid-sync) could fill the GPU. Research-grade, ~20–30 % success odds (see the shelved plan `~/.claude/plans/mega-kernel-mighty-pony.md`). n=2048 is already won by routing.

### 3. Risky ~3 %: fused 2-term-rounded trailing (`experiments/cand_fused2br.py`)
Passes all 12 on benchmark seeds, ~5 % faster on n=512, but only **1.4× margin** on mixed@640 → reseed-DQ risk. Only if you accept the risk (can resubmit if it fails).

### 4. Register-file panel (MAGMA-style) + recursive blocking — incremental, uncertain.

## Dead — do NOT spend time here (proof in DEAD_ENDS.md)
Mega-kernel / single-CTA trailing; naive CholeskyQR; fp8/nvfp4; lowprec-panel; TSQR-output; sub-tf32x3 precision (1×TF32 / 2-term); expanded GEMM autotune; stream/graph anything (illegal).

## Validation discipline (Modal lies on tf32 absolute speed)
`modal run modal_app.py --submission <f> --stress` (22/22) before any submit; **always gate mixed at batch=640** (the STRESS probe is batch-16 and misses the 1×TF32 failure — the benchmark cases gate it). Compare candidates in the SAME modal run (count=1 noise). Confirm absolute speed via real submission. Panel/FP32 changes: Modal is reliable and conservative (real gain ≥ Modal).
