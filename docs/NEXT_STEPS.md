# Next steps — the evidence-based path forward (updated after the full investigation)

## Immediate
**Submit `submission.py` to gpumode to confirm ~5700 official** (5678 Modal; Modal≈official now). It = V4 (7788 confirmed) + this session's gram→1×TF32, adaptive-ib, glue-fusion, panel nw=8 = 1.38×, 22/22. `grep -niE "stream|graph" submission.py` → empty. Run the lab harness for any new candidate (`docs/METHODOLOGY.md`), NOT the old loop.

## Where we are
Last confirmed **7788 µs official** (V4); `submission.py` = **5678 Modal ≈ ~5700 (≈rank 61)**; leader **1292 µs**. The CHEAP deployable levers are now exhausted (gram/adaptive-ib/glue/panel-nw8/blocking). Profiler verdict: the bottleneck is the **PANEL (~40-50%, latency-bound by the sequential reflector reduction chain)** — NOT the trailing GEMM (21-26%). The remaining gap is research-grade: a fundamentally different panel kernel (my current one is tapped in Triton; a two-level peer hits 3352 so ~1.7× headroom likely exists in-family) OR the deployable-but-bounded Gluon tcgen05 engine (branch `gluon-tcgen05`).

## What's left, ranked by EV — UPDATED with the 2026-06-25 engine baseline

**Measured trailing-GEMM efficiency (`experiments/microbench_floor_and_engine.py`, B200):** on the REAL trailing shapes even *cuBLAS 1×TF32* reaches only **21–34 % of tf32 peak on the FAT updates** (n=512 fat 236 TFLOP/s/21 %; n=1024 fat 377/34 %) and a dismal **6–7 % on the THIN K=32 within-panel updates** (75 and 70 TFLOP/s). Our fused tf32x3 Triton kernel runs at **~10 % of its ÷3 ceiling and is ~6× slower than cuBLAS 1×TF32** on the fat shapes (n=512 fat: cuBLAS 136 µs vs ours 841 µs — ~2× slower even after normalizing for tf32x3's 3× work). So the old "~1.5 % of peak" figure was too pessimistic, but the headroom is real and it is **structured** — attack it in this order:

### 1a. Recursive / bigger-K blocking — kill the 6–7 % thin K=32 updates (EASIEST, structural)
The two-level scheme does many narrow K=32 within-panel updates that run at ~6 % even on cuBLAS. Recursive blocking (square up the trailing GEMMs / widen K) converts these into the fat shapes that already hit 21–34 %. No new kernel — a blocking change. Highest EV-per-effort first step.

### 1b. Best-achievable trailing impl — is our fused tf32x3 kernel even the right call?
Our fused kernel is ~2× below cuBLAS on the fat shapes (after the tf32x3 factor). Re-test: 3×cuBLAS hi/lo split tf32x3 vs the fused Triton kernel vs a better-tuned/larger-tile Triton tf32x3, on the real shapes, same run. The journal's "fused beat the split" was an end-to-end Modal claim; the raw-GEMM gap (6×) says re-measure. Cheap, may give a quick 1.5–2× before any hard kernel work.

### 1c. Warp-specialized / raw-PTX trailing engine — ATTEMPTED, BUILT, CONFIRMED NOT WORTH IT (2026-06-25)
Built a correct cp.async double-buffered tf32x3 `mma.sync.m16n8k8` GEMM via load_inline (`experiments/cuda_gate{1,2,3}.py`, `modal_cuda.py`): ~1.2× over Triton's tf32x3 (51 TF/s n=1024). DEAD anyway — see DEAD_ENDS for the full writeup. Three compounding reasons: (1) trailing GEMM is only 26%/21% of n=512/n=1024 and n≥1024 already uses faster 1×TF32, so it only helps n=512 → **~1.5–2% geomean**; (2) **TLX absent**, persistent/TMA Triton variants worse; (3) **load_inline needs nvcc on the competition machine** (torch-only env doesn't have it) → likely undeployable. **The gap is NOT a faster trailing GEMM.** The real trailing lever is the DEPLOYABLE **bigger-K shape** (1a/1b): K-poor shapes cap even cuBLAS at 20–34%; bigger NB (now that adaptive-ib absorbs the narrow-update cost) lifts the existing kernels' efficiency — UNTESTED, cheap, the recommended next trailing experiment. The biggest single n=512 phase remains the **panel (41%, latency-bound by the sequential reflector dependency)** — needs a different panel algorithm, not a faster GEMM.

### 2. n=4096 floor-breaker — PARKED (bounded ~3 %, needs 2–3 custom kernels)
n=4096 b=2 = 52 ms via geqrf. Floor-probe showed a *blocked* Cholesky already beats cuSOLVER 2.86× (8.9 ms) — so a custom CholeskyQR floor-breaker is not dead — but the full path (CQR3-safe blocked chol + custom unpivoted-GETRFNP reconstruction) is ~35 ms = ~1.5× on ONE of twelve cases ≈ ~3 % geomean, for 2–3 hand-written kernels. Lower EV than the engine; revisit only if a custom blocked tensor-core Cholesky gets built anyway. (cooperative multi-CTA Householder panel is the alternative floor-breaker — same hardness, also parked.)

### 3. Risky ~3 %: fused 2-term-rounded trailing (`experiments/cand_fused2br.py`)
Passes all 12 on benchmark seeds, ~5 % faster on n=512, but only **1.4× margin** on mixed@640 → reseed-DQ risk. Only if you accept the risk (can resubmit if it fails).

### 4. Register-file panel (MAGMA-style) + recursive blocking — incremental, uncertain.

## Dead — do NOT spend time here (proof in DEAD_ENDS.md)
Mega-kernel / single-CTA trailing; naive CholeskyQR; fp8/nvfp4; lowprec-panel; TSQR-output; sub-tf32x3 precision (1×TF32 / 2-term); expanded GEMM autotune; stream/graph anything (illegal).

## Validation discipline (Modal lies on tf32 absolute speed)
`modal run modal_app.py --submission <f> --stress` (22/22) before any submit; **always gate mixed at batch=640** (the STRESS probe is batch-16 and misses the 1×TF32 failure — the benchmark cases gate it). Compare candidates in the SAME modal run (count=1 noise). Confirm absolute speed via real submission. Panel/FP32 changes: Modal is reliable and conservative (real gain ≥ Modal).
