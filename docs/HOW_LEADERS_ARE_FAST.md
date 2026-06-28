# How the top qr_v2 submissions are ~10× faster (credible, triangulated)

Method: 9-agent research workflow (academic SOTA + competition OSINT + backwards-reasoning + engineering + 3 adversarial refutations that ran numpy checks) cross-checked against this session's B200 microbenchmarks. Top-3 solutions are PRIVATE (comp ends 2026-06-30), so this is INFERENCE, but it survived adversarial refutation. Overall confidence: **medium-high on the algorithm, medium on the engineering specifics.**

## Headline: it's the SAME algorithm we use — the gap is ENGINEERING, not algorithm
The leaders use **shape-routed right-looking blocked compact-WY Householder QR with native flat (H,tau) output** — exactly our approach. They are NOT using a fundamentally different algorithm. We are on the right track; we're losing on kernel engineering.

### Why NOT CholeskyQR / TSQR + reconstruction (ruled out)
> **MEASURED CORRECTION (2026-06-25).** The "CholeskyQR is dead" call below was originally generalized from the n=512 b=640 microbench ONLY. It is now measured at all four floor shapes and the conclusion *survives but for richer reasons* (see `DEAD_ENDS.md` for the table): at n=4096 b=2 cuSOLVER `cholesky` of one matrix = **24.8 ms = ~1.8 TFLOP/s = the SAME critical-path floor as geqrf** — so library-primitive CholeskyQR does NOT escape the floor (it just swaps a serial Householder panel for a serial Cholesky panel), and the only drop-in-correct reconstruction (`geqrf(Q)`) re-imports the floor. Also the "dense" competition inputs are heavy-tailed ill-conditioned (cond up to ~8e6), so CholeskyQR needs fp32 Gram + a Demmel shift, not a cheap tf32x3 pass. A *custom-kernel* CholeskyQR (custom blocked tensor-core Cholesky + custom unpivoted-GETRFNP reconstruction) could break n=4096, but that is 2–3 hand-written tensor-core kernels for a **bounded** single-case payoff — strictly lower EV than the trailing-GEMM engine. **Reinforces the headline: no library shortcut exists; the gap is custom kernel engineering.**
- CholeskyQR2 + Ballard–Demmel Modified-LU reconstruction sums to **~4.2 ms** for n=512 b=640 (adversary re-derived from anchors) — that's the **mid-board 2000–4300µs cluster**, not the 1332µs podium. The reconstruction adds an n³ batched LU, and ill-conditioning forces a shift + extra pass.
- My B200 microbench confirms the trap: Gram AᵀA = 282µs (fast) but the triangular factorizations (Cholesky 4804µs, trsm/inverse 5890µs) are cuSOLVER/cuBLAS-bound; `A@Rinv` GEMM = 318µs (fast). CholeskyQR needs *custom* tensor-core Cholesky+trsm just to not lose.
- Filename corroboration: top-3 are generic `submission.py`; `submission_tsqr.py` is **mid-board**. The reconstruction family did not win.
- **But** the literature nuggets are real and worth keeping: Ballard/Demmel **Modified-LU reconstruction** (LU-without-pivoting of Q−S, S=−sgn(diagQ)) recovers exact flat (H,tau) from ANY orthonormal Q and is **provably backward-stable independent of condition number**; **shifted-CholeskyQR3** (shift s=11(mn+n(n+1))·u·‖A‖²) *does* rescue clustered (adversary verified orth=1.7e-13). So CholeskyQR is *robust*, just not *fast enough* once you pay reconstruction.

## Where the 10× actually is (the engineering gap)
1. **A warp-specialized persistent tensor-core GEMM engine for the tf32x3 trailing update.** We run the trailing at ~1.5% of TF32 peak via `torch.matmul`/basic Triton `tl.dot`; the leaders run a `tcgen05.mma` warp-specialized persistent kernel (TMA producer / MMA / epilogue warps, 128B swizzle, 2-SM M256 tiles) — gau.nernst-class **raw PTX**, or **TLX** (Triton Low-level eXtensions, the `triton_tlx` filename) as the Triton-native route. The >600µs CUDA-over-Triton gap on the board suggests a **pure-Triton stack may have a ceiling above 1332µs**. This is the BULK of the gap and the hardest to close.
2. **Precision recipe: likely 1×TF32 + a single fp32 residual fixup, not our 3× tf32x3** (the `fixup_tf32` / `fixup_tf32_1024` filenames). tf32x3 is 3× the trailing FLOP; 1×TF32 is at the EDGE of the factor gate (adversary's precision sweep: worst factor ratio 0.787, i.e. the fact-5 "21>20" worst-of-640 failure). A cheap fixup recovering ~3 bits would give the last ~2×. **This is the single most testable lever for us.**
3. **Register-file / multi-warp panel** (MAGMA-style: cache the m×nb panel in registers, one thread/row, unblocked geqr2 in-register). We have the warp idea (panel-warps = 5×) but not register-file caching.
4. **Recursive blocking** to convert tall-skinny trailing GEMMs into square GEMMs (TPDS-2024 reports up to 8.67× FP32 on tensor cores).

## Per-case budget (adversary-checked)
n=512 b=640 ≈ 1.7ms with tf32x3 (trailing GEMM dominates) → needs the 1×TF32+fixup to reach ~1.3ms. n=4096 b=2 ≈ 6–8ms: trailing GEMM floor ~1.5–1.8ms, the rest is the reduction-bound serial panel critical path (n serial reflector steps at b=2) — this is what they shave to single digits and we cannot at b=2 (the geqrf-floor case).

## Two numerical insights that explain why aggressive precision is safe
- **Orthogonality is precision-INDEPENDENT** (adversary verified): the checker rebuilds Q from the stored *unit* reflectors, so Q is orthonormal by construction regardless of trailing-GEMM precision. **Only the factor residual is at risk.**
- The factor residual is **‖A‖₁-relative**, dominated by the large columns, so errors in tiny ill-conditioned columns are absolutely negligible (the ~1000× margin).

## Actionable for us (ranked)
1. **1×TF32 trailing + a single fp32 residual fixup** — the testable ~2× lever (resolves the precision uncertainty; the likely podium differentiator).
2. **A better trailing-GEMM engine** (TLX warp-specialized, or accept a pure-Triton ceiling) — the bulk of the gap, biggest lift.
3. Register-file panel + recursive blocking — incremental.
Reconstruction/CholeskyQR — **not worth it** (mid-board); keep Modified-LU only as a known fallback.


## MEASURED precision margins (mixed@640 is the binding constraint — key correction)
> **⚠️ SUPERSEDED (2026-06-26, branch `fp8-fp4-attack`, see `archive/docs/FP8_SESSION_PROGRESS.md` + memory `qr-v2-solve-tf32-floor-artifact`):** the "tf32x3 = 2.0× = SAFE floor, NO safe win below it, and the lever is 1×TF32+fixup" framing in this section is WRONG. The 2.0× margin was a **tf32 triangular-SOLVE artifact** (cuBLAS trsm honored `allow_tf32=True`), NOT a trailing-GEMM wall — `solve→fp32` gives ~800× margin. The realized deployable trailing-precision win is **FP16x3** (=tf32x3-class ~22-bit accuracy, fp16 TC = 2× tf32, +3.6% geomean, SHIPPED, m1.83). **1×TF32 trailing was MEASURED reseed-DQ** (mixed@640 worst-of-640 sfr 19.7, margin 1.02 across 12 seeds — `experiments/microbench_1x_reseed.py`), so retire the "1×TF32+fixup" recommendation in §2/Actionable above. fp8/fp4 = accuracy-viable (m10.8 @ 6 Ozaki dots) but speed-dead (K-poor shape; dedicated tcgen05 cores verified used). The "warp-specialized GEMM engine is the gap" thesis still stands. Original (now-wrong) text below for history:

The "~1000x margin" applies only to WELL-CONDITIONED cases. The **n=512 mixed batch=640** worst-of-640 is tight:
- tf32x3: scaled_factor_residual **10.1 / gate 20 = 2.0x margin** (this is the SAFE floor).
- fused 2-term-rounded (round-to-nearest tf32 split, keep data operand low bits): **1.4x margin**, ~5% faster than tf32x3 (`experiments/cand_fused2br.py`, fused Triton, passes all 12 on benchmark seeds).
- 1xTF32: FAILS (3% over). 2-term-truncated: 1.0x (marginal fail).
Since the competition RESEEDS (fails submissions that break on seed changes), a <2x margin is disqualification-risky → tf32x3 (2.0x) is the safe choice; 2-term-rounded is a risky ~3% geomean. There is NO safe precision win below tf32x3; the mixed worst-case pins it. The leader's sub-tf32x3 (if any) must use a margin-PRESERVING fixup (iterative refinement) OR accepts reseed risk. Also: expanded GEMM autotune (14 cfg) was WORSE than 5 cfg -> basic-Triton GEMM has no autotune headroom; the gap is the warp-specialized engine.