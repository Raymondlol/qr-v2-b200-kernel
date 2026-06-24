# Dead ends — DO NOT re-explore these (each was tested or rigorously evaluated)

## Precision below tf32x3 — all blocked by the MANTISSA wall
The factor gate at batch=640 is tight enough that **1×TF32 (10-bit mantissa) already fails** n=512 mixed (worst-of-640 → scaled 21 > gate 20). So there is **no room below tf32x3** (~22 effective mantissa bits, which is the minimum that passes).
- **fp8 (e4m3, 3-bit mantissa): DEAD.** Single-pass blows the gate 20-100×. Correctness needs ~6-8 error-compensation passes → at 4× throughput that's a compute wash, then net slower after `.to(fp8)` split overhead. (workflow verdict, high confidence)
- **nvfp4/mxfp4 (1-bit mantissa): DEAD.** Even worse; multi-slice Ozaki form needs ~10 slices → washes out the 8× throughput. The leader's `nvfp4.py` filename almost certainly applies to a DIFFERENT algorithm, not this exact factor-residual trailing update.
- **fp16x3: TESTED, ~wash.** Passes all 22 (range-wall prediction was WRONG — fp16 subnormals + fp32 accum absorb it), but on Modal it's ~tied/slightly slower than tf32x3 because the in-kernel `.to(fp16)` conversions eat the 2× tensor-core gain. `experiments/cand_V_fp16x3.py`. Might win on competition HW (slower tf32) but unconfirmable on Modal.
- **2-term TF32 (split one operand): TESTED, useless.** No speedup (split traffic is the cost, not matmul count) AND worse accuracy (n=512 mixed scaled 44). `experiments/cand_R_2term512.py`.
- **lowprec-panel (run the panel reductions in tf32/fp16): DEAD.** The `_panel_kernel` has NO `tl.dot` — it's pure `tl.sum` reductions + elementwise FMA = CUDA-core/SRAM-bandwidth bound. B200 low-precision speedups apply ONLY to tensor cores → ~0 ALU gain, AND it corrupts the reflectors (tf32 τ → ‖QᵀQ−I‖ ~0.5 vs gate 0.012). (workflow verdict, high confidence)

## Large-n custom path
- **Custom two-level for n=4096 (small ib so SRAM fits): DEAD (cand_U).** n=2048 687ms, n=4096 1228ms — 9-22× WORSE than geqrf. batch=2-8 → only 2-8 one-program-per-matrix CTAs on 148 SMs = catastrophic occupancy. `experiments/cand_U_customLargeN.py`. **NOTE: this was tested with default-4-warp panel; re-tested WITH panel-warps (cand_X) it WINS n=2048 but still loses n=4096.**
- **geqrf-panel (wide block) + TF32 trailing: DEAD (cand_G/N).** ≈ geqrf — a wide cuSOLVER panel on [2048,512] costs ≈ the whole thing; narrow blocks add per-call overhead. `experiments/cand_N_largeN512.py`.
- **Stream-parallel geqrf (one CUDA stream per matrix): ILLEGAL.** Worked great (n=2048 77k→17k) but submission checker rejects any "stream" usage. `experiments/_INVALID_cand_L_streamgeqrf.py`. Same for CUDA graphs.
- **TSQR → flat output: legal but washes.** TSQR's tree reflectors don't fit the flat layout directly, BUT Householder-reconstruction (form Q explicitly O(n³) + LU-without-pivoting of top n×n block → unit-lower Y=v_i, τ from diagonal) recovers valid flat (H,τ). It ~3×'s the per-matrix flop and must still beat cuSOLVER on n=2048/4096 → wash. (workflow verdict, high confidence)

## Other
- **load_inline raw CUDA: marginal.** Only n=176 fits fully in 228KB SRAM (single-CTA in-SRAM QR) → ~2× on that one case = ~1-2% geomean. The dominant cases are tensor-core-GEMM-bound where raw CUDA doesn't beat Triton. PTX micro-scheduling: not worth it (Triton already lowers/pipelines; our bottlenecks aren't instruction-scheduling-bound). The ONE place raw CUDA helps: providing the grid-wide cooperative-sync primitive for a multi-CTA panel (Triton lacks it) — but mind the "stream" substring (cooperative-launch APIs mention streams).
- **block size 16 everywhere:** too small, launch-bound (cand_H2, 21682). NB=512: worse than 256 (cand_P).
- **NB=512 for n=1024:** worse than NB=256 (within-panel updates too wide).
