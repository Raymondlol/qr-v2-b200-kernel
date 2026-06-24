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

## Full mega-kernel (one CTA does panel + trailing) — KILLED by microbench
`experiments/microbench_trailing.py` (B200): a single-CTA-per-matrix trailing GEMM is **1.4–3.8× SLOWER** than the batched tf32x3 GEMM (n=512 fat 1.71×, n=512 NB=32 1.38×, n=1024 fat 3.82×, n=1024 NB=32 2.69×; relerr=0). One CTA looping output tiles per matrix can't match the batched GEMM's multi-CTA tiling. **Keep the trailing batched.** (The `--stress`-gated full-mega plan in `~/.claude/plans/mega-kernel-mighty-pony.md` Phase 3 is therefore NO-GO.)
- **What DID work from that effort:** the **fused-panel tuning** — widen `ib` 32→64 for n=512 (SRAM-max fused panel) + `NB` 256→128. n=512 ~16.5k→~14.4k Modal (~7% apples-to-apples, noise-swamped). n=1024 unchanged (ib capped at 32; NB=256 best — NB=128 is much WORSE for n=1024, opposite of n=512). In `submission.py`. `experiments/cand_fp*.py`.


## Naive CholeskyQR2 (torch primitives) — SLOWER than our Householder
`experiments/microbench_choleskyqr2.py` (B200): CholeskyQR2 via torch = 22ms (n=512 b=640), WORSE than our ~14ms Householder. Breakdown: Gram AᵀA is FAST (718µs, tensor-core) but torch's batched `cholesky` (4742µs) and `solve_triangular` (5822µs) are the killers — cuSOLVER/cuBLAS, NOT tensor-core, batched poorly like geqrf. **Every torch primitive tops out at 14-1070ms; leader is 1.5ms → the 10x is pure CUSTOM-kernel engineering, no library shortcut.** The promising (research-grade) version: CholeskyQR with a CUSTOM blocked Cholesky (trailing = GEMM/tensor-core) + trsm via triangular-inverse+GEMM + Householder reconstruction for flat (H,tau). The Gram being fast (718µs) is the evidence this path can work; it would also break the geqrf floor (all GEMM-bound, fills GPU at any batch). Not yet attempted.

## Other
- **load_inline raw CUDA: marginal.** Only n=176 fits fully in 228KB SRAM (single-CTA in-SRAM QR) → ~2× on that one case = ~1-2% geomean. The dominant cases are tensor-core-GEMM-bound where raw CUDA doesn't beat Triton. PTX micro-scheduling: not worth it (Triton already lowers/pipelines; our bottlenecks aren't instruction-scheduling-bound). The ONE place raw CUDA helps: providing the grid-wide cooperative-sync primitive for a multi-CTA panel (Triton lacks it) — but mind the "stream" substring (cooperative-launch APIs mention streams).
- **block size 16 everywhere:** too small, launch-bound (cand_H2, 21682). NB=512: worse than 256 (cand_P).
- **NB=512 for n=1024:** worse than NB=256 (within-panel updates too wide).
