# Optimization journal — qr_v2

Geomean (µs). "Modal" = my `gpu_bench` on Modal B200 (optimistic on tf32/fp16, accurate on geqrf/FP32/panel). "Official" = real gpumode leaderboard.

| step | file | geomean | what changed |
|---|---|---|---|
| baseline (eager) | m1_eager_123203us.py | **123203 official** | blocked compact-WY, all eager (≈250+ launches/call, dispatch-bound) |
| triton panel | m2_triton_43576us.py | **43576 official** | fused Triton sub-panel kernel + bmm trailing (still has the ~60-launch dlarft loop) |
| cand_D | experiments/cand_D_hybrid.py | 18448 Modal | **3×TF32** emulated-fp32 (hi/lo split + 3 matmuls) + shape-route geqrf for tiny/small-batch + **T via one triangular solve** (kills dlarft loop) |
| cand_H | experiments/cand_H_blktune.py | 16568 Modal | block tune: n=176→64 (5315→1083 µs); n=352/512/1024→32 (block 16 too small) |
| cand_I | experiments/cand_I_baddbmm.py | 14712 Modal | 3×TF32 accumulate via `baddbmm` (kills the 2 big add-tensor passes); helped n=176/352/1024, n=512 barely (→ n=512 is GEMM-shape-bound) |
| cand_J | experiments/cand_J_2level.py | 13940 Modal | **two-level blocking** NB=128/ib=32 (fat K=128 trailing GEMM); n=512 25.4k→19.7k |
| cand_M | experiments/cand_M_1x1024.py | 13222 Modal | **1×TF32 for n≥1024** (looser 20·n·eps tol; passes mixed/nearrank); n=1024 37k→29k |
| cand_O | experiments/cand_O_NB256.py | ~12000 Modal | two-level **NB=256** (fatter K=256); n=512 19.8k→18.7k. 22/22 pass. |
| tf32x3 | m3_tf32x3_14760us.py | **14760 official** | fused `tl.dot(input_precision="tf32x3")` batched-GEMM kernel (no hi/lo HBM split); Modal said ~10.7k but official 14760 ⇒ **Modal calibration discovered** |
| **panel-warps** | m4_panelwarps_8580us_CONFIRMED.py | **8580 official** | launch `_panel_kernel` with `num_warps=4/8/16` by tile size. The per-CTA tile work was starved at 4 warps. **n=1024 54ms→10.9ms = 5× officially** (Modal only showed 2×). 1.72× official jump. |
| **fused-panel** | submission.py (m5 + ib=64/NB=128 for n=512) | n=512 ~16.5k→~14.4k Modal | widen ib 32→64 (SRAM-max fused panel) + NB 256→128 for n=512; n=1024 unchanged. ~7% apples-to-apples (noise-swamped). 22/22. Full mega-kernel KILLED first by `microbench_trailing.py` (single-CTA trailing 1.4-3.8× slower than batched). |
| route n=2048 | (folded into m5) | **~8100 est** (Modal 8117) | route n=2048 (b=8) to the custom one-CTA panel+warps instead of geqrf: 76.8ms→35ms (2.2×). Official gain partial (n=2048 trailing is 1×TF32, Modal-optimistic). NOT yet officially submitted. |

## Key technical facts established (with evidence)
- **~1000× FP32 tolerance margin:** plain FP32 Householder gives scaled_factor_residual ≈ 0.016 vs gate 20.
- **Mantissa wall at batch=640:** 1×TF32 (10-bit mantissa) fails n=512 mixed@640 (worst-of-640 → scaled 21 > 20). Passes n≥1024 (looser tol) and n=512 dense/rankdef/clustered. A batch-16 probe MISSES this — always gate mixed at the real batch (640).
- **tf32x3 (3-pass tf32, FULL fp32 exponent range) passes all 22** incl clustered's 7-order intra-matrix range (fp32 exponent + fp32 accum + residual-relative-to-‖A‖).
- **The hi/lo split (for 3×TF32 in pytorch) was the cost, not the matmul count** — 1×TF32 n=512=10.7k vs split-3×TF32=19k; going 2-term→3-term was free. The fused tf32x3 Triton kernel removes the split HBM traffic.
- **Panel is reduction/SRAM-bandwidth bound** (pure `tl.sum` + elementwise, NO tl.dot). So the lever is PARALLELISM (warps), not precision. The panel-warps win confirms this.
- **geqrf floor:** n=2048 77ms, n=4096 52ms via cuSOLVER (FP32, parallelizes one matrix across SMs). Custom one-CTA-per-matrix beats it ONLY with enough CTAs (=batch): n=2048 b=8 wins 2.2×; n=4096 b=2 loses 3.9× (2 CTAs).
- **Modal calibration:** ~1.5-2× fast on tf32/fp16 tensor-core cases; exact on geqrf/FP32/panel/small. Panel/FP32 wins are UNDER-estimated by Modal (real ≥ Modal).
