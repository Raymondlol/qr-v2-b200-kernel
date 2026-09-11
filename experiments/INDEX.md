# Experiments index

Each `cand_*.py` is a single-variable experiment. Numbers are Modal geomean unless marked official.
Lineage flows roughly A→Y. See `../docs/JOURNAL.md` for the curated milestones and
`../docs/DEAD_ENDS.md` for what failed and why.

> **Coverage.** This file has two halves. **Part 1** (below) is the hand-written narrative for the
> phases that produced the shipped submission — it is the part worth reading. **Part 2** is a
> complete, generated listing of *every* file in this directory. Part 1 used to be the whole index
> and covered 34 of 142 files, which made the rest of the directory an undocumented dump. Regenerate
> Part 2 with `python3 tools/gen_experiments_index.py`; the coverage assertion at the bottom of this
> file is checked by that script.

| file | result | note |
|---|---|---|
| cand_geqrf.py | 131246 | pure `torch.geqrf` (the reference). Profiles cuSOLVER: great small-batch/large-n, terrible large-batch |
| cand_A_tf32.py | fails | triton_1 + global TF32 flags; 1×TF32 fails small-n & mixed |
| cand_C_solveT_tf32.py | fails | + T-via-solve; still 1×TF32 → mixed fails |
| cand_D_hybrid.py | 18448 | **3×TF32 + geqrf routing + solve-T.** First all-pass hybrid |
| cand_E_block64.py | 24273 | block=64 for n≤512: helped n=176, hurt n=512 (panel bottleneck) |
| cand_G_largeN.py | regress | geqrf-panel + 3×TF32 trailing for large-n — loses to geqrf |
| cand_H_blktune.py | 16568 | n=176→64, rest→32 (the good block schedule) |
| cand_H2_blk16.py | 21682 | block 16 everywhere — too small |
| cand_I_baddbmm.py | 14712 | baddbmm-accumulated 3×TF32 |
| cand_J_2level.py | 13940 | **two-level blocking** NB=128/ib=32 |
| cand_M_1x1024.py | 13222 | 1×TF32 for n≥1024 |
| cand_N_largeN512.py | ~same | geqrf-panel block-512 + 1×TF32 large-n — ≈ geqrf (dead) |
| cand_O_NB256.py | ~12000 | two-level NB=256 |
| cand_P_NB512.py | 14088 | NB=512 — worse |
| cand_Q_1x512.py | fails | 1×TF32 at n=512 — fails mixed@640 (the decisive precision test) |
| cand_R_2term512.py | fails | 2-term TF32 — no speedup + worse accuracy |
| cand_T_tf32x3.py | **14760 official** | fused tf32x3 Triton GEMM (= milestone m3) |
| cand_U_customLargeN.py | 687/1228ms | custom two-level large-n, small ib — 9-22× worse than geqrf (occupancy) |
| cand_V_fp16x3.py | ~wash | fp16x3 trailing — passes all 22 but conversion overhead ≈ tf32x3 on Modal |
| cand_W_panelwarps.py | **8580 official** | **panel num_warps 4/8/16** (= milestone m4). THE big win |
| cand_W2_morewarps.py | 10727 | warps 8/16/32 — 32 slightly worse than 16 |
| cand_X_largeN_warps.py | n2048 35.6ms / n4096 201ms | large-n one-CTA + panel-warps: WINS n=2048, loses n=4096 |
| cand_Y_route2048.py | ~8100 est | route n=2048→custom, n=4096→geqrf (= milestone m5 = current submission.py) |

`scratch/` = `_force_*.py` helpers that force a specific code path for CPU correctness testing (the test shapes route to geqrf, so the custom path must be forced to validate it).
`_INVALID_cand_L_streamgeqrf.py` = the illegal stream-parallel geqrf (kept as a record of why streams are banned).

## Phase 2: mega-kernel investigation + leader research (2026-06-25)
| file | result | note |
|---|---|---|
| cand_fp1a_ib64 / fp1b_NB128 / fp1c_NB64 | n=512 ~16.5k→14.4k | **fused-panel**: widen ib 32→64 (=SRAM-max fused panel) + NB tuning. NB=128/ib=64 best for n=512. In submission.py |
| cand_fp2a/2b_n1024 | NB=128 much WORSE | n=1024 wants NB=256 (opposite of n=512); ib capped at 32 by SRAM |
| microbench_trailing | 1-CTA 1.4-3.8× slower | KILLED full mega-kernel (single-CTA trailing loses to batched) |
| microbench_choleskyqr / qr2 / cqr_components | CQR2 22ms > our 14ms | naive CholeskyQR dead: Gram fast (282µs) but torch chol(4804)/trsm(5890) cuSOLVER-slow; A@Rinv GEMM fast (318µs) |
| cand_prec_1x/2a/2b/2br, cand_fused2br | tf32x3 is the floor | precision sweep: 1×TF32 fails mixed@640 (3% over), 2-term-rounded passes at 1.4× margin (risky), tf32x3 = 2.0× (safe) |
| cand_gemmtune | WORSE (9752 vs 9184) | expanded GEMM autotune (14 cfg) — no basic-Triton headroom |

Modal runners: `modal_app.py` (submission → gpu_bench) · `modal_microbench.py` (arbitrary script on B200).

## Phase 3: engine attack + methodology + Gluon (2026-06-25). Modal≈official now (V4 7793 vs 7788).
Audit re-derived the leader direction, then a profiler-driven engine attack: **7793→5678 Modal (1.38×)**, all in `submission.py`. Key turn: `profile_detailed.py`/`profile_phases.py` showed the trailing GEMM is only 21-26% (not THE gap); panel 40-46% (latency-bound); ~20% was elementwise GLUE.
| file | result | note |
|---|---|---|
| cand_gram | 7793→6907 | T-factor Gram VtV → 1×TF32 for n≥1024 (well-cond unit reflectors). PROMOTED |
| cand_adaptive_ib | 6907→6771 | grow ib as remaining m shrinks (`_max_ib`), under SRAM cap → fewer+fatter narrow updates (n=2048 1.14×). PROMOTED |
| cand_tsolve | 7230 (WORSE) | Tsolve→inverse+GEMM: slower + margin 2.0→1.4× (1×TF32 GEMM vs fp32 solve). REJECTED → DEAD_ENDS |
| cand_glue | 6771→5756 (1.18×) | drop redundant `.clone()`, fuse C−=V@Y (`_bmm3_sub`+baddbmm), diagonal-view assign. Caught autotune-in-place bug (needs `restore_value`). PROMOTED |
| cand_pw8 | 5756→5678 | panel nw=8 (was 16/32 for large BN; microbench_panel showed 8 best, panel is latency-bound). PROMOTED |
| cand_panelstream | 6083 (WORSE) | bit-identical panel per-column streamline REGRESSES (loop-carried dep hurts compiler). REJECTED → DEAD_ENDS |
| cand_nb128 | 6052 (WORSE) | NB 256→128: blocking sweep "won" but full-run regressed = noise. REJECTED (blocking is optimal) |
| microbench_floor_and_engine / engine_bakeoff / panel | diagnostics | trailing K-poor (cuBLAS 21-34%); panel latency-bound; blocked-chol beats cuSOLVER 2.86% @n=4096 |
| microbench_cqr_floorbreak (text only) | — | CholeskyQR floor-breaker cost model (parked, see DEAD_ENDS) |
| cuda_gate1/2/3 (modal_cuda.py) | built, DEAD | raw-PTX mma.sync tf32x3 GEMM: ~1.2× over Triton but UNDEPLOYABLE (eval has no nvcc) |
| gluon_gate1/2, gluon_*recon/dump (branch gluon-tcgen05) | G1 ✓, G2 slow | Gluon tcgen05 = deployable warp-spec, but tl_dot is 61-76 TF/s; competitive needs multi-day warp-spec pipeline, bounded ~1.1×. PARKED |

**Tooling upgrade:** `harness/lab.py` + `modal_lab.py` replace the one-run-per-candidate loop — same-container variance-aware A/B (stderr ~1-2µs vs old ±15% noise) + fast correctness gate + profile + JSON log. **Use it (docs/METHODOLOGY.md).** Helper copies `sub_now.py`/`sub_profile.py` were transient (deleted).


---

# Part 2 — complete file listing (generated)

Every file in `experiments/`, with a one-line description taken from its own header. Generated, so
it cannot drift out of coverage; the descriptions are only as good as each file's header comment.
Entries already covered in Part 1 appear here too.

### cute-DSL engine build (M1→M6) — research, never submitted  (25 files)

| file | what it is |
|---|---|
| `cute_apply_chain.py` | M3c probe #3b — FULL 3-GEMM WY apply chain: dV = V @ (T^T @ (V^T @ C)), tcgen05 tf32x3. The decisive apply-chain validation before integration. One su… |
| `cute_apply_gemm1.py` | M3c probe #2 — apply GEMM-1: W1 = V^T @ C via tcgen05 tf32x3, REAL apply orientation + in-reg split. Validates the hardest correctness pieces of the M… |
| `cute_apply_gemm1_async.py` | M6 slice — G1 (W1=V^T C) K-loop via an ASYNC WARP-SPEC RING (PipelineAsyncUmma), NO per-tile CTA barrier. The decisive M6 proof-of-concept: replace th… |
| `cute_apply_gemm1_kloop.py` | M3c probe #3a — apply GEMM-1 W1=V^T C with a K-LOOP (M>32) + per-tile drain. Extends the validated single-tile GEMM-1 (cute_apply_gemm1.py, rel 8.25e-… |
| `cute_apply_probe.py` | M3c probe #1 — HAND-FILLED swizzled SMEM → tcgen05 1x TF32 GEMM (NO TMA). THE decisive de-risk for M3c-0: the build guide wants the per-matrix apply t… |
| `cute_gemm_tf32_m1.py` | M1' step 1 — single TF32 tcgen05 GEMM in cute-dsl (mutation of the shipped fp16_gemm_0.py). Goal: get tcgen05 *TF32* working end-to-end on our B200 ev… |
| `cute_gemm_tf32x3.py` | M3a — tcgen05 TF32x3 GEMM in cute-DSL (3-limb emulated-fp32, the precision the n<=512 apply needs). Extends cute_gemm_tf32_m1.py (1x tf32, rel 8.34e-4… |
| `cute_pipesrc.py` | Print the cutlass.pipeline PipelineAsyncUmma / PipelineAsync producer/consumer source on the image, to learn the EXACT arrive mechanism (why manual-st… |
| `cute_pipesrc2.py` | Get the EXACT UMMA-completion->mbarrier-arrive primitive: PipelineUmmaAsync.producer_commit source + tcgen05 commit functions + cute.arch mma/tcgen05 … |
| `cute_qr_m1.py` | M1 — thinnest CORRECT cute-DSL batched QR (one warp per matrix, unblocked Householder). Purpose: get a CORRECT compact-Householder QR running end-to-e… |
| `cute_qr_m1b.py` | M1.5 — BLOCKED cute-DSL QR (single warp), the structural precursor to the M2 overlap triangle. Reorganizes M1's unblocked QR into super-panels of widt… |
| `cute_qr_m1c.py` | M3c-pre — blocked QR with LARFT T-factor + WY 2-GEMM apply, in WARP-REDUCE (no tcgen05 yet). Decouples the MATH (LARFT + compact-WY apply) from the ME… |
| `cute_qr_m2a.py` | M2a-0 — warp-SPLIT cute-DSL QR (panel warp ‖ apply warp), serial via CTA barrier. First step of the M2 overlap triangle: split each super-panel's two … |
| `cute_qr_m2b.py` | M2b — look-ahead OVERLAP cute-DSL QR (panel[k+1] ‖ far_apply[k]), CTA-barrier delimited. The first overlap measurement. Splits each super-panel's appl… |
| `cute_qr_m3c.py` | M3c — per-matrix warp-spec QR engine in cute-DSL: panel(warp0) ‖ blocked-WY apply(warp1). STAGE A (this file, BASELINE): the full per-matrix 2-warp st… |
| `cute_qr_m3c_occ.py` | M3c-0 — per-matrix QR with tcgen05 tf32x3 blocked-WY apply (the integrated engine). warp0 factors the super-panel + gram + LARFT (FP32, validated m1c)… |
| `cute_qr_m3c_prof.py` | M3c-0 — per-matrix QR with tcgen05 tf32x3 blocked-WY apply (the integrated engine). warp0 factors the super-panel + gram + LARFT (FP32, validated m1c)… |
| `cute_qr_m3c_smem.py` | M3c-0 — per-matrix QR with tcgen05 tf32x3 blocked-WY apply (the integrated engine). warp0 factors the super-panel + gram + LARFT (FP32, validated m1c)… |
| `cute_qr_m3c_tc.py` | M3c-0 — per-matrix QR with tcgen05 tf32x3 blocked-WY apply (the integrated engine). warp0 factors the super-panel + gram + LARFT (FP32, validated m1c)… |
| `cute_qr_m6.py` | M6 per-matrix engine — panel+gram+LARFT (warp0) + ASYNC-RING tcgen05 apply (warps0-2 ‖ warp3). The full integration: cute_ring_chain's async warp-spec… |
| `cute_ring_chain.py` | M6 full-apply ring — dV = V(T^T(V^T C)) with G1 as the async warp-spec RING, G2/G3 simple. Extends the validated G1 ring (cute_ring_g1.py, 1.58x) to t… |
| `cute_ring_g1.py` | M6 ring (HAND-ROLLED mbarriers) — G1 W1=V^T C K-loop, producer warps ‖ MMA consumer, NO per-tile CTA barrier. The deadlock fix: cute-DSL's Umma pipeli… |
| `cute_sass.py` | Dump PTX/SASS of a cute-DSL kernel (static low-level analysis; ncu/nsys are gVisor-dead on Modal). Compiles a small representative kernel (cute_apply_… |
| `cute_split_api.py` | Find the cute-DSL 4.5.2 API for in-register bitcast + bitwise-and (for the tf32x3 limb split). `cute.bitcast` does NOT exist. Hunt the real names. Pla… |
| `cute_split_test.py` | Isolate the in-register tf32 hi/lo split: bitcast+AND variants, compare hi to torch truncation. |

### Gluon warp-specialization / tcgen05 investigation  (19 files)

| file | what it is |
|---|---|
| `gluon_api_recon.py` | Pin down the EXACT Gluon tcgen05 API in our installed triton 3.6.0 (main-branch tutorials may differ). Print signatures/docs of the key ops, and locat… |
| `gluon_dot_body_dump.py` | Dump the exact body of tl_dot_blackwell + its smem-operand helpers (L100-160) — the canonical low-level tcgen05_mma setup we must replicate for an ASY… |
| `gluon_dump2.py` | — |
| `gluon_gate1.py` | Gluon G1: minimal tcgen05 GEMM via the in-tree tl_dot helper (deployable, no nvcc). Validates the 3.6.0 Gluon tcgen05 path compiles + runs + is correc… |
| `gluon_gate2.py` | Gluon G2 (decisive speed): batched tiled tcgen05 GEMM (1-pass tf32) on the real trailing shapes, vs our tl.dot tf32x3 (38/41 TF/s) and cuBLAS-1xTF32 (… |
| `gluon_lowlevel_dump.py` | Dump the EXACT low-level async tcgen05 calling sequence from installed triton 3.6.0: the full source of tl_dot (the only known-correct usage of tcgen0… |
| `gluon_overlap_probe.py` | PHASE 0 — co-host validation: can ONE grep-clean Gluon kernel run a CUDA-core gl.sum reduction (panel-like) CONCURRENTLY with an ASYNC tcgen05_mma seq… |
| `gluon_panel.py` | PHASE 1.5 step 1 — port submission.py::_panel_kernel (the latency-bound Householder reflector reduction, pure gl.sum + FMA, no dot) to Gluon @gluon.ji… |
| `gluon_panel_async.py` | PHASE 1.5 — design-A RESURRECTION test: the REAL register-heavy Gluon panel (default 8w) co-resident with a LOW-LEVEL ASYNC tcgen05 trailing worker (8… |
| `gluon_panel_salvage.py` | PHASE 1.5 salvage DIAGNOSTIC — before rewriting the panel to be shared-memory-resident, decompose the design-A regression into its two possible causes… |
| `gluon_panel_smem.py` | Stage A.2 — SMEM-RESIDENT STREAMING panel: the tile lives in shared memory; each j-reflector streams the tile in RBLK-row blocks (only one block in re… |
| `gluon_panel_trailing.py` | PHASE 1.5 step 2 — THE de-risk: the REAL Gluon panel reduction (default partition, register-heavy [512,64] tile) co-resident in ONE CTA with a real tc… |
| `gluon_pattern_dump.py` | Dump the version-matched (3.6.0) Gluon tcgen05 idioms from the installed package's own translator_helpers.py, and check the convenience APIs (get_defa… |
| `gluon_smem_api.py` | Stage A.1 recon: exact Gluon shared_memory_descriptor API for streaming a [BN,BCOLS] panel tile in ROW-BLOCKS (load a block to regs, update, store bac… |
| `gluon_trailing_penalty.py` | PHASE 1.1 — pin the design-A trailing penalty with the REAL primitive (tcgen05, not tl.dot). At the real n=512 trailing shape (b=640, M=512, K=128(=NB… |
| `gluon_warp_recon.py` | Recon the warp-level primitives for HAND-ROLLED warp specialization in stock Gluon 3.6.0: gl.warp_id / thread_id, ways to confine a layout to a warp s… |
| `gluon_ws_dump.py` | Dump gl.warp_specialize: signature, source, and the code-generator lowering so we learn the exact calling convention (partition functions, num_warps p… |
| `gluon_ws_overlap.py` | PHASE 0 (real): warp-specialized overlap via gl.warp_specialize. Worker partition issues the async tcgen05_mma sequence (trailing-like, tensor cores);… |
| `recon_tlx_gluon.py` | Deployability recon: what LOW-LEVEL control is available in the STOCK triton 3.6.0 we'd actually deploy with (= the modal_app env, ~= the competition … |

### Gluon stage-0/1 overlap foundations  (7 files)

| file | what it is |
|---|---|
| `stage0_regfile_panel.py` | STAGE 0 (the crux): MAGMA-style register-FILE Householder panel. Goal: a panel layout that is BOTH fast (~= the register-resident gluon_panel.py) AND … |
| `stage0b_coresident.py` | STAGE 0.5 — make-or-break co-residence test for the crux. Swap the register-heavy default panel (255 regs / 310 spills, which made design-A REGRESS pe… |
| `stage1a_tf32x3_async.py` | STAGE 1 increment A — a REAL tf32x3 trailing GEMM via async tcgen05. The existing _async_trail_part is only a PROXY (NPASS copies of the SAME A@B summ… |
| `stage1b_compactwy.py` | STAGE 1 increment B — compact-WY trailing APPLY via async tcgen05 (the trailing partition's real job), built from a K-looped tf32xN async GEMM. C := C… |
| `stage1c2_2gemm_apply.py` | STAGE 1C (corrected) — the trailing apply is 2 GEMMs, NOT 3. 1B's ~1.4x was the explicit T-apply (3rd GEMM). Folding T into V (VT = V@T^T, precomputed… |
| `stage1c_fused_apply.py` | STAGE 1C step 1 — RESIDENT-W1 fused compact-WY apply (kill the W1 global round-trip). The 1B apply was ~1.4x slow partly because W1 = V^T@C round-trip… |
| `stage1d_overlap.py` | STAGE 1D — the BUILDABLE design-A core: warp_specialize( rowmagma 1 sub-panel || far-trailing 2-GEMM slice ). Avoids the hard in-kernel super-panel fa… |

### persistent fused one-shot kernel (became V10)  (1 files)

| file | what it is |
|---|---|
| `fused_qr_slice.py` | M1 vertical slice: ONE CTA factors ONE [N,N] matrix end-to-end, in-kernel, serial. This is the FUSED one-shot kernel (one launch, all sub-panels in a … |

### profiling deep-dive tools (roofline, launch-gap)  (1 files)

| file | what it is |
|---|---|
| `deepdive_roofline.py` | DEEP-DIVE 1 — roofline-by-shapes (FREE, pure arithmetic, no GPU). Replicates submission.py::_factor_custom's loop to enumerate EVERY GEMM (M,N,K) in t… |

### microbenchmarks (isolated component timing)  (22 files)

| file | what it is |
|---|---|
| `microbench_1x_reseed.py` | Characterize the reseed-DQ risk of cand_solvefix_1x (1xTF32 trailing + fp32 solve) at the binding mixed@640 case across many seeds. Gate = worst-of-64… |
| `microbench_attrib.py` | ATTRIBUTION: the real submission gives mixed@640 sfr~10.7 (margin 1.9x) with allow_tf32=True, but ~0.025 (margin 800x) with allow_tf32=False. Same Tri… |
| `microbench_blocking_sweep.py` | Two-level tuning: re-sweep blocking (NB, ib_max) now that adaptive-ib + glue-fusion changed the narrow-update/trailing tradeoff. Old finding "NB=512 w… |
| `microbench_choleskyqr.py` | Test the CholeskyQR hypothesis: is a GEMM-heavy CholeskyQR2 ~10x faster than our Householder path on n=512 b=640, and which stress cases survive numer… |
| `microbench_choleskyqr2.py` | Decisive speed test for the CholeskyQR hypothesis. Add a shift so Cholesky never fails, and break the time into Gram / Cholesky / triangular-solve so … |
| `microbench_cqr_components.py` | Probe whether a CUSTOM CholeskyQR could be fast: isolate the two slow torch primitives (cholesky 4742us, trsm 5822us for n=512 b=640) and test tensor-… |
| `microbench_cqr_floorbreak.py` | Decisive B200 test of the CholeskyQR floor-breaker for the DENSE large-n cases (n=2048 b=8, n=4096 b=2) where torch.geqrf collapses (only 2-8 panels o… |
| `microbench_engine_bakeoff.py` | Engine attack, step 1: find the BEST achievable tf32x3 trailing GEMM with available tools (before any raw-PTX/TLX work), and quantify the recursive-bl… |
| `microbench_fattrail_prec.py` | DECISIVE (real pipeline): can the FAT trailing update (C -= V@Y, the biggest GEMM) drop BELOW tf32x3 without breaking the mixed@640 gate? The reconcil… |
| `microbench_feed_gemm.py` | How K-rich must the trailing GEMM be to FEED the tensor cores? Fixed batch=640, M=512, N=384; sweep K (= the block width NB = trailing contraction dim… |
| `microbench_floor_and_engine.py` | Two decisive B200 probes in one run (saves a container spin-up): A) FLOOR PROBE -- can a custom BLOCKED Cholesky (right-looking, trailing update = ten… |
| `microbench_fp16x3.py` | fp16x3 vs tf32x3 trailing: same ~22-bit accuracy (SAFE), but fp16 tensor cores are 2x tf32 on B200, so a tight fused fp16x3 (3 fp16 dots, split in-reg… |
| `microbench_fp8_coreverify.py` | VERIFY we actually hit B200's dedicated fp8 tensor cores in the low-precision tests (the user's challenge). Two checks: (A) PTX inspection: does our p… |
| `microbench_fp8_decisive.py` | DECISIVE fp8/fp4 feasibility microbench for qr_v2 (B200). Settles the question the prior "fp8 DEAD" verdict left open: with the REAL fp32-panel pipeli… |
| `microbench_fp8_fused.py` | Properly FUSED fp8-Ozaki-6dot trailing kernel: split V,Y into 3 e4m3 slices IN-REGISTER (per-tensor scales precomputed), do all 6 dots (i+j<=2) accumu… |
| `microbench_fp8_speed.py` | fp8 trailing SPEED on the real shapes: is fp8-Ozaki-6dot actually faster than the current fused tf32x3 trailing (and the simpler alternatives)? Uses P… |
| `microbench_mxfp8_vs_fp8.py` | #3: does mxfp8 (block-scaled tl.dot_scaled, tcgen05) get BETTER hardware utilization than plain fp8 (tl.dot) on the real K-poor trailing shapes? HW re… |
| `microbench_panel.py` | Panel kernel diagnosis: the panel (_panel_kernel) is now ~50% of n=512/1024 and the dominant cost. Is it COMPUTE-bound (O(BN*BCOLS^2) rank-1 updates -… |
| `microbench_reconcile.py` | RECONCILIATION: does the REAL submission.py (Triton panel + fused tf32x3 kernel) give scaled_factor_residual ~0.05 (my eager harness) or ~10 (the docu… |
| `microbench_trailing.py` | Phase-2 gate microbench: single-CTA-per-matrix trailing GEMM vs batched tf32x3. The full mega-kernel does each matrix's trailing update with ONE CTA. … |
| `microbench_validate.py` | Validate a candidate: (1) correctness on all 12 benchmark shapes (good + gate), (2) mixed@640 / rankdef / clustered / n1024 worst-of-batch margin acro… |
| `microbench_verify_solvefix.py` | VERIFY the solve-fix win across the precision-stress benchmark cases (real pipeline). The attribution proved the n=512 mixed@640 1.9x 'floor' was a tf… |

### per-phase profilers  (2 files)

| file | what it is |
|---|---|
| `profile_detailed.py` | Detailed op-level bottleneck map of the CURRENT submission (gram + adaptive-ib), via torch.profiler CUDA activity -> ground-truth per-kernel time (no … |
| `profile_phases.py` | Engine attack, step 2: where does the n=512 / n=1024 wall-clock ACTUALLY go? Before investing in a warp-specialized trailing-GEMM engine we must confi… |

### eval-environment probes  (2 files)

| file | what it is |
|---|---|
| `probe_eval_env.py` | ZERO-RISK eval-environment probe submission. Prints what's actually available in the qr_v2 eval (nvcc? cutlass? cuda-python?) to stdout+stderr (-> the… |
| `probe_eval_env2.py` | DECISIVE K0 probe v2 — make PASS/FAIL ENCODE whether cute-dsl is deployable on the qr_v2 board. probe v1 printed the env but the board shows Debug Inf… |

### single-variable submission candidates  (58 files)

| file | what it is |
|---|---|
| `_INVALID_cand_L_streamgeqrf.py` | Two-level blocked Householder QR for the large-batch custom path. * super-panel width NB (e.g. 128) factored from ib-wide (e.g. 32) Triton sub-panels … |
| `cand_A_tf32.py` | Batched square Householder QR (compact-WY) with a FUSED Triton panel kernel. *** STATUS: the Triton kernel is UNVERIFIED ON HARDWARE. *** It was writt… |
| `cand_C_solveT_tf32.py` | Batched square Householder QR (compact-WY). * Fused Triton panel kernel (1 launch / panel) -> low dispatch overhead * T-factor via a single batched tr… |
| `cand_D_hybrid.py` | Batched square Householder QR (compact-WY), HYBRID by shape: * geqrf (cuSOLVER) for small-batch / tiny-n (it dominates those shapes) * custom batched … |
| `cand_E_block64.py` | Batched square Householder QR (compact-WY), HYBRID by shape: * geqrf (cuSOLVER) for small-batch / tiny-n (it dominates those shapes) * custom batched … |
| `cand_G_largeN.py` | Adds a LARGE-N path to the hybrid: blocked Householder where each narrow panel is factored by cuSOLVER (torch.geqrf on the tall-skinny panel slice) an… |
| `cand_H2_blk16.py` | Batched square Householder QR (compact-WY), HYBRID by shape: * geqrf (cuSOLVER) for small-batch / tiny-n (it dominates those shapes) * custom batched … |
| `cand_H_blktune.py` | Batched square Householder QR (compact-WY), HYBRID by shape: * geqrf (cuSOLVER) for small-batch / tiny-n (it dominates those shapes) * custom batched … |
| `cand_I_baddbmm.py` | Batched square Householder QR (compact-WY), HYBRID by shape: * geqrf (cuSOLVER) for small-batch / tiny-n (it dominates those shapes) * custom batched … |
| `cand_J_2level.py` | Two-level blocked Householder QR for the large-batch custom path. * super-panel width NB (e.g. 128) factored from ib-wide (e.g. 32) Triton sub-panels … |
| `cand_M_1x1024.py` | Two-level blocked Householder QR for the large-batch custom path. * super-panel width NB (e.g. 128) factored from ib-wide (e.g. 32) Triton sub-panels … |
| `cand_N_largeN512.py` | Two-level blocked Householder QR for the large-batch custom path. * super-panel width NB (e.g. 128) factored from ib-wide (e.g. 32) Triton sub-panels … |
| `cand_O_NB256.py` | Two-level blocked Householder QR for the large-batch custom path. * super-panel width NB (e.g. 128) factored from ib-wide (e.g. 32) Triton sub-panels … |
| `cand_P_NB512.py` | Two-level blocked Householder QR for the large-batch custom path. * super-panel width NB (e.g. 128) factored from ib-wide (e.g. 32) Triton sub-panels … |
| `cand_Q_1x512.py` | Two-level blocked Householder QR for the large-batch custom path. * super-panel width NB (e.g. 128) factored from ib-wide (e.g. 32) Triton sub-panels … |
| `cand_R_2term512.py` | Two-level blocked Householder QR for the large-batch custom path. * super-panel width NB (e.g. 128) factored from ib-wide (e.g. 32) Triton sub-panels … |
| `cand_T_tf32x3.py` | qr_v2 submission: batched compact-Householder QR for B200. Validated: 22/22 official test cases pass; benchmark geomean ~10800us. Design (shape-routed… |
| `cand_U_customLargeN.py` | Two-level blocked Householder QR for the large-batch custom path. * super-panel width NB (e.g. 128) factored from ib-wide (e.g. 32) Triton sub-panels … |
| `cand_V_fp16x3.py` | EXPERIMENT: fp16x3 trailing GEMM (test the fp16 range wall on clustered/rankdef). Validated: 22/22 official test cases pass; benchmark geomean ~10800u… |
| `cand_W2_morewarps.py` | EXPERIMENT: panel kernel with more warps (parallelize per-CTA tile work). Validated: 22/22 official test cases pass; benchmark geomean ~10800us. Desig… |
| `cand_W_panelwarps.py` | EXPERIMENT: panel kernel with more warps (parallelize per-CTA tile work). Validated: 22/22 official test cases pass; benchmark geomean ~10800us. Desig… |
| `cand_X_largeN_warps.py` | EXPERIMENT: large-n one-CTA panel + panel-warps (does it beat geqrf?) Validated: 22/22 official test cases pass; benchmark geomean ~10800us. Design (s… |
| `cand_Y_route2048.py` | qr_v2 submission: batched compact-Householder QR for B200. (n=2048 routed to custom one-CTA panel + warps; n=4096 to cuSOLVER.) Validated 22/22. Valid… |
| `cand_adaptive_ib.py` | Phase 1a proxy: ib=64 for n=512 (fused-panel route, fewer narrow updates). qr_v2 submission: batched compact-Householder QR for B200. (n=2048 routed t… |
| `cand_cublas_trail.py` | Phase 1a proxy: ib=64 for n=512 (fused-panel route, fewer narrow updates). qr_v2 submission: batched compact-Householder QR for B200. (n=2048 routed t… |
| `cand_fp16x3.py` | Phase 1a proxy: ib=64 for n=512 (fused-panel route, fewer narrow updates). qr_v2 submission: batched compact-Householder QR for B200. (n=2048 routed t… |
| `cand_fp16x3_solvefix.py` | Phase 1a proxy: ib=64 for n=512 (fused-panel route, fewer narrow updates). qr_v2 submission: batched compact-Householder QR for B200. (n=2048 routed t… |
| `cand_fp16x3_tsolvecol.py` | qr_v2 submission: batched compact-Householder QR for B200. (n=2048 routed to custom one-CTA panel + warps; n=4096 to cuSOLVER.) Validated 22/22. Desig… |
| `cand_fp16x3_tune.py` | qr_v2 submission: batched compact-Householder QR for B200. (n=2048 routed to custom one-CTA panel + warps; n=4096 to cuSOLVER.) Validated 22/22. Desig… |
| `cand_fp1a_ib64.py` | Phase 1a proxy: ib=64 for n=512 (fused-panel route, fewer narrow updates). qr_v2 submission: batched compact-Householder QR for B200. (n=2048 routed t… |
| `cand_fp1b_NB128ib64.py` | Phase 1a proxy: ib=64 for n=512 (fused-panel route, fewer narrow updates). qr_v2 submission: batched compact-Householder QR for B200. (n=2048 routed t… |
| `cand_fp1c_NB64ib64.py` | Phase 1a proxy: ib=64 for n=512 (fused-panel route, fewer narrow updates). qr_v2 submission: batched compact-Householder QR for B200. (n=2048 routed t… |
| `cand_fp2a_n1024_NB128.py` | Phase 1a proxy: ib=64 for n=512 (fused-panel route, fewer narrow updates). qr_v2 submission: batched compact-Householder QR for B200. (n=2048 routed t… |
| `cand_fp2b_n1024_NB64.py` | Phase 1a proxy: ib=64 for n=512 (fused-panel route, fewer narrow updates). qr_v2 submission: batched compact-Householder QR for B200. (n=2048 routed t… |
| `cand_fused.py` | === SubmissionV9 = V5 + glue fusion (M-builder + V-builder + M^T-no-transpose) — OFFICIAL 5791us CONFIRMED (V5 5915, -2.1%, best) === Phase 1a proxy: … |
| `cand_fused2br.py` | Phase 1a proxy: ib=64 for n=512 (fused-panel route, fewer narrow updates). qr_v2 submission: batched compact-Householder QR for B200. (n=2048 routed t… |
| `cand_gemmtune.py` | Phase 1a proxy: ib=64 for n=512 (fused-panel route, fewer narrow updates). qr_v2 submission: batched compact-Householder QR for B200. (n=2048 routed t… |
| `cand_geqrf.py` | — |
| `cand_glue.py` | Phase 1a proxy: ib=64 for n=512 (fused-panel route, fewer narrow updates). qr_v2 submission: batched compact-Householder QR for B200. (n=2048 routed t… |
| `cand_gluefuse.py` | Phase 1a proxy: ib=64 for n=512 (fused-panel route, fewer narrow updates). qr_v2 submission: batched compact-Householder QR for B200. (n=2048 routed t… |
| `cand_gluefuse2.py` | Phase 1a proxy: ib=64 for n=512 (fused-panel route, fewer narrow updates). qr_v2 submission: batched compact-Householder QR for B200. (n=2048 routed t… |
| `cand_gram.py` | Phase 1a proxy: ib=64 for n=512 (fused-panel route, fewer narrow updates). qr_v2 submission: batched compact-Householder QR for B200. (n=2048 routed t… |
| `cand_ib16.py` | Phase 1a proxy: ib=64 for n=512 (fused-panel route, fewer narrow updates). qr_v2 submission: batched compact-Householder QR for B200. (n=2048 routed t… |
| `cand_implicitV.py` | Phase 1a proxy: ib=64 for n=512 (fused-panel route, fewer narrow updates). qr_v2 submission: batched compact-Householder QR for B200. (n=2048 routed t… |
| `cand_implicitV_smalln.py` | Phase 1a proxy: ib=64 for n=512 (fused-panel route, fewer narrow updates). qr_v2 submission: batched compact-Householder QR for B200. (n=2048 routed t… |
| `cand_nb128.py` | Phase 1a proxy: ib=64 for n=512 (fused-panel route, fewer narrow updates). qr_v2 submission: batched compact-Householder QR for B200. (n=2048 routed t… |
| `cand_panelstream.py` | Phase 1a proxy: ib=64 for n=512 (fused-panel route, fewer narrow updates). qr_v2 submission: batched compact-Householder QR for B200. (n=2048 routed t… |
| `cand_prec_1x.py` | — |
| `cand_prec_2a.py` | — |
| `cand_prec_2b.py` | — |
| `cand_prec_2br.py` | — |
| `cand_prec_3x.py` | — |
| `cand_pw8.py` | Phase 1a proxy: ib=64 for n=512 (fused-panel route, fewer narrow updates). qr_v2 submission: batched compact-Householder QR for B200. (n=2048 routed t… |
| `cand_recblock.py` | Phase 1a proxy: ib=64 for n=512 (fused-panel route, fewer narrow updates). qr_v2 submission: batched compact-Householder QR for B200. (n=2048 routed t… |
| `cand_solvefix.py` | Phase 1a proxy: ib=64 for n=512 (fused-panel route, fewer narrow updates). qr_v2 submission: batched compact-Householder QR for B200. (n=2048 routed t… |
| `cand_solvefix_1x.py` | Phase 1a proxy: ib=64 for n=512 (fused-panel route, fewer narrow updates). qr_v2 submission: batched compact-Householder QR for B200. (n=2048 routed t… |
| `cand_tsolve.py` | Phase 1a proxy: ib=64 for n=512 (fused-panel route, fewer narrow updates). qr_v2 submission: batched compact-Householder QR for B200. (n=2048 routed t… |
| `scratch/_cand_solveT.py` | Candidate: blocked Householder QR, but T-factor via a single batched triangular solve instead of the sequential per-column _form_T loop. Identity used… |

### raw-CUDA / PTX gates  (3 files)

| file | what it is |
|---|---|
| `cuda_gate1.py` | Gate 1: confirm the load_inline CUDA toolchain compiles + runs on B200 sm_100. Trivial elementwise kernel + a minimal mma.sync.m16n8k8.tf32 smoke test… |
| `cuda_gate2.py` | Gate 2 (decisive): a hand-written tiled tf32x3 batched GEMM via mma.sync.m16n8k8, correctness-checked vs fp64 and timed vs cuBLAS-1xTF32 + our Triton … |
| `cuda_gate3.py` | Gate 3: cp.async 2-stage double-buffered tf32x3 tiled GEMM (overlap global->shared loads with mma compute) -- the optimized-raw ceiling. Same tile/mma… |

### other / helpers  (11 files)

| file | what it is |
|---|---|
| `INVALID_cudagraph_REJECTED.py` | Batched square Householder QR (compact-WY) with CUDA-graph capture. Robustness-first numerics (unchanged from the verified FP32 version); the graph wr… |
| `_real_submission.py` | Phase 1a proxy: ib=64 for n=512 (fused-panel route, fewer narrow updates). qr_v2 submission: batched compact-Householder QR for B200. (n=2048 routed t… |
| `m2_narrowapply.py` | M2.1b — NARROW-NB engine: NB=16/32 super-panel (cheap physical panel tile) + a TMEM-padded tcgen05 apply (pads the NB-dim to NBP=max(NB,64) so TMEM's … |
| `m4_overlap.py` | M4 — CORRECT look-ahead overlap PROTOTYPE (panel hidden behind the GEMM). Sidesteps m2_overlap's broken Tt-smem-staging: each apply RECOMPUTES gram+LA… |
| `recon_b200_caps.py` | Hard-engine recon: (A) what does the B200/Modal Triton toolchain support, and (B) can ANY advanced pure-Triton tf32x3 GEMM beat the current fused kern… |
| `scratch/_force_custom_test.py` | — |
| `scratch/_force_g.py` | — |
| `scratch/_force_j.py` | — |
| `scratch/fp8_ozaki_sim.py` | DECISIVE hardware-free fp8/fp4 Ozaki-slice simulation for the mixed@640 gate. Implements the SAME blocked compact-WY Householder QR as submission.py's… |
| `tri_coop_panel.py` | MAKE-OR-BREAK microbench: at the n=4096 panel shape, does a MULTI-CTA cooperative panel beat the single-CTA-per-matrix panel, AFTER paying the ~2.6us/… |
| `tri_grid_barrier.py` | SERIOUS test: can a HAND-ROLLED atomic cross-CTA (grid-wide) barrier run grep-clean, correct, and reusable in plain Triton on B200? This is the founda… |

<!-- total indexed: 151 of 151 -->
