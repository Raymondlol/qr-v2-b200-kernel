# Experiments index

Each `cand_*.py` is a single-variable experiment. Numbers are Modal geomean unless marked official. Lineage flows roughly A→Y. See `../docs/JOURNAL.md` for the curated milestones and `../docs/DEAD_ENDS.md` for what failed and why.

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
