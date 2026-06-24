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
