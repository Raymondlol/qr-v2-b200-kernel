# HANDOVER → CuTe-DSL M6 async warp-spec engine (2026-06-28 session)

> Read after `archive/docs/HANDOVER_CUTEDSL_M3.md`. Branch `claude/gallant-newton-bf3770`. Comp ends 2026-06-30.
> Companion: auto-memory `cutedsl-m3c-validated-primitives.md` (the durable, fuller record).

## §0 — STATE (one line)
M3c-0 tcgen05 tf32x3 apply engine BUILT + CORRECT, then PROFILED + OPTIMIZED, and the **M6 async
warp-spec ring is now WORKING (correct + 1.58× faster than the per-tile-barrier version)** — the
per-tile CTA-barrier wall is broken. Remaining = integrate the ring into the full apply + the panel‖apply
overlap (the 50% lever) + persistent occupancy. **Submission stays the Triton path (official best 4247µs,
id 840028); the cute-DSL engine is a research engine, not yet competitive (per-matrix M3c-0 = 232ms@n512
b640 = ~20× the official 11.8ms).**

## §1 — WHAT WAS BUILT (all B200-validated via modal_cute_lab.py)
| File | What it validates |
|---|---|
| `experiments/cute_apply_probe.py` | hand-fill a swizzled MMA operand by logical coord `sX[(row,ki),0,kb,0]` (refutes "corrupt") |
| `experiments/cute_split_test.py` | in-register tf32x3 split = `(x.bitcast(Int32) & Int32(-8192)).bitcast(Float32)` (python `&`; `cutlass.and_` is LOGICAL=nan; Dekker/TFloat32 round-trip both FOLD to identity) |
| `experiments/cute_apply_gemm1.py` | GEMM-1 W1=Vᵀ·C tf32x3, real orientation+mask+in-reg-split, rel 8.25e-7 |
| `experiments/cute_apply_gemm1_kloop.py` | GEMM-1 K-loop M≤512 via per-tile readback+register-accumulate drain |
| `experiments/cute_apply_chain.py` | FULL 3-GEMM chain dV=V(Tᵀ(VᵀC)) rel 1.8e-6 (transposed restages, K-tail zero, Tᵀ index-swap) |
| `experiments/cute_qr_m3c_tc.py` | **★ INTEGRATED M3c-0** per-matrix engine (panel+gram+LARFT ‖ tcgen05 apply, RMW H−=dV), CORRECT all shapes incl n=512 |
| `experiments/cute_qr_m3c_prof.py` | ablation profile (APPLY/NPASS/DOWR toggles) |
| `experiments/cute_qr_m3c_smem.py` | smem-staged (no gmem scratch; W2-in-sBh G3 fix), −15% |
| `experiments/cute_ring_g1.py` | **★★★ M6 ASYNC RING** — hand-rolled mbarrier ring, G1 producer‖MMA-consumer, NO per-tile barrier, CORRECT + **1.58× faster** |
| `experiments/cute_apply_gemm1_async.py` | the deadlocked PipelineAsyncUmma attempt (kept as the diagnosis artifact) |
| `experiments/cute_pipesrc{,2}.py` | introspection that diagnosed the deadlock + found `tcgen05.commit` |
| `modal_cute_lab.py` | + `run_candidate_quick` (150s subprocess kill, for M6 deadlock-debug) |

## §2 — THE PROFILE (ablation; ncu/nsys are gVisor-dead on Modal)
M3c-0 full 55ms@b64 n=512 = gmem-scratch-write 54% + apply-skeleton-barriers 25% + panel 18% + tf32x3
GEMM-compute 2%. **MEMORY/LATENCY-bound, NOT compute-bound.** smem-staging killed the scratch but only
netted −15% (non-additive overlap) → **the per-tile CTA barriers + serial latency are the TRUE wall.**
2 CTAs/SM (tmem.allocate 256) gave 0 speedup → not occupancy-hideable at that overhead level.
**Codex's profile (Triton submission) CONVERGED:** n1024 panel 50.8% / solve 17.1% / GEMM 16.9%; n2048
panel 42.8% / solve 29.9% / GEMM 15.1% → the algorithm is NOT GEMM-dominated; the win is HIDING the
panel (overlap + occupancy), not faster apply GEMM. Codex's 3ms gate = n512+n1024 bundle −25%.

## §3 — ★ THE M6 RING (the breakthrough; the recipe to reuse)
The deadlock root: cute-DSL's Umma pipelines (`PipelineTmaUmma/AsyncUmma`) arrive the full-barrier via the
async-copy INSTRUCTION (TMA tx / cp.async commit) — `producer_commit` is a `pass` for TMA. Our MANUAL
`st.shared` fill (mandatory for the transposed+masked+split apply operands) never arrives → hang. **FIX =
hand-roll low-level mbarriers** + the UMMA-completion primitive **`tcgen05.commit(mbar)`** (the UMMA arrives
on mbar when the mma-group completes).

Recipe (cute_ring_g1.py, proven correct + 1.58×):
- SMEM: `full_mbar[stages]`, `empty_mbar[stages]`, `acc_done[1]` (`MemRange[Int64]`).
- init (tidx==0): `mbarrier_init(full+s, NPROD)`, `mbarrier_init(empty+s, 1)`, `mbarrier_init(accd, 1)`;
  **PRIME** `mbarrier_arrive(empty+s)` (stages start free); `mbarrier_init_fence()`; one `barrier()`.
- PRODUCER warps: per kt(s=kt%2, ph=(kt//2)%2): `mbarrier_wait(empty+s, ph)`; fill smem[…,s];
  `fence_view_async_shared()`; `mbarrier_arrive(full+s)` (each producer thread → count NPROD).
- CONSUMER (MMA) warp: per kt: `mbarrier_wait(full+s, ph)`; 3-pass `cute.gemm` ACC-accumulate into TMEM;
  **`if lane_idx()==0: tcgen05.commit(empty+s)`** ← elect-one is MANDATORY (all-32 over-arrive the count-1
  barrier → phase desync → deadlock). After loop: `if lane0: tcgen05.commit(accd)`.
- ALL 128: `mbarrier_wait(accd, 0)`; readback (Ld32x32b) → W1.

## §4 — NEXT STEPS (multi-day, foundation proven)
1. **Integrate the ring into the full per-matrix apply** (G1+G2+G3 all warp-spec rings) → measure whole
   n=512 b640 vs the 232ms barrier engine (clean batched test, dispatch-amortized).
2. **panel‖apply overlap** — the SAME mbarrier technique at the super-panel level: panel warp produces
   V/T for super-panel k+1 while the apply consumer processes k. This hides the panel (Codex's 50% lever).
   Look-ahead split (narrow apply k → k+1's cols; far apply k ‖ panel k+1) = the M2b structure, validated
   1.16-1.65× with warp-reduce; now with the tcgen05 ring apply.
3. **persistent grid + occupancy** (cross-matrix panel hiding) — `StaticPersistentTileScheduler`, 148
   CTAs, 2-3 CTAs/SM so matrix m's panel hides behind matrix m'+1's apply.
4. **n2048/n4096** is a SEPARATE lever (Codex Route-3: block-level multi-CTA sharded apply, few matrices
   underfill 148 SMs) — `microbench_block_apply_shard.py` in the codex worktree is the gate.

## §5 — HARD GATES (unchanged)
`grep -niE "stream|graph" <file>` empty before submit · relerr<1e-4 vs geqrf · mixed@640 margin ≥1.83 ·
gate every WIN on a real gpumode submission (Modal≈official−4%) · NEVER touch the official submission
(Triton 4247µs) until a cute engine actually beats it whole-geomean.
