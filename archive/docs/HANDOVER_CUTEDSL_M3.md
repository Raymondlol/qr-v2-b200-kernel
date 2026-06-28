# HANDOVER → CuTe-DSL engine build (archive of the 2026-06-27 build session)

> Read this, then `docs/M3C_BUILD_GUIDE.md` + `docs/FA4_BLUEPRINT_FOR_QR.md` + auto-memory
> ([[cutedsl-modal-loop]]). Branch `claude/naughty-sammet-cbcc8d`. Comp ends **2026-06-30**.

## §0 — STATE (one line)
Built the cute-DSL warp-spec QR engine bottom-up; **EVERY building block is validated on B200**
(correct QR, blocked+LARFT+WY, warp-split+setmaxregister, look-ahead overlap = **panel HIDES 1.16-1.65×**,
tcgen05 tf32x3). **Remaining = M3c-0: the one integration** (tcgen05 tf32x3 apply inside the per-matrix
overlap CTA). Official best untouched: V10 5343µs (`cand_fused.py` on `persistent-engine`), main = V9 5791.

## §1 — THE ITERATION LOOP (the big unlock; use it)
**cute-DSL iterates on Modal, NO gpumode burn needed.** `modal_cute_lab.py` on a faithful eval-replica
B200 image (cuda12.9.1-devel + py3.13 + torch2.12 + nvidia-cutlass-dsl==4.5.2 + cuda-python13). Run a
candidate: `…/.modalenv/bin/modal run modal_cute_lab.py::run_candidate --script <file>.py` (mounts
experiments/ at /work, execs, prints). `info` = 4.5.2 API introspection; `kernel_smoke`/`run_gemm_example`
= sanity. ~30-60s/run, cents. **Confirmed: the shipped Blackwell tcgen05+TMA+TMEM GEMM PASSES on our image.**

## §2 — VALIDATED BUILDING BLOCKS (all on B200, all committed, all relerr ~1e-6)
| File | What it validates |
|---|---|
| `experiments/cute_qr_m1.py` | correct cute-DSL QR (1 warp, unblocked Householder, `warp_reduce`, geqrf convention) |
| `experiments/cute_qr_m1b.py` | blocked (panel-factor / far-apply phase split) |
| `experiments/cute_qr_m1c.py` | **blocked + LARFT T-factor + WY 2-GEMM apply** C-=V(Tᵀ(VᵀC)) in warp-reduce — the MATH the tcgen05 engine needs, LOCKED |
| `experiments/cute_qr_m2a.py` | **warp-split** panel‖apply + per-warp `setmaxregister_increase(192)/decrease(128)` — the FA4 mechanism that defeats the Gluon static-budget ib=16 wall (COMPILES+RUNS+CORRECT) |
| `experiments/cute_qr_m2b.py` | **look-ahead overlap** panel[k+1]‖far_apply[k] (CTA-barrier delimited) — bit-identical to serial, ov0→ov1 **1.16-1.65×** → **THE SERIAL PANEL HIDES** |
| `experiments/cute_gemm_tf32_m1.py` | own tcgen05 1×TF32 GEMM (rel 8.34e-4) |
| `experiments/cute_gemm_tf32x3.py` | **tcgen05 TF32x3** (rel 2.89e-6) — 3 MmaTF32Op passes hi*hi+hi*lo+lo*hi into one TMEM acc; the apply-precision template |

Key cute-DSL facts confirmed working: dynamic `while`/`for` + scalar gmem `Hm[r,c]` + `shuffle_sync_bfly`
+ ternary on dynamic scalars; `cute.math.sqrt(x,fastmath=True)` (NOT cute.arch.sqrt); `SmemAllocator` +
`@cute.struct` + `get_tensor` + `make_fragment`; `setmaxregister_increase/decrease`; `cute.arch.barrier()`
= __syncthreads (cross-warp gmem visibility). NO `warp_specialize` in 4.5.2 (manual warp roles). cutlass
lib reused for ALL GEMM/TMEM/pipeline (PipelineTmaUmma/UmmaAsync, TmemAllocator, make_smem_layout_a/b,
MmaTF32Op, make_tiled_tma_atom_A/B) — only QR-specific panel/LARFT hand-rolled (like FA's softmax).

## §3 — ★ THE NEXT STEP: M3c-0 integration (do this first)
**Combine the validated blocks into one per-matrix warp-spec CTA.** Follow `docs/M3C_BUILD_GUIDE.md`
§3 (apply tiling), §4 (LARFT — already done in m1c, reuse it), §7 (file skeleton). Concretely:
- Start from `cute_qr_m1c.py` (blocked+LARFT+WY, correct) — it's the closest base (has the WY apply math).
- Split panel(warp0) ‖ mma(warp1) like `cute_qr_m2a/m2b` (setmaxregister + CTA-barrier).
- **Swap m1c's warp-reduce WY reductions for tcgen05 tf32x3 GEMMs** (the mma warp): W1=VᵀC, W2=TᵀW1,
  dV=V·W2, each 3-pass tf32x3 into a TMEM acc (machinery = `cute_gemm_tf32x3.py` verbatim), tiling the
  trailing C, RMW C in gmem. **The one UNVALIDATED piece = in-register tf32x3 limb-split** (`_frag_split`,
  guide §3); SMEM budget forbids 4 limb tensors at full tile. **Gate: if mixed@640 margin <1.83, fall
  back to host-split + smaller tile (BW=64,BM=64,1 stage).**
- Sub-steps (each Modal-tested vs geqrf): M3c-0 (serial, n=128/256 correct) → M3c-1 (overlap ov=1
  bit-identical) → M3c-2 (scale n=512 b640, whole-kernel vs V10 ~9805µs/case + mixed@640 margin≥1.83).
- **Watch:** the (128,256,32) tiler in cute_gemm_tf32x3 is sized for big GEMMs; the apply outputs are
  small ([16,BW], [NBP,BW]) → heavy padding (correct but low-peak; tune tiler AFTER correctness).

## §4 — ★ HONEST EXPECTATIONS (do not over-promise)
- **M3c is 1 CTA/SM** (tcgen05 TMEM ~250 cols + panel regs). So a matrix's serial panel is hidden ONLY
  behind its OWN apply (in-CTA overlap) — **NO cross-matrix masking**. → **realistic landing ~1.0-1.2× V10**.
  M3c's job = prove the tcgen05 apply doesn't regress + the look-ahead recovers the panel. NOT a win alone.
- **The 0.87× (~4650µs) breakthrough needs M6** = persistent work-queue (`StaticPersistentTileScheduler`):
  a persistent CTA pipelines matrix m+1's panel ‖ matrix m's apply across the batch (cross-matrix masking).
  That is THE load-bearing go/no-go. M3c is the prerequisite that makes M6 buildable.
- The leader's 1292µs needs BOTH the panel hidden AND the K-poor trailing lifted (roofline-bounded, ~25%
  peak at NB=64 — memory-bound, low arithmetic intensity). cute-DSL = same PTX as their Gluon; the lever
  is the OVERLAP + (secondary) restoring V9's NB=256 trailing-K inside the fused engine (V10 gave it up).
- **Low-precision (mxfp8/nvfp4):** dead for n≤512 (fp32-C traffic floor + Ozaki-limb bytes); the one real
  opening = n≥1024 (6/12 cases, looser tol) via native Blackwell block-scaled tcgen05 — probe AFTER the
  n=512 engine works. See the low-precision analysis in the session transcript / NEXT_STEPS.

## §5 — HARD GATES (every iter)
1. `grep -niE "stream|graph" <file>` MUST be empty (static scan). cute-DSL launch needs NO stream arg
   (G0/G1 template); AVOID `dense_gemm.py`'s stream plumbing + don't paste `cute.arch.sm_id()` docstring.
2. relerr<1e-4 vs geqrf; **mixed@640 worst-of-640 margin ≥1.83** (probe separately; tf32x3 gives it).
3. Gate every WIN on a real gpumode submission (Modal≈official−4%, Modal image≠eval image).
4. NEVER touch main(=V9) or cand_fused(=V10). B200 = the user's $, be deliberate.

## §6 — K0 still open (the env-confirm)
`experiments/probe_eval_env2.py` (committed earlier; PASS ⟺ cute-dsl import+JIT+launch+correct on the
qr_v2 board) is READY but **not yet submitted** to confirm the board uses the cute-dsl image. The general
kernelbot main image is verified to have it; the board *could* override. Submit once when convenient (zero
leaderboard risk — it falls back to geqrf). v1 (`probe_eval_env.py`) was inconclusive (board hides stdout
on PASS); v2 encodes the answer in PASS/FAIL.

## §7 — repo pointers
`docs/M3C_BUILD_GUIDE.md` (the next-step recipe) · `docs/FA4_BLUEPRINT_FOR_QR.md` (Tri Dao FA cute study →
warp-role table, overlap handshake, reg-realloc, M-plan reorder) · `docs/CUTEDSL_CHEATSHEET.md` (4.5.2 API,
verbatim) · `docs/CUTEDSL_PORT_DESIGN.md` (M-plan + kill-criteria; banner-superseded by FA4 §8) ·
`experiments/cute_ref/` (gitignored NVIDIA example sources — re-fetch via cheatsheet URLs) · FA source at
`~/Downloads/flash-attention-main 2/flash_attn/cute` (flash_fwd_sm100.py mma(), pipeline.py, blackwell_helpers.py).
Session commits: `b691c0c` (M0-M2b), `6515fa9` (M3a), `49178b6` (M3c-pre + guide).
