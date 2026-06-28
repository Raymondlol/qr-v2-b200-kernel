# qr_v2 — Batched Compact-Householder QR kernel (GPU MODE competition)

**One-line status:** confirmed official: 7788 (V4) → 5915 (V5) → 5791 (V9) → 5343 (V10, id 838292) → **~4247µs = CURRENT BEST (4ms-level, id 840028, Triton path; see `docs/HANDOVER_CUTEDSL_M6.md`)**. `submission.py` on this branch = V9 (5791µs), 22/22; the 4247 submission is snapshotted at `milestones/submissionV10sub5_fused32_ib32_nw2_official4247.py` (V10 + n≤64/B>16 routed to the fused Triton kernel; codex branch `xinbranch`). The `fp8-fp4-attack` session (2026-06-26) tried **FP16x3** (submissionV6) — Modal said +3.6% but **official = 5997 ≈ V5 5915 = WASH** (reverted; fp16-fp32acc = tf32 rate, the Modal gain was a codegen artifact). **fp8/fp4 = speed-dead** (K-poor batched shape; dedicated tcgen05 cores verified used). Net session speed gain ≈ 0; lasting value = the findings + the solve-tf32 robustness option. Leader **1292 µs**. Deadline **2026-06-30**. See `archive/docs/FP8_SESSION_PROGRESS.md`.

**Corrected understanding (this session's profiler overturned the old thesis):** the gap is NOT just a trailing-GEMM engine — the trailing GEMM is only **21–26%**; the **panel is ~40–50%** (latency-bound by the sequential reflector reduction chain) and is the real bottleneck. raw-PTX tcgen05 trailing engine: built + works (~1.2×) but **UNDEPLOYABLE** (no nvcc in eval) → dead. Gluon tcgen05 (`triton.experimental.gluon`, in stock triton 3.6.0) IS the deployable warp-spec path but bounded ~1.1× and doesn't touch the panel (parked on branch `gluon-tcgen05`). A two-level peer hits 3352 (rank 34) → ~1.7× headroom likely still exists in our own family. **Precision (CORRECTED 2026-06-26):** tf32x3 (~22-bit) is the deployable floor, but the mixed@640 "~2× margin" was a **tf32-SOLVE artifact** (solve→fp32 → ~800× margin, a robustness option); fp16x3 = same accuracy AND ~same speed (official wash); fp8/fp4 = accuracy-viable but speed-dead. **UPDATE 2026-06-26 research session: the Gluon warp-spec panel/trailing OVERLAP is now measured-DEAD** (design A blows the 16-warp register budget; smem-streaming panel 2.56-6.74× too slow), along with CholeskyQR (LinAlgError + conditioning-routing forbidden), multi-CTA cooperative panel (barrier works but per-column granularity 7-8× too coarse), recursive-blocking (0.976×), cuBLAS-trailing (0.866×), and TLX (infeasible on Modal). **The one remaining crux = a panel that is BOTH fast AND co-residable** (MAGMA register-file panel ½-day gate → design B / a multi-week warp-spec persistent tcgen05 engine). **`archive/docs/HANDOVER_NEXT_SESSION.md` = the final state; `archive/docs/NEXT_SESSION_PROMPT.md` = the autonomous plan.** See also `docs/HOW_LEADERS_ARE_FAST.md` + `docs/DEAD_ENDS.md`.

This README is the handoff. Read `CLAUDE.md` first, then **`docs/METHODOLOGY.md` (how to run experiments — use `modal_lab.py`, NOT the old one-run-per-candidate loop)**, then `docs/JOURNAL.md` (how we got here), `docs/DEAD_ENDS.md` (what NOT to retry), `archive/docs/NEXT_STEPS.md` (path forward), `docs/HOW_LEADERS_ARE_FAST.md`.

---

## 1. The task (authoritative spec)

GPU MODE **qr_v2**, leaderboard https://www.gpumode.com/leaderboard/774 . GPU = **B200**.

Implement batched square compact-Householder QR matching `torch.geqrf`.
- **Input** `A`: `(batch, n, n)` FP32 CUDA tensor.
- **Output** `(H, tau)`: `H` `(batch,n,n)` with `R` in the upper triangle and Householder vectors below the diagonal; `tau` `(batch,n)` reflector coefficients — the **flat compact layout** (one reflector per column).
- **Checker** (FP64-measured, per matrix, ALL must pass): builds `Q = torch.linalg.householder_product(H, tau)` (reads ONLY strict-lower(H)+tau), `R = triu(H)`.
  - factor residual: `‖triu(H) − Qᵀ@A‖₁ ≤ 20·n·eps32·‖A‖₁`
  - orthogonality: `‖Qᵀ@Q − I‖₁ ≤ 100·n·eps32`   (eps32 = 1.19e-7)
- **Ranking:** geometric mean of 12 benchmark cases. `recheck=True` every timed iter (no output caching).

### The 12 benchmark cases + current per-case official times (from the 8580µs run)
| case | batch | n | official time | path |
|---|---|---|---|---|
| dense | 20 | 32 | 322 µs | geqrf |
| dense | 40 | 176 | 700 µs | custom (block 64) |
| dense | 40 | 352 | 2.77 ms | custom (two-level) |
| dense | 640 | **512** | **14.8 ms** ×1 | custom two-level + tf32x3 |
| mixed/rankdef/clustered | 640 | 512 | ~14.9 ms ×3 | same |
| dense | 60 | **1024** | **10.9 ms** ×1 | custom + 1×TF32 |
| mixed/nearrank | 60 | 1024 | ~11 ms ×2 | same |
| dense | 8 | **2048** | **76.8 ms** | geqrf (in m4; m5 routes to custom → ~35ms) |
| dense | 2 | **4096** | **52.2 ms** | geqrf |

**The cost is dominated by n=2048 + n=4096 (the geqrf floor) and the four n=512 cases.** See `archive/docs/NEXT_STEPS.md`.

---

## 2. HARD CONSTRAINTS (learned the hard way — violate and you get rejected/disqualified)

1. **NO multiple CUDA streams. NO CUDA graphs.** Enforced by a **naive static substring scan** at submission time that rejects the literal substrings **"stream"** and **"graph"** ANYWHERE in the file **including comments**. (The CUDA-graph version AND a clean tf32x3 version were both rejected — the latter only because its header comment contained the words "CUDA streams / graphs".) **Before every submission run: `grep -niE "stream|graph" submission.py` must return nothing.**
2. Output must be the **flat** one-reflector-per-column compact-H layout (TSQR's tree reflectors do NOT fit — see DEAD_ENDS).
3. Shape-based routing to two *correct* algorithms is fine. Conditioning-based routing to a numerically-invalid path is **not** (mixed batches interleave well/ill-conditioned; each matrix must factor correctly on its own merits).

---

## 3. How the current solution works (`submission.py` = m5 / cand_Y)

Two-level blocked Householder, **shape-routed** (all paths are exact QR):
- `n ≤ 64` or `batch ≤ 16` (small) → `torch.geqrf` (cuSOLVER wins there).
- `n = 2048, batch ≥ 4` → custom one-CTA-per-matrix panel (8 CTAs = enough occupancy to beat geqrf 2.2×) + 1×TF32 trailing.
- `n ≥ 4096` (or n=2048 small-batch) → `torch.geqrf` (only 2 CTAs otherwise — geqrf floor, see NEXT_STEPS).
- **large-batch medium-n** → two-level: super-panel `NB=256` factored from `ib=32`-wide **fused Triton sub-panels** (`_panel_kernel`, one program/matrix, holds `[next_pow2(m), 32]` tile in SRAM, **launched with `num_warps=4/8/16` by tile size — THE recent 5× win on n=1024**); compact-WY `T = (diag(1/τ)+striu(VᵀV))⁻¹` via ONE batched `solve_triangular`; big trailing GEMMs via a **fused tf32x3 Triton kernel** (`_bmm_x3_kernel`, in-register 3-pass = emulated FP32) for `n≤512`, plain **1×TF32** `torch.matmul` for `n≥1024`.

**Why tf32x3 and not lower precision:** the factor gate at batch=640 is tight enough that even 1×TF32 fails n=512 mixed (worst-of-640 → scaled ~19.7 > 20 on the worst seed = reseed-DQ). tf32x3 keeps FP32 exponent range + ~22 mantissa bits and passes all 22 tests. **(CORRECTED 2026-06-26)** fp16x3 has the SAME ~22-bit accuracy AND ~same speed (fp16-fp32acc = tf32 rate → official wash); fp8/fp4 are accuracy-viable (m10.8 @ 6 Ozaki dots) but **speed-dead** on the K-poor batched shape (dedicated tcgen05 cores verified used; even cuBLAS `_scaled_mm` gets 0% of fp8 peak there). See `archive/docs/FP8_SESSION_PROGRESS.md` + DEAD_ENDS.

---

## 4. How to test (no local GPU — this is a Mac)

### CPU correctness (free, instant; validates the eager/torch path + the geqrf contract)
```bash
/Users/raymond/opt/anaconda3/bin/python harness/check_local.py submission.py 512
```
(anaconda py3.8 has torch 2.2.2. The harness `reference.py`/`eval.py` have `from __future__ import annotations` prepended and `task.py` shimmed for py3.8.)

### B200 timing + the full 22-case correctness gate (Modal)
```bash
source .modalenv/bin/activate              # modal 1.5.1, already authed
modal run modal_app.py --submission submission.py --stress   # --stress = full 22-case gate
modal run modal_app.py --submission experiments/cand_X.py --filt 2048   # one case
```
`harness/gpu_bench.py` reproduces eval.py's leaderboard timing (CUDA events, L2 clear, batch-count, recheck) + a full warmup pass; prints per-case µs + geomean.

### ⚠️ CRITICAL: Modal ≠ competition hardware for absolute speed
Modal's B200 is **~1.5-2× FASTER** than the competition's on **tf32/fp16 tensor-core-heavy** cases (n=512/1024 trailing GEMMs). BUT **geqrf cases, FP32/panel work, and small cases MATCH official EXACTLY**. So:
- Trust Modal for **correctness** and **relative ranking** of candidates, and for **panel/FP32 speed**.
- Do NOT trust Modal absolute speed of low-precision GEMMs — **validate via real submissions**.
- Surprises: the panel-warps win measured ~2× on Modal but **5× officially** (Modal under-estimates panel/FP32 wins → real gains ≥ Modal estimate for that class).

To submit: upload the `.py` via the gpumode.com site (B200, leaderboard mode). ~4 min/run.

---

## 5. The journey (123203 → 8580, 14.4×)
`submission.py`(eager 123203) → triton 43576 → tf32x3+two-level **14760** → +panel-warps **8580** → +route n=2048 **~8100 (candidate)**. Full detail + per-step B200 data in `docs/JOURNAL.md`.

## 6. Next steps
The only positive-EV lever left is breaking the **geqrf floor on n=4096** (and squeezing n=512). See `archive/docs/NEXT_STEPS.md` — the leading hypothesis is **full kernel fusion + GEMM-heavy (CholeskyQR) narrow panels** to make the panel tensor-core-bound and high-occupancy. The dead ends (fp8/fp4/lowprec-panel/TSQR/load_inline) are in `docs/DEAD_ENDS.md` — **do not re-explore them.**
