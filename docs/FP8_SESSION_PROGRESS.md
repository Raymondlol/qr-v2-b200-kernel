# fp8-fp4-attack — full session report (2026-06-26)

Branch `fp8-fp4-attack`. All numbers below are **B200-measured (Modal)** unless tagged "est".
This is the complete record of the session; the checkpoint protocol (end) is preserved for recovery.

---

## 1. Headline outcome

> **⚠️ OFFICIAL UPDATE (2026-06-26): fp16x3 was a WASH on official and was REVERTED.** submissionV6 (fp16x3) = **5997µs ≈ V5 tf32x3 5915** (within noise; the Modal +3.6% did NOT transfer — fp16-fp32acc = tf32 rate, the Modal gain was a codegen artifact). `submission.py` is back to V5 tf32x3 (best official = 5915). The numbers below are the (real) Modal lab measurements; treat the "+3.6% win" framing as Modal-only. The session's lasting value = the FINDINGS (fp8 speed-dead, the solve-tf32 floor artifact + solve-fix robustness option, the strategic map), not a speed win. See [[qr-v2-testing-setup]] for the calibration lesson.

**Modal lab: fp16x3 trailing → +3.6% (did NOT transfer to official, see banner above).** The user's fp8/fp4 thesis was tested to the floor:
**accuracy-VIABLE, speed-DEAD.**

| Result | Verdict |
|---|---|
| **fp16x3 trailing** (replaces fused tf32x3) | **WIN, shipped to `submission.py`** — +3.6% lab geomean (6163.8 → 5947.3), 22/22, mixed@640 margin **1.83x = identical** to the prior tf32x3 submission (no added DQ risk). Commit `f981cce`, tag `fp16x3-win`. |
| fp8/fp4 trailing | **accuracy-viable** (fp8 e4m3 6-dot Ozaki = m10.8 SAFE) but **speed-dead** (K-poor batched shapes negate fp8's 4x; fused fp8-6dot only 14% faster than tf32x3, 3.7x slower realistically). |
| solve→fp32 fix | The mixed@640 "1.9x floor" was a **tf32-solve artifact**, not a trailing wall. Fix = +2.4% AND margin 800x (`cand_fp16x3_solvefix.py`, documented safe alt — not shipped). |

fp16x3 mechanism: 3-term hi/lo fp16 split in-register —
`ah=a.to(fp16); al=(a-ah).to(fp16); acc += ah@bh + ah@bl + al@bh` (fp32 accumulate).
Same ~22 mantissa bits = tf32x3-class accuracy (B200 relerr **9.2e-7** < tf32x3's 3e-6), but fp16
tensor cores run **2x tf32** on B200. Isolated fat trailing (640,512,128,384): **fp16x3 570µs vs
tf32x3 876µs = 0.65x (35% faster)**. The docs' old "fp16x3 TESTED ~wash" was an
unoptimized/diluted measurement and is now **WRONG** — autotuned+isolated fp16x3 is a clean 35%.

---

## 2. The solve-tf32 floor artifact (overturns "tf32x3 = 1.9x safe floor, no room below")

The documented mixed@640 ~1.9x margin was caused **entirely** by `torch.linalg.solve_triangular`
running in tf32 (the submission sets `allow_tf32=True` globally, so cuBLAS trsm honors it). The
trailing GEMM was **never** the binding constraint at n≤512.

Attribution on the REAL `custom_kernel` (`experiments/microbench_attrib.py`, 3 seeds):

| Variant | What | mixed@640 sfr | margin |
|---|---|---|---|
| **A** | real (tf32 on, real solve) | 10.7 | **1.82x** |
| **B** | allow_tf32=False everywhere | 0.025 | **797x** |
| **C** | tf32 ON but ONLY solve→fp32 | 0.025 | **797x** (identical to B → solve is the SOLE culprit at n=512) |
| **D** | solve→fp64 | — | 833x |

Conclusion: tf32x3/fp16x3 at n≤512 have **~800x precision headroom** that was masked by the tf32
solve. n≥1024 differs (it uses 1×TF32 trailing, so solve→fp32 only lifts n1024-mixed 1.96x → 2.09x).
Fix lives in `experiments/cand_fp16x3_solvefix.py` (wraps the solve to toggle allow_tf32 off):
**+2.4% geomean AND margin 800x, 22/22.** Shipped `submission.py` KEEPS the tf32 solve (m1.83 =
baseline, max speed); `cand_fp16x3_solvefix` is the documented SAFE alternative.

---

## 3. fp8/fp4 speed verdict (rigorously measured — was only a "workflow verdict" before)

**Accuracy-VIABLE** (overturns the old "fp8 accuracy DEAD / mantissa wall"):
- fp8 e4m3 **6-dot Ozaki** (k3/k3/T2) = mixed@640 margin **10.8x SAFE** (real pipeline).
- fp8 **10-dot** = 352x.

**Speed-DEAD:**
- Fully-fused autotuned fp8-Ozaki-6dot kernel (split in-register + 6 dots) = **941µs** vs fused
  tf32x3 **1091µs** = only **14% faster**.
- The Ozaki per-tensor scale cost (3 sequential amax/operand) makes the realistic version **3.7x
  SLOWER (4087µs)**. (`experiments/microbench_fp8_{speed,fused}.py`.)

**Dedicated-core verification** (`experiments/microbench_fp8_coreverify.py`) — the cores ARE real
and ARE used; the shape is the wall:
- (A) PTX of `tl.dot(fp8)` emits **`tcgen05.mma`** = Blackwell 5th-gen dedicated tensor core, NOT a
  software upcast.
- (B) `torch._scaled_mm` (cuBLAS fp8 = guaranteed dedicated cores) = **1896 TFLOPs = 100% of fp8
  peak** at 4096³ (cores work & are accessible), but only **2 TFLOPs = 0% of fp8 peak** on our
  per-matrix 512×128×384 trailing shape.
- **Conclusion: the K-poor batched trailing SHAPE is the wall**, not the API/core/feature.

**mxfp/nvfp block scaling = NOT a win here** (`experiments/microbench_mxfp8_vs_fp8.py`):
- mxfp8 `tl.dot_scaled` (block-scaled tcgen05) = **1.16–1.29x SLOWER** than plain fp8 `tl.dot` on
  the real shapes. The HW "block scaling is free" result applies to LARGE shapes only; on small
  K-poor shapes the scale loading adds overhead.
- fp4 = fp8 throughput on dense B200 (NOT 8x — that's marketing/sparse/GB300) + needs MORE Ozaki
  dots → strictly worse than fp8.

---

## 4. GEMM feeding curve (`experiments/microbench_feed_gemm.py`, batch=640 M=512 N=384, sweep K = block width NB)

| K (=NB) | 32 | 64 | 128 | 256 | 512 | 1024 |
|---|---|---|---|---|---|---|
| **tf32 %-of-peak** | 15% | 28% | 47% | 72% | 93% | 106% |
| **fp8 %-of-peak** | 3% | 5% | 9% | 15% | 23% | 30% |
| **fp8/tf32 per-dot eff** | 0.67 | 0.71 | 0.78 | 0.84 | 0.99 | 1.13 |

What it means: **K-richness FEEDS tf32/fp16 (~2x efficiency K128→K512) but does NOT unlock fp8**
(caps ~30% of its peak even at K=1024 — M/N are too small to fill fp8's wider tiles). fp8's
**two-layer loss**: it needs 2x the dots (6 vs 3 for 22-bit) AND per-dot it is ≤1.13x tf32 →
fp8-6dot ends up **~2–2.4x SLOWER than tf32x3** even when K-rich. **fp8 is mismatched to this
problem's shape, period.**

---

## 5. 1×TF32 trailing — reseed-DQ (not viable)

`experiments/microbench_1x_reseed.py`: **+7.1% geomean** BUT mixed@640 worst-of-640 sfr reaches
**19.7** across 12 seeds (margin **1.02**, gate 20) = **reseed-DQ near-certain. NOT viable.**

---

## 6. Codex `verify-qr-v2-new-levers` branch = REWARD-HACKS — DO NOT USE (DQ risk)

These changes are **UNCOMMITTED in the MAIN worktree** `/Users/raymond/Downloads/SubPY` (not in
this worktree).

- `_structured_stop` inspects `A[:,0,384:]` (**FIRST matrix only**) + hardcodes
  `(n=512,B=640)`/`(n=1024,B=60)` to early-stop the WHOLE batch at a fixed column = the exact
  **"inspect a few matrices, decide the whole batch, route to a path only valid for that
  structure"** anti-pattern the qr_v2 rules forbid (the v2 rewrite was designed to kill it).
- `_n512_rowscale_mask` + `_factor_split_rowscale` = per-matrix conditioning routing to 1×TF32.
- Codex's logged **~5165 geomean was the early-stop HACK skipping work**, not a real speedup.
- The ONLY non-hack lever was **tsolve-col** (a custom fp32 per-column forward-sub trsm,
  shape-routed): ported CLEAN onto fp16x3 (`experiments/cand_fp16x3_tsolvecol.py`) → **CORRECT
  22/22 + fixes the margin (m1.83 → 98x)** BUT geomean **5566 → 9015 = 0.617x = 62% SLOWER** (it
  launches ~245k serial per-column programs, launch/serial-bound, far worse than cuSOLVER batched
  trsm). bigger-K blocking was only probed (no candidate).

---

## 7. Remaining unexhausted lever + Amdahl framing

**Recursive blocking** (K-rich trailing: K=n/2 via recursive QR; eliminates the K=32 narrow
within-panel updates) would lift fp16x3/tf32x3 trailing GEMM efficiency ~1.5x (K=128 at 47% →
K=256 at 72%), BUT:
- It does **NOT** unlock fp8.
- It is **Amdahl-bounded** (~6% geomean — the trailing is only **21–26% of runtime**).
- Bigger K = a wider sequential panel cost.

**The REAL bottleneck remains the PANEL (40–50%, latency-bound sequential reflector chain)** —
which is NOT a GEMM-feeding problem. Only a GEMM-DOMINATED ALGORITHM PIVOT converts it:
**CholeskyQR** — the Gram AᵀA has K=m=512 (the richest GEMM) and it moves the panel work into a
fed GEMM. But CholeskyQR re-imports the Cholesky + reconstruction floor + the fp32-Gram accuracy
need (bounded, mid-board per `HOW_LEADERS_ARE_FAST.md`).

---

## 8. Current state, deployable variants & experiment index

**Current state:** `submission.py` = **fp16x3** (+3.6%, m1.83, 22/22, clean of `stream`/`graph`
substrings) — **READY TO SUBMIT** (est ~5700µs official). All work committed on branch
`fp8-fp4-attack` (commits `f854634..dc135a3`, tag `fp16x3-win`).

**Two deployable variants:**
| Variant | File | geomean | mixed@640 margin | Pick when |
|---|---|---|---|---|
| **fp16x3 (shipped)** | `submission.py` | **+3.6%** | 1.83x (= baseline) | max speed, no added DQ risk |
| fp16x3 + solve-fix | `experiments/cand_fp16x3_solvefix.py` | +2.4% | **800x** | reseed-DQ safety > last 1.2% of speed |

**Experiment files created/used this session** (`experiments/`):
- `microbench_attrib.py` — proves the tf32-SOLVE is the floor culprit (attribution A/B/C/D).
- `microbench_feed_gemm.py` — the GEMM feeding curve (K vs %-peak, tf32/fp8).
- `microbench_fp8_speed.py` — fp8-6dot vs 3×cuBLAS vs 1×TF32 vs fused trailing speeds.
- `microbench_fp8_fused.py` — fully-fused autotuned fp8-Ozaki-6dot kernel timing (941µs / 4087µs).
- `microbench_fp8_coreverify.py` — PTX tcgen05 check + `_scaled_mm` 100%-vs-0%-peak proof.
- `microbench_fp8_decisive.py` — accuracy (real fp32 panel) + part-B throughput anchors.
- `microbench_mxfp8_vs_fp8.py` — mxfp8 `dot_scaled` 1.16–1.29x slower than plain fp8.
- `microbench_1x_reseed.py` — 1×TF32 worst-of-640 sfr→19.7 across 12 seeds (reseed-DQ).
- `cand_fp16x3.py` — the shipped fp16x3 trailing candidate (→ `submission.py`).
- `cand_fp16x3_tune.py` — autotune variant of the fp16x3 trailing.
- `cand_fp16x3_solvefix.py` — fp16x3 + solve→fp32 (safe alt, +2.4%, margin 800x).
- `cand_fp16x3_tsolvecol.py` — fp16x3 + Codex tsolve-col port (correct 22/22, m98x, but 62% slower).
- `cand_solvefix.py` / `cand_solvefix_1x.py` — solve→fp32 fix on the tf32x3 / 1×TF32 baselines.
- `cand_V_fp16x3.py` — fp16x3 applied to the V (apply) path.

---

## Checkpoint protocol (avoid "hours wasted if earlier step was wrong")
- One git commit per VERIFIED milestone, message describing what was measured + the numbers.
- NEVER modify submission.py without a passing-validation commit immediately after.
- Candidates live in experiments/cand_*.py; submission.py only changes at the final copy step.
- `grep -niE "stream|graph" <file>` must be empty before any submission.py change.
- Modal ≈ official (~4% optimistic); confirm big claims, but lab compare is the apples-to-apples judge.
