# fp8-fp4-attack — session progress & autonomous-exploration plan (2026-06-26)

Branch `fp8-fp4-attack`. This doc is the **checkpoint/recovery note** + the plan for overnight
autonomous work. Each verified milestone = one git commit (see "Checkpoint protocol").

## ✅ FINAL OUTCOME (2026-06-26) — SHIPPED a +3.6% win
1. **fp8/fp4 verdict (the user's thesis): accuracy-VIABLE, speed-DEAD.** Sub-tf32 precision works
   (the prior "wall" was a tf32-solve artifact, see below); fp8 e4m3 6-dot Ozaki holds mixed@640
   m10.8 (SAFE). BUT fp8 is NOT a speed win: K-poor batched trailing shapes negate fp8's 4x
   throughput — fused autotuned fp8-6dot = 941us ≈ fused tf32x3 1091us, and the Ozaki scale cost
   makes it 3.7x slower realistically. fp4=fp8 throughput on B200 + more dots = strictly worse.
2. **Solve-fix discovery:** mixed@640's "1.9x floor" = the tf32 triangular solve, not the trailing.
   Forcing solve→fp32 → margin 1.83x→800x at +~1%. (`cand_fp16x3_solvefix`, `microbench_attrib`.)
3. **1xTF32 trailing:** +7.1% geomean BUT mixed@640 worst-of-640 sfr→19.7 (m1.02) across seeds =
   reseed-DQ near-certain. Not viable.
4. **WINNER, SHIPPED to submission.py (tag `fp16x3-win`, commit f981cce):** FP16x3 trailing —
   tf32x3-class accuracy (relerr 9e-7 < tf32x3 3e-6), but fp16 TC=2x tf32 → fat trailing 570 vs
   876us. **Lab geomean +3.6% (6163.8→5947.3), 22/22, mixed@640 m1.83 = identical to the old
   tf32x3 submission (no added DQ risk).** Strictly better than the prior submission.
   - **Safe alt (not shipped):** `cand_fp16x3_solvefix` = fp16x3 + solve-fix = +2.4% AND margin
     800x. Pick this if reseed-DQ safety > the last 1.2% of speed.


## BREAKTHROUGH (measured, trustworthy — see memory `qr-v2-solve-tf32-floor-artifact`)
The documented "mixed@640 tf32x3 = 1.9x safe floor, no room below" was **WRONG** — it was a
**tf32 triangular-SOLVE artifact**, not a trailing-GEMM precision wall.
- Real submission.py mixed@640 → sfr 10.7, margin 1.82x (reproduced, all seeds).
- Forcing ONLY `torch.linalg.solve_triangular` to fp32 → sfr 0.025, **margin 797x** (B200 attribution,
  `experiments/microbench_attrib.py`). The solve was running in tf32 (global allow_tf32=True).
- Fix = `experiments/cand_solvefix.py` (wrap solve to toggle allow_tf32 off). Cost **+0.7% geomean**
  (lab 6371.7→6413.6, 22/22-shaped 12 cases pass). Unlocks n≤512 margins to 800–1700x.

## With the solve fixed, sub-tf32 trailing is VIABLE (the fp8 win, real pipeline, worst-of-640):
- fat trailing 1xTF32 → m1.23 (risky, <2x)
- fat trailing **fp8 e4m3 6-dot (Ozaki k3/k3/T2)** → **m10.8 SAFE** ← fp8's niche
- fat trailing fp8 10-dot → m352 ; tf32x3 → m797
(narrow within-panel updates kept tf32x3 in these tests; panel always fp32.)

## SPEED anchors (B200, real shapes, `experiments/microbench_fp8_decisive.py` part B):
- fused tf32x3 trailing (current) on (640,512,128,384) = **1267µs** (slow, ~16% of ceiling)
- 1xTF32 cuBLAS same shape = 139µs ; tf32x3-as-3xcuBLAS ≈ ~400-500µs (est, MEASURE)
- fp8 4x tf32 peak on B200, fp4≈fp8 (NOT 8x), block-scaling free (research, high-conf)
- fp8-6dot est ~250µs → ~5x faster than fused; MEASURE via `microbench_fp8_speed.py`
  (uses PLAIN tl.dot on fp8 slices — NOT dot_scaled, which threw CompilationError; plain works
  because Ozaki slices are per-tensor scaled in fp32 outside the matmul).

## PLAN (ranked; do in order, checkpoint each)
1. **Measure trailing speeds**: run `microbench_fp8_speed.py` → get fp8-6dot vs 3xcuBLAS vs 1xTF32
   vs fused, on fat + narrow + n1024 shapes. Pick fastest SAFE option for the fat trailing.
2. **Build winning candidate** = solve-fix + fastest-safe fat trailing. Likely either:
   (a) fp8-6dot fat trailing (deployable batched plain-fp8 Triton kernel), or
   (b) tf32x3-via-3xcuBLAS (no new kernel, m797) if it's competitive — simplest/safest.
   Keep narrow updates + gram + panel as-is first; extend later if margin allows.
3. **VALIDATE (gate before any submission.py change):**
   - 22/22 correctness: `modal run modal_lab.py --mode correctness --subs "experiments/cand.py"`
   - mixed@640 worst-of-640 margin ≥2.0x: adapt `microbench_verify_solvefix.py` (lab corr is
     batch-16, MISSES worst-of-640 — must probe separately!)
   - geomean: `modal run modal_lab.py --mode compare --subs "submission.py,experiments/cand.py"`
4. If win confirmed (faster + 22/22 + margin≥2x): copy cand → submission.py, re-verify, commit+tag.
5. Stretch: extend low-prec to narrow updates / gram / W (more speed if margin holds); n≥1024 path
   (already 1xTF32 trailing; solve-fix gives it 2.09x — could add fp8 for more margin or speed).
6. Update docs (DEAD_ENDS, NEXT_STEPS, CLAUDE.md, README) once a win lands.

## Checkpoint protocol (avoid "hours wasted if earlier step was wrong")
- One git commit per VERIFIED milestone, message describing what was measured + the numbers.
- NEVER modify submission.py without a passing-validation commit immediately after.
- Candidates live in experiments/cand_*.py; submission.py only changes at step 4.
- `grep -niE "stream|graph" <file>` must be empty before any submission.py change.
- Modal ≈ official (~4% optimistic); confirm big claims, but lab compare is the apples-to-apples judge.

## Key files this session
- `experiments/cand_solvefix.py` — solve→fp32 fix (the foundation; +0.7%, margin 800x). 22/22 TBD-confirm.
- `experiments/microbench_attrib.py` — proves the solve is the floor culprit.
- `experiments/microbench_verify_solvefix.py` — stress-case margins for solve-fix variants.
- `experiments/microbench_fp8_decisive.py` — accuracy(real fp32 panel) + part-B throughput.
- `experiments/microbench_fp8_speed.py` — fp8-6dot vs alternatives speed (RUN THIS FIRST on resume).
- `experiments/_real_submission.py` — copy of submission.py for the microbenches to import.
- `experiments/scratch/fp8_ozaki_sim.py` — the agent's CPU Ozaki sim (heavy on CPU; numbers in sweep_out.txt).
