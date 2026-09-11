# Experiment methodology — the lab harness (use this, not the old one-run-per-candidate loop)

The old loop (`modal_app.py --submission cand_X.py --stress`, then eyeball geomean vs a
baseline from a *different* run) had two structural flaws that repeatedly bit us:
- **cross-run clock drift** → single-shot geomean comparisons lie (the NB128 sweep
  "won" in isolation but regressed in the full run; count=1 n=512/1024 is ±15%).
- **sequential round-trips** → one candidate per Modal run + several model turns each.

`harness/lab.py` + `modal_lab.py` replace it. Five upgrades:

1. **Same-container variance-aware A/B.** `--mode compare --subs "base.py,cand1.py,cand2.py"`
   ships baseline + all candidates to ONE B200 container and times them **interleaved**
   (all kernels timed back-to-back within each rep → shared clock state), R reps. Reports
   per-case mean ± stderr and the **paired Δ vs baseline** + its stderr. A per-case win is
   flagged only if `|Δ| > 2·stderr` AND `> 2%` — so noise can't masquerade as a win.
   Ranks all candidates by geomean in one table, one run, one model turn.
2. **Fast correctness gate.** `--mode correctness` runs the 22 official-shape STRESS cases
   (small batches, **every code path**, NO timing) → catches Triton-kernel bugs (e.g. the
   autotune-in-place `restore_value` bug) in seconds, before any timing run.
3. **Profile.** `--mode profile --subs sub.py` → torch.profiler op-level breakdown
   (PANEL / tf32x3 GEMM / cuBLAS GEMM / cuSOLVER / elementwise glue), per case. Profile
   FIRST, predict each candidate's effect from the breakdown, then test only the promising
   ones — guess-and-check is expensive (profiling is what found the 20% glue = the 1.18× win).
4. **Deterministic generate→test→promote loop** (orchestrate with a Workflow):
   - design phase (model / workflow agents) proposes K candidate *diffs* against the
     current `submission.py`;
   - apply each to a `cand_*.py`; run `modal_lab --mode correctness` on all (drop failers);
   - run `modal_lab --mode compare` on `submission.py + survivors` (one run);
   - promote the candidate whose geomean win is significant (>2σ); else stop.
   The model is only in the design/synthesis steps; the grind is one batched run per round.
5. **Structured log.** Every run appends its JSON (`{mode,paths,t,result}`) to
   `results/lab_log.jsonl` — never re-measure; trend/regression analysis; "what class of
   change helped" meta-analysis.

## Cadence discipline — keep the three concerns separate
| concern | tool | when |
|---|---|---|
| correctness (exhaustive, fast, pre-timing) | `--mode correctness` | every candidate, first |
| relative ranking (same-container, variance) | `--mode compare` | batch of survivors, one run |
| absolute calibration (truth) | real gpumode submission | rare, deliberate (Modal ≈ official now) |

## Rules carried over
- Submission file must contain NEITHER substring `stream` nor `graph` (incl. comments —
  "streamline" tripped it once). `grep -niE "stream|graph" submission.py` before every submit.
- tf32x3 is the precision floor for n≤512 (mixed@640 margin 2.0×). Panel stays fp32.
- Promote by overwriting `submission.py` only after a significant `compare` win + 22/22 gate.
- CPU `harness/check_local.py` still validates the eager path + contract for free (instant),
  but it does NOT exercise the Triton kernels — use `--mode correctness` for those.

## Example
```bash
source .modalenv/bin/activate   # or use modal
modal run modal_lab.py --mode correctness --subs "experiments/cand_new.py"
modal run modal_lab.py --mode compare --subs "submission.py,experiments/cand_new.py"
modal run modal_lab.py --mode profile --subs "submission.py"
```
