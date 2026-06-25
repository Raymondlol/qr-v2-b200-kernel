# CLAUDE.md — qr_v2 batched-QR kernel (GPU MODE competition, B200)

Read this first, then `README.md`. The `docs/` files are the full handoff.

## What this is
A GPU MODE **qr_v2** kernel competition entry: batched square compact-Householder QR matching `torch.geqrf`, on **NVIDIA B200**, ranked by **geometric mean of 12 benchmark cases**. Deadline **2026-06-30**. We went **123203 → 8580 µs official** (14.4×, ~rank 90); leader is **1332 µs**.
- **Current best = `submission.py`** (shape-routed two-level blocked compact-WY Householder; tf32x3 trailing for n≤512, 1×TF32 for n≥1024; fused Triton panel with per-tile `num_warps`; geqrf/custom routing by shape). 22/22 official tests pass.
- The remaining ~6× gap to the leader is **kernel ENGINEERING (a warp-specialized raw-PTX/TLX tensor-core GEMM engine)**, NOT algorithm — see `docs/HOW_LEADERS_ARE_FAST.md`.

## ⚠️ HARD CONSTRAINTS — violating any = rejected or disqualified
1. **NEVER write the substrings `stream` or `graph` ANYWHERE in a submission, including comments.** The submission checker does a naive static substring scan. `grep -niE "stream|graph" submission.py` MUST be empty before every submit. (No CUDA streams, no CUDA graphs, no cooperative-launch either — but the words alone also trip it.)
2. **Trailing GEMM precision floor = tf32x3 for n≤512.** 1×TF32 FAILS n=512 **mixed at batch=640** (worst-of-640); even tf32x3 has only **~2× margin** there. The competition reseeds and DQs on failure, so don't go below tf32x3 for a safe submission. n≥1024 may use 1×TF32. Panel reductions must stay FP32 (low-precision panel corrupts reflectors).
3. Output must be **flat compact-Householder** (one reflector v per column in strict-lower(H), R=triu(H), tau). `householder_product` reads only strict-lower(H)+tau.
4. **Shape-routing** to correct algorithms is allowed; **conditioning-based routing** to a numerically-invalid path is NOT. `recheck=True` every timed iter (no output caching).

## How to test (no local GPU — this is a Mac). **PRIMARY = the lab harness; full guide in `docs/METHODOLOGY.md`.**
`.modalenv` is in the MAIN repo only — invoke modal by absolute path (works from any worktree): `/Users/raymond/Downloads/SubPY/.modalenv/bin/modal` (modal 1.5.1, authed).
- **Unified lab** (`modal_lab.py` + `harness/lab.py`) — batches baseline+candidates into ONE B200 container (apples-to-apples, no cross-run drift):
  - `modal run modal_lab.py --mode correctness --subs "experiments/cand.py"` — fast 22-case gate, all paths, no timing.
  - `modal run modal_lab.py --mode compare --subs "submission.py,experiments/cand.py"` — same-container variance-aware A/B (first = baseline; per-case paired Δ±stderr, only >2σ counts; ranks by geomean; logs to `results/lab_log.jsonl`).
  - `modal run modal_lab.py --mode profile --subs "submission.py"` — op-level breakdown (panel/GEMM/solve/glue). PROFILE FIRST, test only promising candidates.
- **CPU correctness** (free, instant; eager path + contract ONLY, not Triton kernels): `/Users/raymond/opt/anaconda3/bin/python harness/check_local.py <file> 512`
- **CALIBRATION (2026-06-25): Modal ≈ OFFICIAL** with full warmup (V4: Modal 7793 vs official 7788) — the old "Modal 1.5-2× optimistic" is STALE. count=1 n=512/1024 is noisy → that's why `compare` uses same-container paired reps. Confirm big calls via a real gpumode submission.
- **Legacy** (single candidate): `modal run modal_app.py --submission <f> --stress`; `modal run modal_microbench.py --script experiments/<x>.py`. (raw-CUDA `modal_cuda.py` exists but raw CUDA is UNDEPLOYABLE in eval — see DEAD_ENDS.)

## What's already been tried (DO NOT re-explore — see `docs/DEAD_ENDS.md`)
Mega-kernel/single-CTA trailing (1.4–3.8× slower than batched), naive CholeskyQR2 (22ms, torch chol/trsm are cuSOLVER-slow), fp8/nvfp4 (mantissa wall), lowprec-panel (panel is reduction-bound, not tensor-core), TSQR-output, stream-parallel & CUDA-graph (illegal), sub-tf32x3 precision (1×TF32 fails, 2-term risky), expanded GEMM autotune (no headroom).

## Repo map
`submission.py` (current best) · `milestones/` (score-tagged official versions, m4=8580 confirmed) · `experiments/` (all `cand_*.py` + `microbench_*.py` + `INDEX.md`) · `harness/` (CPU `check_local` + **`lab.py` = unified B200 lab** + gpu_bench + reference) · **`modal_lab.py` (PRIMARY runner)** / `modal_app.py` / `modal_microbench.py` / `modal_cuda.py` · `results/lab_log.jsonl` (structured run log) · `docs/` (README → **METHODOLOGY** → JOURNAL → DEAD_ENDS → NEXT_STEPS → HOW_LEADERS_ARE_FAST). Git: tagged milestones; branch `gluon-tcgen05` = parked deployable-tcgen05 investigation.

## Working notes
- Commit/push only when asked. End commit messages with the Co-Authored-By line.
- Don't edit `harness/` files while a `modal run` is launching (mount snapshot).
- B200 runs cost ~cents each (user's money) — be deliberate; prefer same-run apples-to-apples comparisons over many separate runs.
