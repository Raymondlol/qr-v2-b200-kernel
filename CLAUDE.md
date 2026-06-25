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

## How to test (no local GPU — this is a Mac)
- **CPU correctness** (free, validates eager/torch path + contract): `/Users/raymond/opt/anaconda3/bin/python harness/check_local.py <file> 512`
- **B200 (timing + 22-case gate)**: `source .modalenv/bin/activate && modal run modal_app.py --submission <file> --stress`  (modal 1.5.1, already authed; `--stress` = full 22-case correctness gate; gpu_bench prints per-case µs + geomean + factor-residual margin).
- **Microbench an arbitrary script on B200**: `modal run modal_microbench.py --script experiments/<x>.py`
- **⚠️ Modal ≠ competition B200 for ABSOLUTE speed**: Modal is ~1.5–2× FASTER on tf32-tensor-core cases (n=512/1024 trailing), but EXACT on geqrf/FP32/panel/small. Trust Modal for correctness + relative ranking + panel/FP32 speed; **validate absolute speed via real submission** (upload to gpumode.com). Modal n=512/1024 timing is also noisy (count=1, ±15%) — compare candidates in the SAME `modal run` (apples-to-apples).

## What's already been tried (DO NOT re-explore — see `docs/DEAD_ENDS.md`)
Mega-kernel/single-CTA trailing (1.4–3.8× slower than batched), naive CholeskyQR2 (22ms, torch chol/trsm are cuSOLVER-slow), fp8/nvfp4 (mantissa wall), lowprec-panel (panel is reduction-bound, not tensor-core), TSQR-output, stream-parallel & CUDA-graph (illegal), sub-tf32x3 precision (1×TF32 fails, 2-term risky), expanded GEMM autotune (no headroom).

## Repo map
`submission.py` (current best) · `milestones/` (score-tagged official versions, m4=8580 confirmed) · `experiments/` (all `cand_*.py` + `microbench_*.py` + `INDEX.md`) · `harness/` (CPU + gpu_bench + reference checker) · `modal_app.py` / `modal_microbench.py` · `docs/` (README → JOURNAL → DEAD_ENDS → NEXT_STEPS → HOW_LEADERS_ARE_FAST). Git: tagged milestones (`v-8580-confirmed`, `v-fusedpanel`, …).

## Working notes
- Commit/push only when asked. End commit messages with the Co-Authored-By line.
- Don't edit `harness/` files while a `modal run` is launching (mount snapshot).
- B200 runs cost ~cents each (user's money) — be deliberate; prefer same-run apples-to-apples comparisons over many separate runs.
