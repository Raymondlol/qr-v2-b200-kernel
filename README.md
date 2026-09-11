# qr_v2 — batched Householder QR on a B200, 29× faster than my first working version

A competition entry for the [GPU MODE **qr_v2** leaderboard](https://www.gpumode.com/leaderboard/774):
batched square compact-Householder QR on an NVIDIA B200, ranked by the geometric mean of 12 benchmark
shapes. My first correct submission scored **123,203 µs**. My last one scored **~4,247 µs**.

**This repo is the whole record, not just the winning file** — including the ~15 optimizations that
measured beautifully and then failed on the real leaderboard, and the profiling that explains why I
plateaued 3.3× behind first place. The parts I'd actually point at are
[`docs/DEAD_ENDS.md`](docs/DEAD_ENDS.md) (what didn't work, and why) and
[`docs/METHODOLOGY.md`](docs/METHODOLOGY.md) (how I stopped fooling myself with benchmarks).

| | |
|---|---|
| **Best official score** | **~4,247 µs** geomean (submission id 840028) |
| **Starting point** | 123,203 µs (correct, all-eager PyTorch) |
| **Speedup** | **29×** |
| **Correctness** | 22/22 official test cases, every submission |
| **Leader** | 1,292 µs — I finished ~3.3× behind, and [I know exactly why](#the-gap-i-didnt-close) |
| **Hardware** | NVIDIA B200, no local GPU (developed from a Mac against rented B200s) |
| **Stack** | PyTorch + Triton, plus dead-end excursions into Gluon, raw PTX/CUDA, and CUTLASS cute-DSL |

---

## The task

Implement batched QR matching `torch.geqrf`, in the **flat compact-Householder layout**:

- **Input** `A`: `(batch, n, n)` FP32 CUDA tensor.
- **Output** `(H, tau)`: `R` in `triu(H)`, one Householder vector per column in `strict_lower(H)`,
  reflector coefficients in `tau` `(batch, n)`.
- **Correctness** (measured in FP64, per matrix, all must pass) — the checker rebuilds
  `Q = torch.linalg.householder_product(H, tau)`, which reads *only* `strict_lower(H)` and `tau`:
  - factor residual: `‖triu(H) − Qᵀ@A‖₁ ≤ 20·n·eps32·‖A‖₁`
  - orthogonality: `‖Qᵀ@Q − I‖₁ ≤ 100·n·eps32`   (eps32 = 1.19e-7)
- **Score:** geometric mean over 12 cases, `recheck=True` on every timed iteration (so you cannot
  cache outputs). Batches range from 2 matrices at n=4096 to **640 matrices at n=512**.

Two constraints shaped everything:

1. **The flat layout is load-bearing.** TSQR and other tree-reduction schemes produce reflectors that
   don't fit it, which killed an entire family of otherwise-attractive algorithms.
2. **The submission scanner is a naive substring scan.** A file containing the literal substring
   `stream` anywhere — comments included — is rejected. I lost two submissions to this, one of them a
   perfectly legal kernel whose header comment merely *mentioned* CUDA streams.

## Results

Official leaderboard geomeans. "Modal" numbers below are from my own rented-B200 lab.

| Step | Official geomean | What changed |
|---|---|---|
| eager baseline | 123,203 µs | Correct blocked compact-WY, all PyTorch ops. ~250 launches per call — dispatch-bound. |
| Triton panel | 43,576 µs | Fused Triton sub-panel kernel + `bmm` trailing update. |
| tf32x3 + two-level | 14,760 µs | Emulated-FP32 via 3-pass TF32 inside one fused kernel; two-level blocking (`NB=256` from `ib=32` sub-panels); T-factor via a single batched triangular solve instead of a 60-launch `dlarft` loop. |
| **panel warps** | **8,580 µs** | Launch the panel kernel with `num_warps=4/8/16` by tile size. One line. The per-CTA tile work had been starved at 4 warps: **n=1024 went 54 ms → 10.9 ms**. |
| V4 | 7,788 µs | + route n=2048 to the custom one-CTA-per-matrix panel (cuSOLVER only wins when there aren't enough matrices to fill the GPU). |
| V5 | 5,915 µs | The "engine attack": per-phase profiling, precision routing for the Gram matrix, adaptive sub-panel width, glue elimination. |
| V9 | 5,791 µs | Glue fusion — collapse a ~7-launch elementwise chain into one kernel, and emit `Mᵀ` directly so the triangular solve drops a `.transpose()` that was costing a full copy per apply. Bit-identical output. |
| V10 | 5,343 µs | A **one-shot fused kernel** for n≤512: one CTA factors one whole matrix end-to-end — in-kernel blocked Householder + LARFT + compact-WY apply — killing ~32 launches and the HBM round-trips between them. |
| **V10-sub5 (best)** | **~4,247 µs** | Route `n≤64, batch>16` to that fused kernel too, instead of `geqrf`. The b=20/n=32 case went **318 µs → 30 µs**. |

Reverted, because the leaderboard disagreed with my lab:

| Attempt | Lab said | Official said |
|---|---|---|
| fp16x3 trailing (V6) | +3.6% | 5,997 µs ≈ wash |
| implicit-V apply (V7) | +4.0% | 6,145 µs — **a 3.9% regression** |

Every milestone is a runnable file in [`milestones/`](milestones/), named with the score it got.

## How the final version works

Shape-routed, and **every path is an exact QR** — routing is on shape only, never on conditioning
(a "mixed" batch interleaves well- and ill-conditioned matrices; each one has to factor correctly on
its own merits, so conditioning-based routing would be a correctness bug dressed up as an
optimization).

- **n ≤ 64, batch > 16** → the one-shot fused Triton kernel.
- **n ≤ 256, or n ≤ 512 with batch ≥ 128** → the same fused kernel: 1 CTA per matrix, in-kernel
  blocked Householder → in-kernel LARFT `T` → compact-WY apply, all in registers/SRAM. At these
  shapes the batch itself provides occupancy (640 CTAs fill 148 SMs), so the win is *launch and
  traffic elimination*, not parallelism.
- **larger n** → two-level blocked Householder: super-panel `NB=256` built from `ib=32` fused
  sub-panels, compact-WY `T = (diag(1/τ) + striu(VᵀV))⁻¹` via one batched triangular solve, and one
  fat trailing GEMM.
- **n = 2048, batch ≥ 4** → custom one-CTA-per-matrix panel + 1×TF32 trailing.
- **n ≥ 4096** → `torch.geqrf`. With 2 matrices there are 2 CTAs of work; cuSOLVER parallelizes a
  single matrix across the whole GPU and I can't beat it. This is an honest loss, documented in
  [`docs/DEAD_ENDS.md`](docs/DEAD_ENDS.md).

**On precision.** The trailing GEMM runs in **tf32x3** (three TF32 passes accumulated in FP32 ≈ 22
mantissa bits, full FP32 exponent range) for n≤512, and plain 1×TF32 for n≥1024 where the tolerance
scales with n. Plain TF32 at n=512 *almost* works: across 640 matrices the worst one lands at ~19.7
against a gate of 20. That is not a margin, that's a coin flip against the random seed — so tf32x3 it
is. I later found that even that margin was being eaten by an unrelated bug: `solve_triangular` was
silently running in TF32, and forcing it to FP32 costs ~1% and buys back ~800× of headroom.

## What didn't work

This is the part I'd read first. Full autopsies in [`docs/DEAD_ENDS.md`](docs/DEAD_ENDS.md); the
short list:

- **fp8 / nvfp4.** Accuracy was fine (6-term Ozaki splitting clears the gate). Speed was the problem:
  these shapes are K-poor, and `torch._scaled_mm` hits 100% of fp8 peak at 4096³ and **0%** at
  512×128×384. The FLOP/s number on the spec sheet was never available to this workload.
- **fp16x3.** Same ~22-bit accuracy as tf32x3, and — because fp16 with FP32 accumulate runs at the
  *TF32* rate — the same speed. The 2× fp16 peak is fp16-accumulate only.
- **CholeskyQR.** Rejected as numerically hopeless, then re-examined: library CholeskyQR hits the same
  floor as `geqrf` (cuSOLVER's Cholesky of one n=4096 matrix is itself 24.8 ms), and the only
  drop-in-correct reconstruction re-imports the panel I was trying to escape. *The first-place
  submission used CholeskyQR anyway, with a hand-written blocked tensor-core Cholesky and per-matrix
  precision routing. My "dead end" was really "I stopped one layer too early" — see
  [`docs/HOW_LEADERS_ARE_FAST.md`](docs/HOW_LEADERS_ARE_FAST.md) (written in Chinese).*
- **A raw-PTX tcgen05 trailing engine.** Built it, it worked, ~1.2×. Then I concluded the eval
  environment had no `nvcc` and shelved it — a conclusion that was itself wrong (it was an artifact of
  *my* test image, not the real one). Both the mistake and the correction are in the docs.
- **Warp-specialized panel/trailing overlap.** The panel is ~41% of the time and latency-bound on a
  serial reflector-reduction chain, so hiding it behind the trailing GEMM is *the* lever. Two designs,
  both measured dead: in-CTA `warp_specialize` co-resides only at `ib=16`, which is 6.6% slower
  everywhere else; splitting across CTAs makes it *worse* (the CTAs contend rather than overlap).
- **A hand-rolled atomic grid barrier.** Works, is legal, costs ~2.6 µs — and the panel's per-column
  work is ~0.66 µs, so a cooperative panel is 7–8× slower than just doing it serially. A clean
  negative result.

## Method: how I stopped fooling myself

I had no local GPU. Everything ran on rented B200s, where a naive loop ("run candidate, compare to
last week's number") drowns 3% effects in cross-run drift.

- **[`harness/lab.py`](harness/lab.py)** runs baseline and candidates **interleaved inside a single
  container**: R reps, each rep timing every variant back-to-back, producing per-case paired deltas
  with a standard error. Only >2σ counts.
- **Calibration is a first-class artifact.** My lab reads ~4% optimistic versus the official
  leaderboard — but I only know that because I periodically spent a real submission to measure it.
  The early docs confidently claim "1.5–2× optimistic"; that was true of an earlier harness and went
  stale. Both numbers are in the record.
- **The lesson that cost me three submissions:** *a sub-5% lab win is not a win.* fp16x3 (+3.6%),
  implicit-V (+4%) — both evaporated or reversed officially. The pattern turned out to be legible:
  **structural** changes (fewer kernels, less traffic, one less transpose) transfer; **codegen-
  sensitive** micro-optimizations do not. Glue fusion showed +7.4% in the lab and +2.1% officially —
  smaller, but real, and in the same direction.
- **Profiling substitutes.** `ncu` hardware counters are impossible on my provider (gVisor sandbox —
  the capability nodes simply aren't there), so [`docs/PROFILING.md`](docs/PROFILING.md) documents
  what I used instead: kineto op breakdowns, launch-gap analysis, Triton compile-time register/spill
  counts, and SASS instruction mixes via `cuobjdump`.

## The gap I didn't close

First place is 1,292 µs; I'm at 4,247 µs. The profiling is unambiguous about why. At n=512 I run the
trailing GEMM at **5.5% of tf32x3 peak** — not because the GEMM is bad, but because it's only ~27% of
the wall clock. The panel (43%), the triangular solve (15%), and the glue (~14%) run at ~0% tensor-core
utilization, and they're *serialized* with it.

The leaders don't make those phases faster. They **hide** them: a persistent warp-specialized `tcgen05`
engine with TMA producer warps, a multi-stage mbarrier pipeline and 2-SM tiles, so the critical path
collapses to the GEMM alone. Plus an algorithmic reformulation (CholeskyQR panel), per-matrix precision
routing, and launch-gap elimination.

I built pieces of that engine three different ways — Gluon, raw PTX, CUTLASS cute-DSL (all validated on
B200, all in this repo) — and never got the whole thing integrated. [`docs/HOW_LEADERS_ARE_FAST.md`](docs/HOW_LEADERS_ARE_FAST.md)
is my post-mortem against the actual top-3 code once it was published: I had predicted the engine and
missed the algorithm.

## Repo map

```
submission.py            the ~4,247 µs entry (identical to the milestones/ snapshot of it)
milestones/              every scored version, filename = its official score
experiments/             ~120 single-variable candidates + INDEX.md
harness/                 CPU correctness gate, B200 benchmark, the A/B lab  (mixed provenance: see NOTICE)
modal_*.py               B200 runners: lab, microbench, profiling probes
results/                 raw run logs, including the GO/NOGO verdicts
docs/                    METHODOLOGY · JOURNAL · DEAD_ENDS · PROFILING · HOW_LEADERS_ARE_FAST
archive/                 superseded docs, kept rather than deleted
CLAUDE.md                working instructions for the AI agent (see below)
```

Suggested reading order: `docs/JOURNAL.md` (the score ladder, step by step) →
`docs/DEAD_ENDS.md` → `docs/METHODOLOGY.md` → `docs/PROFILING.md`.

## Reproducing

CPU correctness gate — free, instant, no GPU. Validates the eager path and the `(H, tau)` contract
against the competition's own checker (Triton paths don't run on CPU):

```bash
python harness/check_local.py submission.py 512
```

B200 timing and the full 22-case gate need a GPU. I used [Modal](https://modal.com); with a
`modal` install and an account:

```bash
modal run modal_lab.py --mode compare --subs "submission.py,experiments/cand_X.py"
```

`--mode correctness` for the 22-case gate, `--mode profile` for the op-level breakdown. Costs cents
per run. Before any submission: `grep -niE "stream|graph" submission.py` must be empty.

## Notes

**AI-assisted.** Most of this was written with Claude Code driving, which is why the history is full
of `Co-Authored-By: Claude` and why `CLAUDE.md` exists — it's the instruction file that kept the agent
oriented across sessions, and it doubles as a log of what I'd learned the hard way. The judgment calls
that mattered were mine: deciding that a +3.6% lab win wasn't real, that conditioning-based routing
was cheating rather than clever, and that the overlap design was dead before sinking another week into
it. Two of those three, I got right.

**License.** MIT (`LICENSE`) for my code. `harness/` contains the competition's own evaluation
harness, which is not mine — see [`NOTICE`](NOTICE) for what's vendored and where it came from.
No third-party submission source is reproduced anywhere in this repo.
