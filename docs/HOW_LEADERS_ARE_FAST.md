# How the top qr_v2 submissions are fast — a post-mortem against the real top-3 code

> **Source note.** The qr_v2 submissions were published on the
> [leaderboard](https://www.gpumode.com/leaderboard/774) after the 2026-06-30 deadline; this document
> is my own reading of that public code. **No third-party source is reproduced here** — only short
> quotes of a few comment lines, and the kernel names needed to make the analysis checkable. Every
> architectural claim is an inference about what the code does, not a claim about anyone's intent.
> Index and provenance: [`docs/leader_top3/README.md`](leader_top3/README.md).
>
> **Written 2026-06-30, after the deadline**, replacing an earlier pure-speculation version. That
> earlier version had *ruled CholeskyQR out*. The real code shows the opposite: a CholeskyQR
> reformulation is the winner's number-one lever. My old headline judgement — "same algorithm, the
> gap is pure engineering" — was half right, and the wrong half was the expensive half.
>
> A Chinese-language version of this document is kept verbatim at
> [`HOW_LEADERS_ARE_FAST.zh.md`](HOW_LEADERS_ARE_FAST.zh.md).
>
> **How to read it.** §2 is the skeleton all three share. §3 is where they differ. §4 is the lesson
> in the *ranking itself*. **§5 is the part I'd read — what I got wrong and why.** §6 is a
> copy-this-next-time table with real function names.

---

## 0. TL;DR

**The gap is ≈20% algorithm and ≈80% three things:** (1) turn the latency-bound Householder panel
into tensor-core work; (2) route precision **per matrix**, which is what makes fp16 safe on
mixed@640; (3) eliminate the launch gap (Triton fusion / PDL / explicit-node CUDA graph).

And: **first place is pure Triton. The hand-written tcgen05 engine finished third.**

---

## 1. What the three submissions are (A = 1st > B = 2nd > C = 3rd)

| # | Vehicle | How the panel is attacked | Trailing / precision | Launch gap | Distinctive |
|---|---|---|---|---|---|
| **A** (1st) | **pure Triton** | **CholeskyQR reconstruction**: Gram → blocked Cholesky → back out compact-WY (V, τ) | per-matrix precision classifier (6+ structural classes) + fp16 working buffer + mxfp8 two-term | Triton few-kernel fusion + `USE_TAIL_SKIP` | the most aggressive precision classifier; recursive 32→64→128 WY coupling |
| **B** (2nd) | CUDA + cublasLt | **exact Householder, but the panel made physically fast**: register-file warp-per-column + **2-SM cluster TMA** | cublasLt per-member compute type: tf32 / **bf16x9** / fp16 + custom `build_t128_diag` T kernel | thin Python dispatch + in-C++ sweep | `register_2sm_panel_kernel` (`__cluster_dims__(2,1,1)`), `gmem_panel_kernel` (atomic multi-CTA cooperative panel) |
| **C** (3rd) | raw CUDA + **inlined CUTLASS/CuTe PTX headers** | CQR-BDGHK (same as A) + **hand-written warp-specialized tcgen05** GEMMs | per-member (`eh_set_trail_mode` / `ksafe` / `dual_mode`) + fp16 Schur | **PDL** + **explicit-node CUDA graph** + **in-grid co-dispatch** | `gramdc_kernel` (dual-consumer SYRK), `k_fused_diag_codisp` (chol ∥ lu ∥ Schur-tail in one grid), block pipeline |

A reports a geomean around 1.32 ms **in its own source comments**; all three land in the ~1.1–1.3 ms
band. The board's final figure for first place is **1,292 µs**, which is the number used everywhere
else in this repo and the one the 3.3× gap is computed from (4,247 / 1,292). I was at 4,247 µs.

---

## 2. What all three share (the common skeleton)

These eight are present in **all three** submissions. They are the common factor in "why they're fast".

1. **Nobody changed the algorithm family.** All three are shape-routed blocked compact-WY
   Householder QR emitting native flat (H, τ). No one swapped the decomposition or the output
   contract. My original read — "same algorithm" — was correct.

2. **All three attack the panel directly** — that 40–50%, latency-bound serial reflector chain. None
   of them tolerate it dominating. But **by different means** (see §3): A and C use Gram + Cholesky
   to turn the panel into GEMM work; B keeps exact Householder and makes the panel itself extremely
   fast and co-residable.

3. **Per-matrix precision routing** (not one precision for the whole batch). All three sample a few
   rows/columns → classify dense / band / rankdef / clustered / near-collinear → write a per-matrix
   flag → route that matrix to fp16 / tf32 / tf32x3 / bf16x9 / exact fp32.
   - A: `_s9_n512_n512_classify_precision_kernel`, `_safe512_classify_precision_kernel`
   - B: cublasLt `COMPUTE_32F_FAST_TF32` / `EMULATED_16BFX9` / `COMPUTE_32F` chosen per member
   - C: `eh_set_trail_mode` + `ksafe` safe-prefix splitting + `dual_mode`

   **This is how you get through the mixed@640 "precision floor": you don't break the wall, you go
   around it one matrix at a time.**

4. **fp16 / low precision in the trailing update, as a bandwidth lever.** The trailing update is
   bandwidth-bound; fp16 halves the bytes. A carries an entire fp16 working buffer (`Hh` +
   `larfb_qr_run_nested_fp16`); B has `fp16_baddbmm`; C has an fp16 Schur update. Safety comes from
   point 3.

5. **Fat blocked applies / recursive WY coupling** — turn the trailing update from many thin rank-32
   updates into large rank-64/128 GEMMs. A: recursive 32→64→128
   (`_s9_n1024_n1024_assemble_recursive_t64_kernel`). B: rank-OB compact-WY apply. C: NB=64 +
   block-T (`assemble_blockT_kernel`).

6. **Structural truncation** — for rankdef/clustered inputs, only factor the active prefix. The
   trailing n/4 (rankdef) or n/2 (clustered) columns of R are ≈0, so factor the prefix and zero or
   scale the tail with a coalesced write. A: `zero_tail` / `nearrank_tail`. (This is the general
   form of what my own V10-sub5 n≤64 routing did in one special case.)

7. **Launch-gap elimination.** None of them accept one Python dispatch per op. A: Triton fuses whole
   phases into very few kernels, plus `USE_TAIL_SKIP` to skip zero regions. B: the sweep loop lives
   inside C++. C: PDL + explicit-node graph (see §5.3).

8. **Fill idle SMs when the batch is small.** At n=2048/4096 with b≤8, ~140 SMs are idle. A: k-split
   + row-tile grids. B: multi-CTA cooperative panel. C: in-grid co-dispatch megakernels.

---

## 3. Where they differ — three philosophies for one bottleneck

### A (1st) — algorithm-first, pure Triton
- **Not one line of PTX, CUDA, or graph code.** All Triton kernels and `tl.dot` (autotuned tensor
  core, ~25% of peak — which turns out to be *enough*).
- The wins: **CholeskyQR reconstruction turns the panel into GEMM**, plus the most aggressive
  per-matrix precision classifier of the three (6+ structural classes; near-collinear and near-rank
  are each tested separately). Algorithm + precision takes most of the gap on its own.
- Triton's few-kernel fusion eliminates the launch gap for free — **launch-lean without needing a
  graph at all**.
- Highest value per line of code, and the easiest of the three to reason about. This is "a smart
  enough algorithm plus a good enough Triton GEMM", not "an extreme hand-written kernel".

### B (2nd) — panel-first, CUDA + cublasLt
- **Keeps exact Householder** and instead drives the panel to its physical limit:
  - `register_panel_kernel` / `register_2sm_panel_kernel`: **one warp owns a whole column**,
    reflectors live in registers, `mbarrier` + `elect.sync`, and a **2-SM cluster**
    (`__cluster_dims__(2,1,1)` + `cp.async.bulk` TMA + cross-CTA `mapa.shared::cluster` +
    `st.async.shared::cluster`). *This is exactly the MAGMA-style register panel plus 2-SM tile that
    my own notes named as "the thing we never built". They built it.*
  - `gmem_panel_kernel`: a **multi-CTA cooperative panel** using `st.release.gpu` / `ld.acquire`
    atomic flags — each CTA consumes reflectors published by earlier CTAs. For n=2048/4096.
- Trailing: compact-WY apply through cublasLt, choosing per member between **bf16x9**
  (`COMPUTE_32F_EMULATED_16BFX9` — nine bf16 terms emulating fp32), tf32, and fp16. The T factor
  uses a hand-written `build_t128_diag` register-block triangular inverse.
- The most "hand-written CUDA" of the three, yet it still uses cublasLt for the GEMMs. **It kills
  panel latency with registers and co-residence rather than by changing the math.**

### C (3rd) — engine-first, raw CUDA + inlined CUTLASS PTX
- **Inlines the whole CuTe/CUTLASS PTX-wrapper header set**: `ptx_tcgen05.cuh`, `mma_desc.cuh`,
  `ptx_tma.cuh`, `ptx_mbarrier.cuh`, `tensor_map.h`, `swizzle.h`.
- **A hand-written warp-specialized persistent tcgen05 engine** owning every GEMM: `gramdc_kernel`
  (**dual-consumer SYRK**: producer TMA warp + 2 consumer warpgroups + mbarrier ring + TMEM double
  buffering + `tcgen05.commit` — textbook FA4 structure), plus `swaptrail` / `usolve_tc` /
  `extinv_fused` for the CQR staircase GEMMs.
- **PDL** (`griddepcontrol` / `launch_pdl` / cudaLaunchAttribute id 6) + **explicit-node CUDA graph**
  (`cudaGraphAddKernelNode`) + **in-grid co-dispatch** (`k_panel_codisp`, `k_fused_diag_codisp`:
  panel(k+1) ∥ trailing-tail(k), or chol-diag ∥ lu-diag ∥ chol-Schur-tail, **partitioned by
  `blockIdx` inside a single grid**) + a **block pipeline** (left-looking LU maintaining a leading
  inverse; a fused-diag megakernel running chol(k+1) ∥ lu(k)).
- Same CQR-BDGHK algorithm as A, but every piece in raw CUDA on an owned tcgen05 engine. Several
  thousand lines, the most complete FA4-style pipeline of the three — **and it placed third.**

---

## 4. The ranking is itself the biggest lesson

**A (Triton; algorithm + precision) > B (register panel + 2-SM) > C (full tcgen05 engine). The
deepest engineering came last.** That is not a coincidence:

- **C's own source admits its owned tcgen05 GEMMs are mostly slower than cuBLAS.** A comment that
  recurs through the file:
  > `SUB-CUBLAS GRAPH-BLOCK (0.84-0.96x@n2048-b8): owned to enable the explicit-node graph (launch-gap), NOT a per-GEMM beat.`

  So C rewrote every GEMM in raw tcgen05 **not because it was faster — it is slower — but to make the
  whole CQR sequence capturable as an explicit-node graph.** The launch-gap win was worth shipping
  deliberately slower kernels to get it.
- Meanwhile **A got the same launch-leanness for free** from Triton's few-kernel fusion, without the
  several thousand lines and without the per-kernel sub-cuBLAS loss.
- **Conclusion: the value of eliminating the launch gap is ≥ the value of hand-writing tcgen05
  kernels — and there is a cheap path to it.** My working assumption all competition was that ~1500 µs
  required a multi-week hand-written engine. That was wrong. First place didn't write one.

---

## 5. What I got wrong (the part worth reading)

Each item: what they did, and how I talked myself out of it.

### 5.1 ★ CholeskyQR reconstruction — I filed it under DEAD_ENDS
**They did:** replace the panel with Gram → Cholesky → back out (V, τ), putting the panel on tensor
cores. It is the number-one lever in both A and C.

**Where I went wrong:** `DEAD_ENDS.md` said "CholeskyQR = LinAlgError-dead / conditioning routing is
forbidden / library primitives hit the same geqrf floor". I **treated a stability problem as a death
sentence** instead of as something to engineer around — the answer is per-matrix precision routing +
jitter + Newton refinement + an exact fp32 fallback for ill-conditioned members (§5.2). The *library*
CholeskyQR really does hit the floor. A **hand-written blocked tensor-core Cholesky** does not. An
earlier version of this very document argued at length that CholeskyQR was ruled out — that is the
single largest error in this repo's analysis history.

### 5.2 ★ Per-matrix precision routing — I treated tf32x3 as a hard floor
**They did:** classify each matrix by structure; safe ones get fp16 (2× bandwidth), ill-conditioned
ones stay exact.

**Where I went wrong:** I measured "mixed@640 has only ~2× margin at tf32x3, and 1×TF32 gets a
reseed-DQ" and concluded there was a **wall for the batch**. Correct observation, wrong conclusion:
it isn't that low precision is unusable, it's that **it can't be applied uniformly**. This one
unlocks both fp16 trailing and CholeskyQR's stability story.

### 5.3 ⚠★ "graph" was never banned — only "stream" was
**They did:** explicit-node CUDA graph (`cudaGraphAddKernelNode`, not capture) + PDL, killing the
40–48% launch-bound cost my own profiler had measured at small n.

**Where I went wrong:** my top hard constraint read "NEVER write `stream` **or** `graph`". But **C's
submission uses `cudaGraph*` throughout**, and states the rule explicitly:
> `Production path = 2 (clean explicit-node CUDA graph) ... NO capture API used (grep -ic on the banned token == 0)`

C uses "graph" freely and contains no "stream" (dependencies are expressed via the default queue and
PDL), and it passed the checker. That is a legitimate reading — the checker scans for a literal
substring. **The banned token is `stream` (plus cooperative launch). `graph` was my own addition**,
and it sealed off the single fattest small-n lever. **This is the most expensive self-inflicted wound
in the project** — it now has its own entry as [`DEAD_ENDS.md` §0](DEAD_ENDS.md).

### 5.4 ★ Overlap has to happen *inside one grid* (in-grid co-dispatch)
**They did:** put panel ∥ trailing, and chol ∥ lu, into a single grid partitioned by `blockIdx.x`,
filling otherwise-idle SMs.

**Where I went wrong:** my `design-B gate-0` tested **separate CTAs** (a reduce-CTA alongside a
GEMM-CTA), measured "co-residence is worse, overlap is dead", and stopped. The variant that works is
**`blockIdx` partitioning within one grid**. My own notes even contain the self-correction
("design-B gate-0 tested the WRONG variant") — I wrote it down and never went back to build the right
one.

### 5.5 I picked the wrong vehicle
**They did:** 1st place = **pure Triton**; 2nd/3rd = **raw CUDA + PTX wrappers**.

**Where I went wrong:** I invested heavily in **cute-DSL**. But (a) first place proves **plain Triton
is sufficient** — algorithm and precision are the dominant levers and `tl.dot` at ~25% of peak is
good enough; and (b) if you do want to go low-level, the winners used **raw CUDA with inlined CUTLASS
PTX headers**, not cute-DSL and not CUTLASS C++ templates. The cute-DSL work was not wrong
engineering, but it was **neither necessary (Triton suffices) nor the winners' low-level path**.

### 5.6 Methodology: I micro-benchmarked and shelved, so no lever ever compounded
**They did:** build the whole thing, and let CholeskyQR + fp16 + recursive WY + co-dispatch + graph
**compound**.

**Where I went wrong:** my loop was "microbenchmark each lever → gate on Modal → revert or shelve
anything under 5%". That loop **systematically under-values compounding** (five levers at <5% each is
not <5%, it can be 1.9×) and it **never pays the activation energy of a whole-engine rewrite**. I
accumulated a collection of "walls" — design-A ib=16, design-B, single-CTA trailing, smem panel —
every one of which is **an artifact of testing a piece in isolation**, and every one of which the
integrated design simply goes around. My own notes contain exactly this warning. I didn't follow it.

### 5.7 I spent my effort in the wrong place
- I spent it on **Modal-vs-official calibration and on reverting sub-5% deltas** (fp16x3 and
  implicit-V are both round trips of this kind).
- They spent the same effort on **algorithm (CholeskyQR) + precision routing + launch-gap
  elimination**.
- **Polishing crumbs on a frozen architecture is not the same as changing the architecture.** That is
  the top-level lesson.

---

## 6. Copy-this-next-time table (with real function names)

| Lever | Who | Real kernel / mechanism |
|---|---|---|
| CholeskyQR panel reformulation | A, C | `_safe512_cholesky_factor_32_launch` (A); CQR-BDGHK `k_chol_inv` + `k_lu_inv` + `k_RplusAeq` (C) |
| per-matrix precision classifier | A, B, C | `_safe512_classify_precision_kernel` (A); cublasLt compute type (B); `eh_set_ksafe` / `dual_mode` (C) |
| fp16 trailing (bandwidth) | A, B, C | fp16 working buffer `Hh` (A); `fp16_baddbmm` (B); fp16 Schur (C) |
| bf16x9 fp32 emulation | B | `CUBLAS_COMPUTE_32F_EMULATED_16BFX9` |
| mxfp8 two-term | A | `_v10_mxfp8_two_term_dot` |
| recursive WY 32→64→128 | A | `_s9_n1024_n1024_assemble_recursive_t64_kernel` |
| register warp-per-column panel | B | `register_panel_kernel` |
| **2-SM cluster panel** | B | `register_2sm_panel_kernel` (`__cluster_dims__` + TMA + `st.async.shared::cluster`) |
| multi-CTA cooperative panel | B | `gmem_panel_kernel` (`st.release.gpu` / `ld.acquire`) |
| **PDL (launch-gap, no stream)** | C | `launch_pdl` / `griddepcontrol` / cudaLaunchAttribute id 6 |
| **explicit-node CUDA graph** | C | `cudaGraphAddKernelNode` / `og_graph_build` / `cudaGraphLaunch` (**"graph" is allowed**) |
| **in-grid co-dispatch** | C | `k_panel_codisp` / `k_fused_diag_codisp` (blockIdx partitioning, one grid) |
| block pipeline chol ∥ lu | C | left-looking LU + leading inverse + `k_fused_diag` |
| warp-specialized persistent tcgen05 | C | `gramdc_kernel` (dual-consumer SYRK), inlined `ptx_tcgen05.cuh` |
| structural truncation | A, C | `zero_tail` / `nearrank_tail` (A) |
| owned block-triangular T | B | `build_t128_diag` / `build_t96_diag` |

---

## Appendix: reconciling this against my own result

- Me: **4,247 µs** (V10-sub5, Triton path, `milestones/11_V10sub5_fused32_official4247us_BEST.py`),
  about 3.3× behind first place.
- **What I had right:** same algorithm family; the panel is the bottleneck; the trailing GEMM's peak
  fraction is low; and my design-B experiment had tested the wrong variant (I knew this and wrote it
  down).
- **What I was missing:** all six items in §5. Of those, **5.1 (CholeskyQR), 5.2 (per-matrix
  precision) and 5.3 (graph was never banned)** are the three where knowing it at the time would have
  had the highest expected value — and 5.3 cost nothing to check.
