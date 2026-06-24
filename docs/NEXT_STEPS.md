# Next steps — the path forward

## Immediate (do this first)
**Submit `submission.py` (m5, route-n=2048) to confirm its official score** (~8100 est, from 8580). It's 22/22 on Modal and clean of "stream"/"graph". `grep -niE "stream|graph" submission.py` → must be empty.

## Where the cost is now (official, from the 8580 run; m5 cuts n=2048 to ~35ms Modal)
```
n=4096  52.2 ms   geqrf floor   ← biggest single, needs cooperative panel
n=2048  35-77 ms  (m5: ~35ms custom; m4: 77ms geqrf)
n=512   ~14.8 ms ×4  custom two-level + tf32x3
n=1024  ~11 ms ×3    custom + 1×TF32
small   <2.8 ms
```
The geqrf floor (n=2048+4096) caps geomean at ~2000µs even if all else were perfect. Leader is 1332µs ⇒ they do n=2048/4096 in single-digit ms.

## The remaining levers, ranked by EV

### A. (asked-for, in progress) Cooperative multi-CTA panel for n=4096 — HARD
n=4096 b=2 = only 2 one-CTA panels. To beat geqrf, split each matrix's panel across G CTAs (so total 2·G ≈ 128 CTAs fill the GPU). Blocked micro-panel ib=16-32, split [m×ib] tile across CTAs along m, reduce partial norms / partial VᵀC via a global scratch buffer + an **atomic-counter spin barrier** (one counter per matrix, last-arriver releases), TF32 trailing.
- **Risk:** ~2 grid-barriers/column ≈ 8000 spin-barriers for n=4096; Triton has NO proper grid-wide cooperative-sync primitive — atomic spin-barriers with correct acquire/release semantics are fragile (deadlock/incorrectness). Honest estimate ~20-30% it works AND beats geqrf this session. Targets 1 case, ~9% geomean.
- Start with a CORRECTNESS prototype on small forced-n, abort fast if the barrier mechanism is flaky.

### B. (highest-ceiling hypothesis) Full kernel fusion + GEMM-heavy narrow panels
This is how the leader is almost certainly at 1332µs. We are at **~1.5% of TF32 peak** on n=512; they're at ~15% (10× efficiency gap). The gap is **engineering (fusion + occupancy), not algorithm**:
- We orchestrate in PyTorch: ~250 launches/call, V/T round-trip HBM↔SRAM between panel and `_apply_block`.
- Leader: the whole QR per matrix (or per block) in 1-2 fused kernels, everything SRAM-resident, tensor-core trailing in-kernel, zero PyTorch dispatch. For n≤1024 that's one-CTA-per-matrix (640/60 CTAs = full occupancy).
- **The killer sub-idea:** make the PANEL tensor-core-bound too, via **CholeskyQR on the narrow panel** (Aᵀ_panel·A_panel small GEMM + tiny Cholesky + triangular solve → GEMM-heavy + high-occupancy; narrow panel ⇒ low condition number ⇒ CholeskyQR stable), then cheap Householder-reconstruction to emit flat (H,τ). This simultaneously fixes BOTH walls we hit: "panel is a reduction bottleneck" AND "large-n occupancy". **Recommended direction to try.**

### C. Squeeze n=512 (4 cases, ~14.8ms) — limited
Bottleneck is the tf32x3 trailing GEMM (mantissa floor — can't go lower). Options: better `_bmm_x3_kernel` autotune configs for the skinny M=32 shape; fuse panel+trailing to cut HBM round-trips (overlaps with B). Modal can't measure absolute tf32 speed reliably here.

## Validation discipline (because Modal lies on tf32 absolute speed)
- Use Modal `--stress` (full 22-case gate) for CORRECTNESS before any submission. **Always test mixed at batch=640**, not the batch-16 probe (1×TF32 failure only shows at 640).
- For SPEED of low-precision changes: Modal gives relative ranking only; confirm absolute via real submission.
- Panel/FP32/occupancy changes: Modal is reliable (and conservative — real gain ≥ Modal).
