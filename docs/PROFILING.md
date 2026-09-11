# Profiling record — qr_v2 on Modal B200 (canonical, for future agents)

> **READ THIS for "how fast is each part and why".** It consolidates TWO independent
> profiling efforts that agree to <0.2% (so the numbers are real, not flukes):
> the `ncu-profiler-modal` substitute and the Codex `ncu-lite-deep-probe`. NVIDIA's own
> profilers (`ncu`/`nsys`) are **dead on Modal** (gVisor) — this is the best signal we can get
> without a bare-metal host. All numbers are the dominant **n=512, batch=640** case (4 of 12
> ranked cases). Deep gVisor evidence: `docs/PROFILING_ON_MODAL.md`. Raw dumps:
> `results/ncu_lite_deep_probe.txt` (Codex) + `results/ncu_substitute_analysis_2026-06-26.txt`.

## 0. What is / isn't measurable on Modal
- **`ncu` (Nsight Compute) = IMPOSSIBLE.** Modal runs under **gVisor** (`uname` = `4.19.0-gvisor`);
  the perfmon device interface (`/dev/nvidia-caps`, `/proc/driver/nvidia/capabilities`) is not
  proxied → `LibraryNotLoaded` on every metric set, version-independent (ncu 2025.1.1 & 2025.3.1).
  NOT a permission bug (`RmProfilingAdminOnly: 0`, container is root). **`nsys` is also dead**
  (GPU-tick→wallclock `ConvertGpuTicksToSyncNs InternalError`, no report). To get REAL occupancy /
  stall-reasons / bandwidth / roofline you need a **non-gVisor (bare-metal / full-VM) host**.
- **What DOES survive gVisor (the substitute):** (a) Triton **compile-time** resources
  (`n_regs`/`n_spills`/`shared`/analytic occupancy); (b) **SASS** instruction mix via `cuobjdump`
  on the cubin; (c) **kineto** (`torch.profiler`, in-process CUPTI — no global clock sync) for
  per-kernel durations + launch counts; (d) **timed micro-ablation** (build stripped probe
  kernels, time each) — a wall-clock stand-in for "where in the kernel does time go".

## 1. Re-runnable tools (all grep-clean; cost ~cents/run)
| tool | what it dumps |
|---|---|
| **`modal_ncu_lite_deep_probe.py`** (Codex; the fuller one) | kineto timeline + compiled resources + **panel component ablation** + **num_warps sweep** + **per-Householder-phase SASS probes** + GEMM SASS (HMMA). **Start here.** |
| `modal_kernel_analysis.py` (original substitute) | kineto timeline + compiled resources + multi-config **panel SASS** (4 reg-variants) + the gVisor methodology. |
| `modal_ncu_probe.py`, `modal_ncu_probe2b.py` | diagnostics that PROVE ncu/nsys are gVisor-blocked (env, /dev/nvidia*, ncu error). |
```
modal run modal_ncu_lite_deep_probe.py
modal run modal_kernel_analysis.py
```

## 2. Timeline breakdown (n=512 b=640, GPU self-time/iter, total ≈ 12252 µs)
Reproduced by both profilers to within 0.2% → trustworthy.
| category | µs/iter | % | kernels (launches/iter) |
|---|---|---|---|
| **PANEL** | ~5311 | **43.4%** | `_panel_kernel` ×8 |
| tf32x3 trailing | ~3366 | 27.5% | `_bmm_x3_kernel` ×14 + `_bmm_x3_sub_kernel` ×7 |
| solve (cuSOLVER trsm) | ~1794 | 14.6% | `batch_trsm_left_kernel` ×10 |
| **glue** | ~1674 | 13.6% | triu/tril ×14 (≈751) + **Memcpy DtoD ×2 (≈430)** + direct_copy ×14 (≈353) + where/fill/recip/arange (≈140) |
| cublas | ~98 | 0.8% | cutlass s1688gemm ×3 |
- Matches the older `panel 41 / gram 15 / trailing 26 / solve 18` profile (panel is ~43%, a touch
  higher). ~**55 kernel launches/iter** — many are tiny glue ops.

## 2b. Peak fraction — what 5.5% does and does not mean
The headline "we run n=512 at **5.5% of tf32x3 peak**" is a **whole-workload** figure. It is NOT the
efficiency of the trailing GEMM, and conflating the two (an earlier version of `README.md` did) makes
the GEMM look ~3x worse than it is and mis-aims the optimization.

Reference peak: `results/deepdive_roofline_raw.txt` uses **eff tf32x3 peak ~= 165 TF/s** (B200 dense
TF32 divided by the 3 passes tf32x3 costs).

**Whole workload** (exact, no attribution assumptions):
```
useful QR flops = (4/3) * n^3 * batch = (4/3) * 512^3 * 640 = 1.145e11 FLOP
wall time (§2 total)                                      = 12252 us
                                    => 9.35 TF/s = 5.7% of 165 TF/s   (~5.5%, the quoted number)
```

**Trailing GEMM kernels alone** (bracketed, because the flop attribution is not exact):
```
tf32x3 kernel time (§2: _bmm_x3 x14 + _bmm_x3_sub x7) = 3366 us
if those kernels carry 60% of the flops -> 20.4 TF/s = 12.4% of peak
if 75%                                  -> 25.5 TF/s = 15.5% of peak
if 90%                                  -> 30.6 TF/s = 18.6% of peak
```
The bracket is wide because (a) blocked compact-WY does **more** flops than the `(4/3)n^3` textbook
count -- the `T`-multiply and the within-super-panel applies are extra -- and (b) some of that extra
lands in the panel kernel, not the tf32x3 kernels. Round number to quote: **~15%**.

**The actionable reading.** The GEMM at ~15% of peak is mediocre; it is not the problem. The problem
is that 73% of the wall clock (panel 43% + solve 15% + glue 14%) does ~zero tensor-core work and is
**serialized** with the GEMM. Making the GEMM 2x faster moves the geomean by a few percent; *hiding*
the other 73% behind it is what takes the critical path to the GEMM alone. That is exactly the
overlap lever measured in §7.1 and (for us) walled in `docs/DEAD_ENDS.md`.

Re-derive with:
```bash
python3 -c "n,b=512,640; f=(4/3)*n**3*b; print(f/12252e-6/1e12, 'TF/s ->', 100*f/12252e-6/165e12, '%')"
```

## 3. ★ Panel component ablation (the headline NEW signal — Codex)
Subtractive probe kernels time each stage of `_panel_kernel` (B=640 n=512 ib=64 nw=8). The
`_panel_full_probe` was verified **bit-identical** to the production `_panel_kernel`
(255 regs / 18 spills, same SASS family counts) → the ablation is a faithful proxy.
| stage (cumulative) | µs | µs/col | % of full |
|---|---|---|---|
| mem_load_store (HBM only) | 51.6 | 0.81 | **5.3%** |
| + mask_predication (triangular masking) | 175.9 | 2.75 | 18.1% |
| + **reduce_norm_tau** (the reflector norm/τ reduction) | 493.0 | 7.70 | **50.7%** |
| + rank1_update | 650.5 | 10.16 | 66.9% |
| full_householder | 971.6 | 15.18 | 100% |

**Reads:** HBM is **only 5.3%** → the panel is NOT memory-bound. The **norm/τ reduction alone is
~51%**, and reduce+rank1 span essentially the whole kernel → the panel is **latency-bound by the
serial reflector-reduction chain** (SHFL-heavy), not FMA-throughput-bound, not addressing-bound.

## 4. ★ num_warps sweep (NEW — Codex). nw=8 is the timed optimum; do NOT retune.
| nw | µs | µs/col | note |
|---|---|---|---|
| 4 | 16082.8 | 251.3 | **16× cliff** — drops to 32 regs but **1100 spills** |
| **8** | **971.4** | **15.18** | **optimum** |
| 16 | 1085.7 | 16.96 | +11.8% |
| 32 | 1523.7 | 23.81 | +57% (64 regs / occ 0.5 but 16 spills) |

## 5. Compiled resources & occupancy (every kernel is REGISTER-limited, occ 0.125–0.25)
| kernel | warps | regs | spills | occ | note |
|---|---|---|---|---|---|
| `_panel_kernel` (shipped, tall) | 8 | **255** | **18** | 0.125 | the in-CTA **co-residence blocker** |
| `_panel_kernel` (other cfgs) | 4–8 | 141–179 | 0 | 0.125–0.188 | |
| `_bmm_x3_kernel` (trailing) | 4 | **108** | 0 | 0.25 | best cfg |
| `_bmm_x3_sub_kernel` | 4 | 128 | 0 | 0.25 | |
- All limiter = `regs`. Fine at n≤512 (640 CTAs over 148 SMs hide latency ACROSS CTAs), but it's
  exactly why the panel **can't co-reside** with a tcgen05 trailing worker on one SM: at 255
  regs/8 warps it already owns the 64K register file. The `gluon-regfile-panel` MAGMA rowmagma
  panel at **108 regs / 0 spills** is the structural fix (see `archive/docs/STAGE1_PROGRESS.md`).

## 6. SASS instruction mix
- **Panel** (`_panel_kernel` 255r): ~**63% addressing/predication+misc**, **warp-reduce (SHFL) 154
  ≈ 7.5%**, FP-math ~25–44% (taxonomy-dependent), **0 HMMA**. Signature of a reduction-serial,
  predication-heavy kernel. Per-phase probes: mask = 67% int/addr-pred; reduce = highest SHFL
  (13.7%); update = FP/FFMA-heavy (49%).
- **Trailing GEMMs** (NEW — Codex disassembled them): `_bmm_x3` & `_bmm_x3_sub` each issue **96
  HMMA** (tensor-core, confirms tf32x3 uses TCs) **AND are >50% int/addr-pred** → they're
  address-gen/predication-bound, not raw-HMMA-throughput-bound.

## 7. Actionable conclusions (what this means for optimizing qr_v2)
1. **Panel: don't chase memory or FMAs.** mem = 5.3%, FMA isn't the share — it's the **serial
   norm/τ reduction (51%)**. The only panel levers are (a) **hide it behind the trailing** (the
   design-A/B overlap) or (b) **restructure the reduction chain**. (This also explains why the
   reverted implicit-V — a traffic reduction — could never help the latency-bound parts.)
2. **Warps are settled: nw=8.** nw=4 is a 16× spill cliff; nw=16/32 are +12–57%. Don't retune.
3. **The 255r/18-spill panel ceiling is the hard in-CTA co-residence blocker** — consistent with
   the measured verdict "ship V5; in-CTA design-A is bounded by the ib=16 penalty"
   (`docs/DEAD_ENDS.md`). The 108r rowmagma panel is the structural unlock for design-A/B.
4. **glue = 13.6% is a real, separate deployable target** — but attack the **op/launch count**
   (triu/tril 751µs, Memcpy DtoD 430µs, ~14 tiny elementwise), **not** per-element ALU (implicit-V
   added mask-ALU and regressed officially; see DEAD_ENDS). Whether it nets out is HW-dependent →
   gate on a real gpumode submission.

## 8. Follow-ups — status
> This section used to list these as "NOT built". They were built the next day, on branch
> `profiling-deepdive`; raw output in `results/deepdive_*`, written up in
> `results/deepdive_findings.md`. Leaving the stale "NOT built" text here made this file — the
> canonical profiling record — contradict `README.md`. Fixed.

- **Roofline-via-shapes** — **BUILT** (`experiments/deepdive_roofline.py`, free, exact shape
  arithmetic; raw: `results/deepdive_roofline_raw.txt`). Establishes the B200 tf32x3 ridge at
  ~20.6 FLOP/byte and confirms the n=512 trailing/apply is **compute-side** — i.e. it would have
  predicted, in advance and for free, that implicit-V's traffic cut could not help there. It didn't
  exist when implicit-V was shipped; that is why implicit-V was shipped.
- **Launch-gap / GPU-idle** (= wall-clock − Σ kernel self-time) — **BUILT**
  (`modal_deepdive_launchgap.py`; raw: `results/deepdive_launchgap_raw.txt`). Result: small-n is
  **40–48% launch-bound**, n=512 only 1.9%. This is the single most important number in this file
  for the small-n half of the geomean, and it is what motivated V9 glue fusion and V10's fused
  one-shot kernel. **See also `docs/DEAD_ENDS.md` §0** — the other obvious answer to a 40–48%
  launch gap was ruled out by a constraint I had invented and never checked.
- **Profile the small-n cases (n=176/352)** — **STILL NOT DONE.** Only n=512 was ever profiled at
  kernel granularity; the small-n conclusions above come from the aggregate launch-gap measurement,
  not from a per-kernel timeline. This remains the least-explored territory in the repo.

## 9. Still genuinely un-measured
- **Real ncu** (occupancy / stall sampling / DRAM throughput / TC utilization): needs a non-gVisor
  GPU host (Lambda / RunPod bare-metal / local) — see §0. The competition eval box is likely locked
  down the same way.

## Provenance / why two outputs
Both profilers independently reproduced the timeline (panel 43.3 vs 43.4%) + every reg/spill/occ
row to <0.2% → the numbers are real. The original `modal_kernel_analysis.py` added the gVisor
methodology + multi-config panel SASS; the Codex `modal_ncu_lite_deep_probe.py` added the
ablation + warp-sweep + per-phase probes + GEMM HMMA (§3,4,6). The lone "disagreement" (SASS family
%s) is a taxonomy difference (FSEL/int-addr bucketing), not a real discrepancy — raw opcode counts
match. Keep both tools; this doc is the single canonical record.
