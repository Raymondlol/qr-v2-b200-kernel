# FA4 → QR cute-DSL Engine — Build Blueprint (generated 2026-06-27 from Tri Dao FA cute source study)

# FA4 → QR cute-DSL Engine — Build Blueprint

> Companion to `docs/CUTEDSL_PORT_DESIGN.md`. That doc is the milestone/kill-criteria spine; **this doc is the concrete FA4-to-QR transfer map** — exact warp-role table, the overlap handshake wiring, register budgets with FA4's real numbers, and the M-plan reorder that follows from the 7-reader study of `flash_fwd_sm100.py` / `pipeline.py` / `tile_scheduler.py` / `blackwell_helpers.py`.
>
> **Load-bearing caveat carried through every section:** FA4's softmax is *parallel-over-rows*; our panel is *serial-over-columns*. That asymmetry is the single biggest risk to the overlap and is called out wherever it bites. The build order in §8 is chosen to **answer the serial-panel-hideability question first**, before sinking days into TMA rings and persistent schedulers.

---

## 1. The warp-role state machine for QR

FA4 splits 16 warps into load / mma / softmax(×2 groups) / correction / epilogue / scheduler. We collapse that to a **5-role** machine, because our "correction" (LARFT) is cheap and folds into the panel warpgroup, and we run **1 SM = 1 CTA = 1 matrix** (`cta_group=1`, no 2-CTA cluster). 8 warps is the sweet spot (one warpgroup of panel + supporting warps), but the table below is sized to a 16-warp CTA so each FA4 role maps 1:1 and you can drop unused warps to `empty`.

| Warp IDs | QR Role | Reg budget (setmaxregister) | What it does | FA4 template method |
|---|---|---|---|---|
| **0–3** | **PANEL-compute** (warpgroup) | **`increase(192)`** | FP32 serial Householder on `H[c0:n, c0:c0+IB]` (right-looking, per-column), warp-shuffle column-norm reductions, in-kernel **LARFT T16** recurrence. Produces V (reflectors, in-place in stril(H)), `tau`, and the FP32 `T` matrix into SMEM/TMEM for the trailing GEMM. | `softmax_loop` + `softmax_step` (`flash_fwd_sm100.py:1953–2363`); `SoftmaxSm100` state machine (`softmax.py:244–440`) |
| **4–7** | **PANEL-compute group 2** *(optional)* | **`increase(192)`** | Second panel warpgroup ONLY if look-ahead double-buffers two super-panels (panel[k+1] in group A while group B drains panel[k]'s LARFT). For the minimal slice, set these to `empty` (`decrease(48)`). | `softmax1_warp_ids` (second softmax group) |
| **8–11** | **CORRECTION / LARFT-apply** | **`decrease(88)`** | Consumes panel V+T, drives the **compact-WY apply** bookkeeping (W1=VᵀC staging, scale/rescale of the trailing C tile read back from TMEM), signals MMA that the apply operands are ready. In QR this is thin — it mostly does the TMEM read-back + FP32 fixups; in the minimal slice it can be folded into the panel warpgroup. | `correction_loop` + `correction_rescale` (`flash_fwd_sm100.py:2366–2576`) |
| **12** | **TRAILING-MMA issuer** | **`decrease(128)`** *(FA4 mma uses ~128; see §3)* | Issues `tcgen05.MmaTF32Op` K-loops into the TMEM accumulator for the 4 GEMMs: `gram VtV`, `W1=VᵀC`, `W2=TᵀW1`, `C -= V·W2`. tf32x3 = 3 limb passes with `use_acc` chaining. Leader-thread-only PTX dispatch. | `mma` loop (`flash_fwd_sm100.py:1656–1829`); `gemm_ptx_partial` (`blackwell_helpers.py:396–614`) |
| **13** | **EPILOGUE** | **`decrease(48)`** | Writes `R=striu(H)`, reflectors `=stril(H)`, `tau` → HBM after the last super-panel's apply is consumed. Producer-acquire/commit/tail structure. | `epilogue_warp_ids` branch (`flash_fwd_sm100.py:1220–1234`) |
| **14** | **LOAD / TMA producer** | **`decrease(40)`** | TMA-async streams `H[m]` tiles GMEM→SMEM ring (`PipelineTmaUmma` producer). Prefetches TMA descriptors **once** at kernel entry (`warp_idx==0` guard). | `load` loop (`flash_fwd_sm100.py:1361–1543`) |
| **15** | **EMPTY / (scheduler)** | **`decrease(48)`** | Idle, or runs CLC scheduler warp if you ever go dynamic. For QR: keep static, leave empty. | `empty_warp_ids` / `clc_scheduler_warp` |

**Register accounting (must sum ≤ 512 per the 4-warpgroup pool, FA4's invariant `flash_fwd_sm100.py:314–328`):**
- Minimal slice (panel + mma + load + epilogue, group 2 & correction = empty):
  `192 (panel) + 128 (mma) + 40 (load) + 48 (epi) = 408 ≤ 512` ✓ — room to lift panel toward 224 if ib>16 needs it.
- Full machine (both panel groups + correction):
  `2×192 + 88 + 128` overflows; FA4 resolves this because the **two softmax groups are time-staggered with the same physical budget via increase/decrease** — only one is "increased" at a time. For QR, prefer the **single panel-group** layout (rows 0–3) unless the serial dependency forces double-buffering.

**Dispatch skeleton** (port `flash_fwd_sm100.py:1138–1318`, ascending warp_idx, register call **inside** each branch):

```python
warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())
if warp_idx == 0:                                  # one-time TMA descriptor prefetch
    cpasync.prefetch_descriptor(tma_atom_H)        # exactly once per CTA
if warp_idx < 4:                                   # PANEL
    cute.arch.setmaxregister_increase(192)
    self.panel_loop(..., tile_scheduler)
elif warp_idx == 12:                               # TRAILING-MMA
    cute.arch.setmaxregister_decrease(128)
    self.mma_loop(..., tile_scheduler)
elif warp_idx == 13:                               # EPILOGUE
    cute.arch.setmaxregister_decrease(48)
    self.epilogue_loop(..., tile_scheduler)
elif warp_idx == 14:                               # LOAD/TMA
    cute.arch.setmaxregister_decrease(40)
    self.load_loop(..., tile_scheduler)
else:                                              # CORRECTION (8-11) or EMPTY
    cute.arch.setmaxregister_decrease(88 if 8 <= warp_idx < 12 else 48)
    self.correction_loop(...) if 8 <= warp_idx < 12 else None
```

> GOTCHA (readers [0],[6]): `setmaxregister_*` must be **inside** the branch, ascending order, and the panel `increase` must fire after the low-reg warps' `decrease` so the freed budget exists to claim. FA4 enforces this purely by ascending warp_idx ordering — copy that ordering exactly.

---

## 2. The overlap mechanism — trailing-MMA[k] ‖ panel[k+1]

This is the heart. FA4's `mma()` issues `P@V→O` for the **previous** S-stage while `softmax()` computes the **current** P — the GEMM hides the softmax latency. We transpose this to: **the MMA warp applies the trailing update for super-panel `k` (into the far columns) while the panel warpgroup factors super-panel `k+1`.**

### The dependency break (the precondition for any overlap)

The naive QR loop has a hard chain: `panel[k] → apply(k) → panel[k+1]` (k+1 reads columns that apply(k) just wrote). FA4 has no such chain because softmax rows are independent. **We manufacture independence with the look-ahead split** (port `m4_overlap.py` / `m2_narrowapply.py`):

```
panel[0]                                       # serial seed
for k in super_panels:
    apply_narrow(k → [k+IB : k+2·IB])          # update ONLY the next panel's columns; barrier
    # now panel[k+1]'s input columns are final → it can run independent of the FAR apply
    far = k+2·IB
    OVERLAP {
        MMA+CORRECTION:  apply_far(k, [far : N])   # tcgen05 into TMEM-acc, the BULK GEMM
        PANEL:           panel[k+1] + LARFT T16     # FP32 rowmagma, serial-in-columns
    }
epilogue
```

The **narrow apply** (one super-panel wide) is cheap and serial; the **far apply** (the bulk of the N−far columns) is the tcgen05 workload that masks the next panel. This is exactly FA4's split-P trick (`flash_fwd_sm100.py:2344–2359`) in spirit: do a small synchronous piece, then overlap the large piece.

### Pipeline objects (concrete, copied from FA4's set)

| Pipeline | Type | Producer → Consumer | Carries | FA4 analog |
|---|---|---|---|---|
| `pl_load_H` | `PipelineTmaUmma` | LOAD(14) → MMA(12) | H tiles GMEM→SMEM, num_stages=2–3 | `pipeline_Q` (`:920–928`) |
| `pl_panel_VT` | `PipelineUmmaAsync` | PANEL(0–3) → MMA(12) | V reflectors + `T` matrix ready for `far apply` | `pipeline_s_p_o` S→P side (`:961–968`) |
| `pl_apply_done` | `PipelineAsync` | MMA(12) → PANEL(0–3) | trailing-C far tile written back / TMEM slot free | `pipeline_p_lastsplit` (softmax→mma) |
| `pl_tau_stats` | `NamedBarrier` | PANEL → CORRECTION | per-column `tau`, norms (scalar, in `sScale` SMEM) | `sm_stats_barrier` (`:1006–1008`) |
| `pl_epi` | `PipelineAsync` | MMA/CORRECTION → EPILOGUE(13) | final R/v/tau tile ready to store | `pipeline_O_epi` |

`num_stages = 2` everywhere → binary phase XOR is safe (GOTCHA reader [5]: >2 stages needs mod-N phase). TMEM holds **two** accumulator slots so MMA can write the far-apply of step k while the panel reads/writes the narrow region of step k+1 without aliasing (reader [1] GOTCHA: S0/S1 are *different* TMEM offsets, not one slot with a flip bit — pre-allocate both via `tmem_*_offset[stage]`).

### Acquire / commit / wait / release order (the exact handshake)

Modeled on `mma()`↔`softmax()` (`flash_fwd_sm100.py:1705–1786` MMA side, `2285–2363` softmax side). One super-panel iteration:

```
PANEL warp (producer of V,T  /  consumer of apply_done):
  pl_apply_done.consumer_wait(stage^1)        # wait: far-apply of k-1 freed the TMEM/C slot
  ... factor panel[k+1] (FP32 serial columns) ...
  ... build LARFT T16 (FP32 recurrence) ...
  cute.arch.fence_view_async_tmem_store()     # MANDATORY before signal (readers [1],[5])
  pl_panel_VT.producer_commit(stage)          # V,T ready → wake MMA
  pl_apply_done.consumer_release(stage^1)
  # phase toggle handled by PipelineState.advance()

MMA warp (consumer of V,T  /  producer of apply_done):
  pl_panel_VT.consumer_wait(stage)            # wait V,T from panel of THIS k
  gemm_W1 = Vᵀ·C   (tcgen05 → TMEM acc[0])    # far columns
  gemm_W2 = Tᵀ·W1  (tcgen05 → TMEM acc[1])
  gemm_C -= V·W2   (tcgen05 → TMEM acc, accumulate)
  cute.arch.fence_view_async_tmem_store()
  pl_panel_VT.consumer_release(stage)
  pl_apply_done.producer_commit(stage)        # far-C done → unblock panel[k+2]
```

The **overlap window** is the gap between the panel's `producer_commit(stage)` and its next `consumer_wait` — during that window the panel is already grinding column-serial Householder on k+1 while the MMA chews the three tf32x3 GEMMs on k. The critical path becomes `max(panel[k+1]_latency, far_apply[k]_throughput)`.

> **Where the serial panel bites (THE risk):** FA4's softmax fully overlaps because *all 128 rows compute in parallel* — the softmax latency per stage is short and fixed. Our panel is `IB` serial columns, each a reflector that depends on the previous (`H_j → tau_j → update → H_{j+1}`). So `panel[k+1]_latency` is **IB × (reduction-chain depth)**, not hidden internally. The overlap only wins if `far_apply[k]` is *long enough* to cover one whole super-panel's serial chain. **Mitigations, in priority order:** (1) make IB small (IB=16 ⇒ shorter chain, the same value Gluon was forced to — but here it's a *latency* choice, not a *register* wall); (2) widen `far` so the GEMM is large (route only big-batch n=512 to the engine); (3) **fallback** (§6) if `panel_latency > far_apply` on a shape, drop to the serial narrow+far apply with no overlap (the `else` branch in the look-ahead, identical to V10's path) — never a correctness risk, just no speedup on that shape.

---

## 3. Per-warp register realloc plan

FA4's whole trick for 1-CTA/SM co-residence is **dynamic per-warp register caps** so the sum fits the 512-reg-per-warpgroup-quad pool, while each role gets what it needs. The numbers FA4 actually uses (readers [0],[3],[4],[6]):

| FA4 role | FA4 reg budget | QR role | QR budget | Rationale |
|---|---|---|---|---|
| softmax (×2 groups, 4 warps each) | **176–192** (`num_regs_softmax`) | **PANEL** | **192** (room to 224) | FP32 reductions + reflector temporaries + LARFT T accumulator are the reg-hungry path — mirrors softmax exactly. |
| correction (4 warps) | **64–88** (`num_regs_correction`) | **CORRECTION/LARFT-apply** | **88** | Reads stats + rescales a TMEM tile; medium. |
| mma (1 warp) | **128** effective (`num_regs_other` band; reader [4] cites ~96–128) | **TRAILING-MMA** | **128** | tcgen05 is register-*light* — the accumulator lives in **TMEM**, not registers (readers [4],[5]). The 128 is for SMEM descriptors + leader-election PTX, not accumulators. |
| load / epilogue / empty | **40–48** (`num_regs_other`) | **LOAD / EPI / EMPTY** | **40 / 48 / 48** | TMA + plain stores; minimal. |

**Why this co-resides at 1 CTA/SM where Gluon's design-A could not:**
The Gluon wall was a *static* per-CTA register budget — the panel and the MMA worker each demanded their full budget *simultaneously*, forcing ib=16 (panel at 108 regs) to fit a 128-reg worker, and ib=32 (160 regs) overflowed → 0.938× regression. FA4's `setmaxregister_decrease`/`increase` makes the budget **dynamic and time-multiplexed**: the MMA warp *decreases* to 128 freeing ~64+ regs back to the pool, which the panel warp then *increases* to claim (192). The hardware records per-warp caps at the call instruction (reader [0] GOTCHA), so the SASS encodes distinct per-thread budgets and the warp scheduler interleaves them. **This is the specific FA4 mechanism that the port buys us** and is the entire reason cute-dsl is worth the multi-day spend over re-fighting the Gluon wall.

> **Three hard caveats (don't over-promise):**
> 1. **TMEM-acc relieves the *accumulator* pressure, not the descriptor/load-store machinery** (reader [5] explicitly). So dynamic realloc lets panel+mma co-reside in *one* resident CTA, but it does **not by itself restore 2–3 CTAs/SM**. Cross-matrix occupancy masking is the persistent queue's job (§4), separate lever.
> 2. **`setmaxregister` is a hint; over-budget → silent spill to local memory** (readers [1],[2] GOTCHAs). Every budget choice must be validated by SASS `n_regs`/`n_spills` (= 0) via `modal_kernel_analysis.py`, not assumed.
> 3. **`setmaxregister` is per-warp, not per-thread** (reader [0]) — the warp's max-thread reg demand sets the cap. Keep panel threads uniform (rowmagma row-distribution gives this).

---

## 4. Persistent scheduler — one CTA loops the matrix batch

FA4: `grid = min(num_SMs, total_tiles)`, each CTA pulls work via `StaticPersistentTileScheduler`, never one-CTA-per-tile. For QR, **one work-tile = one matrix**; the tile coord degenerates to `(matrix_id,)`.

**Grid sizing** (port `tile_scheduler.py:228–245 get_grid_shape`):
```python
grid = min(num_SMs=148, batch_B)          # 148 persistent CTAs at b640; advance grid-stride
```
At b640 each of 148 CTAs processes ~4.3 matrices serially over the persistent loop. The advance is **grid-stride** (`advance_to_next_work()` increments by `grid_dim`, reader [6] GOTCHA), NOT by 1.

**Work-loop skeleton** — identical in every role method (port `flash_fwd_sm100.py:1953–1954`, `tile_scheduler.py:287–391`):
```python
work_tile = tile_scheduler.initial_work_tile_info()
while work_tile.is_valid_tile:
    m = work_tile.tile_idx[0]                 # this CTA's matrix index
    # --- role body for matrix m (panel / mma / load / epilogue) ---
    work_tile = tile_scheduler.advance_to_next_work()
```

**Why this is THE load-bearing increment (not the warp_specialize overlap):**
The 0.87× prize requires **panel latency masked**, and there are two ways to mask it: (a) in-CTA overlap (§2) hides panel[k+1] behind far_apply[k] *within one matrix* — but bounded by the serial chain; (b) **cross-matrix** — while CTA's matrix-`m` panel stalls on its reduction chain, the *same SM's other resident CTA* (matrix `m'`) runs its trailing GEMM. Mechanism (b) needs **2–3 CTAs/SM**, which only the persistent queue delivers (by NOT static-register-sharing all roles into one fat resident CTA). FA4 gets dense occupancy because the persistent scheduler + dynamic realloc keep per-CTA reg footprint low enough to co-reside. **This is the design-B "per-matrix work-queue" the prior session wanted but tested wrong** (it split panel/trailing across *separate* CTAs and measured contention; the correct structure is whole-matrix-per-CTA with in-CTA roles, exactly FA4).

> GOTCHAs: `is_valid_tile` is the *only* loop exit — exiting early hangs peers waiting on pipelines (reader [0]). Excess CTAs (grid>batch) must tolerate `is_valid_tile=False` immediately (reader [3]). Static scheduling only — **avoid `Clc*` classes**: their "cluster-launch-control" docstrings can leak the banned `stream`/`graph` substrings into the submission (CUTEDSL_PORT_DESIGN §5).

---

## 5. tcgen05 trailing GEMM into TMEM (tf32x3)

Port `blackwell_helpers.py` + the FA4 mma setup. Four GEMMs per super-panel: `gram VtV`, `W1=VᵀC`, `W2=TᵀW1`, `C-=V·W2`. All tf32x3 (3 limb passes), all into TMEM accumulators.

**(a) Build the tiled MMA** (port `flash_fwd_sm100.py:402–417`):
```python
tiled_mma = sm100_utils.make_trivial_tiled_mma(
    a_dtype=Float32,                       # tf32x3: operands split into hi/mid/lo TF32 limbs
    a_major=tcgen05.OperandMajorMode.K,
    b_major=tcgen05.OperandMajorMode.K,
    acc_dtype=Float32,                     # accumulate FP32 (the tf32x3 floor, §Precision)
    cta_group=1,                           # single-CTA QR, NOT 2-CTA
    tile_shape_mn=(128, 128),
)
mma_kind = sm100_utils._tcgen05_mma_kind(tiled_mma.op)   # compute ONCE, reuse (reader [0] GOTCHA)
```

**(b) Allocate TMEM** (port `flash_fwd_sm100.py:678–687`, `is_two_cta=False`):
```python
tmem = TmemAllocator(holding_buf_ptr, barrier_for_retrieve=tmem_alloc_barrier,
                     allocator_warp_id=12, is_two_cta=False)     # MMA warp allocates
# MMA(12): tmem.allocate(cols);  tmem.wait_for_alloc();  acc_ptr = tmem.retrieve_ptr(Float32)
# PANEL/CORRECTION: tmem.wait_for_alloc();  acc_ptr = tmem.retrieve_ptr(Float32)  # BEFORE any TMEM read
```
Two accumulator offsets (`tmem_acc_offset[0]`, `[1]`) for the W1/W2 ping-pong + the C-update. NBP≥64 is the hardware wall — the [NBP, BW] tile pads to 64 rows minimum (the known ~2.0× serial-tcgen05 floor at n≤512; padding waste is real but the overlap is what pays it back).

**(c) tf32x3 K-loop into TMEM** (port `gemm_ptx_partial`, `blackwell_helpers.py:396–614`). The limb split is the tf32x3 mechanism: each FP32 operand becomes 3 TF32 limbs; `tCrA.shape[2] == 3`, `tCrB.shape[2] == 3`, and the function precomputes all 3 K-offsets at compile time:
```python
blackwell_helpers.gemm_ptx_partial(
    tiled_mma.op,
    acc_tmem_addr = cute.arch.make_warp_uniform(acc_ptr + tmem_acc_offset[stage]),  # MUST be warp-uniform
    tCrA = V_limbs,            # [.,.,3]  tf32x3 limbs of V
    tCrB = C_limbs,            # [.,.,3]
    sA = None, sB = smem_C,    # K-major operand from SMEM ring
    zero_init = (k == 0),      # ACCUMULATE flag via pred; first K-iter zeroes acc
    cta_group = 1,
)
# leader-thread election (elect.sync) is INSIDE gemm_ptx_partial — only lane-leader issues the MMA
```
For the 3-pass: issue with `use_acc` chaining (hi·hi → +hi·lo+lo·hi → keep), accumulating across limb passes into the same TMEM tile. **Drop the lo·lo cross-term only on the non-dominant Gram** if a checkpoint shows the 3rd pass binds AND mixed@640 margin stays ≥1.83 — never on the final `C-=V·W2`.

**(d) Read back from TMEM** (port `flash_fwd_sm100.py:1545–1557`, used by CORRECTION/PANEL):
```python
ld_atom = cute.make_copy_atom(tcgen05.Ld32x32bOp(tcgen05.Repetition(32)), Float32)
tiled_ld = tcgen05.make_tmem_copy(ld_atom, acc_slice)
thr_ld   = tiled_ld.get_slice(tidx)
cute.copy(tiled_ld, thr_ld.partition_S(acc_slice), reg_frag)
cute.arch.fence_view_async_tmem_load()     # MANDATORY before consuming reg_frag
```
Final output read-back in **FP32** (never read the tf32 acc directly as output — rounds to 19 bits, reader [4] GOTCHA). Store atom `St32x32bOp(Repetition(8 or 16))` for writing the corrected C tile back.

> GOTCHAs (reader [4]): TMEM addr **warp-uniform** via `make_warp_uniform`; descriptor swizzle/layout wrong → silent data corruption; K-loop must be `const_expr` (n=512 ⇒ K=256 ⇒ unroll const ✓); inline-asm constraint string order must align with `$N` args or runtime garbage.

---

## 6. The panel-compute warp (serial, the hard part)

Modeled on `softmax.py::SoftmaxSm100` (`:244–440`) consume-S/produce-P structure, but with the column-serial asymmetry made explicit.

**FA4 softmax pipeline (parallel, the template):**
`consumer_wait(S)` → `Ld32x32bOp` load S from TMEM → `compute_row_max_local` → `update_row_max` → `scale_subtract_rowmax` (FMA) → `apply_exp2` → `update_row_sum` → `St32x32bOp` store P to TMEM → `fence` → `consumer_release` → phase XOR. **All 128 rows in parallel.**

**QR panel pipeline (serial-over-columns, the port):**
```
pl_apply_done.consumer_wait(stage^1)                 # the input columns for this panel are final
acc_ptr load: H[c0:n, c0:c0+IB] from TMEM/SMEM       # tcgen05.Ld32x32bOp, same atom as softmax
for j in range(IB):                                  # ← SERIAL. each col depends on prior.
    norm_x   = warp_reduce(sum(H[j+1:,j]**2), op=+, width=32)   # utils.py:318 warp_reduce
    beta     = -sign(alpha)*sqrt(alpha**2 + norm_x)             # FP32, alpha=H[j,j]
    tau_j    = (beta - alpha)/beta
    H[j+1:,j] /= (alpha - beta);  H[j,j] = beta                 # rowmagma in-place (108-reg panel)
    # within-super-panel rank-1 update of cols (j+1 : c0+IB):
    w        = warp_reduce(Vᵀ @ Sub, op=+)                      # V=[1;H[j+1:,j]]
    Sub     -= tau_j * outer(V, w)
    sScale[j] = tau_j                                            # SMEM scalar handoff (like FA's sScale)
    pl_tau_stats.arrive_w_index(stage*IB + j)                   # NamedBarrier per-column signal
# LARFT T16 (FP32 recurrence) — the one piece with no FA4 analog:
G = Vᵀ@V   (tiled BM=64 K-loop, tf32x3 dots)
T[0,0]=tau0
for i in 1..IB-1:  z=G[0:i,i];  T[:i,i] = -tau_i*(T[:i,:i]@z);  T[i,i]=tau_i
fence_view_async_tmem_store()
pl_panel_VT.producer_commit(stage)                   # V,T ready → wake MMA far-apply
```

**Reductions** use FA4's `warp_reduce` (`utils.py:318–334`, `shuffle_sync_bfly` butterfly over log2(width) stages) — each thread owns a row-slice (rowmagma row-distribution), reduces locally, warp-shuffles to the column-norm. **Cross-warp** reductions (panel taller than 32 rows per warp) need per-warp `warp_reduce` then an SMEM tree (reader [6] GOTCHA: `shuffle_sync_bfly` does NOT cross warp boundaries).

### The serial-vs-parallel asymmetry — where hiding gets hard, and the fallback

| | FA4 softmax | QR panel |
|---|---|---|
| Inner structure | 128 rows **parallel**; latency = one short reduction tree | IB columns **serial**; latency = IB × (reflector + norm-reduce + rank-1-update) chain |
| Per-stage cost | short, fixed | IB-proportional, *cannot* shrink without raising IB count |
| Hideable by? | always (short vs long PV-GEMM) | only if `far_apply[k]` ≥ IB-chain length |

**Consequences for the build:**
1. **IB is now a latency knob, not just a register knob.** Small IB (16) → short serial chain per super-panel → easier to hide, but more super-panels (more LARFT, more barriers). This is a *different reason* to land on ib=16 than Gluon's register wall — and here it might actually be fine because dynamic realloc removed the register reason, so we can pick IB purely for latency balance.
2. **Cross-matrix masking (§4) is the real escape** from the asymmetry: even if one matrix's panel chain can't be hidden behind *its own* far-apply, a *second resident CTA's* GEMM hides it. This is why M6 (persistent occupancy), not M4 (in-CTA overlap), is the go/no-go.
3. **Fallback (always available, never a correctness risk):** the look-ahead `else` branch (§2) is the plain serial `narrow-apply; far-apply; panel[k+1]` — identical to V10's fused path. If a shape shows `panel_latency > far_apply` (overlap loses), route that shape to the serial branch or to V9 entirely. The overlap is strictly opt-in per shape, gated on a whole-kernel measurement.

---

## 7. Concrete reuse list — exact FA4 files/functions to copy or adapt

**Copy near-verbatim (machinery, problem-agnostic):**
- `pipeline.py` — `PipelineStateSimple` + `make_pipeline_state` (`:38–96`), `_PipelineIndexPhaseMixin` (`:118–157`), `PipelineAsync` + `elect_one_commit`/`syncwarp_before_commit` (`:195–259`), `NamedBarrier` + `arrive_w_index`/`arrive_and_wait_w_index` (`:162–189`). **The whole file ports unchanged** — it's CUTLASS pipeline wrappers.
- `tile_scheduler.py` — `StaticPersistentTileScheduler` (`:287–391`), `WorkTileInfo` (`:94–102`), `get_grid_shape` (`:228–245`). Degenerate the 4-tuple tile coord to `(matrix_id,)`. **Skip `ClcState`/`Clc*`** (`:40–92`) — banned-substring risk + unneeded for static.
- `named_barrier.py` — the `NamedBarrierFwdSm100` enum pattern; rename to `NamedBarrierQrSm100` with QR roles (`PanelColDone`, `VTReady`, `TmemPtr`, `ApplyDone`). Keep barrier count < 12 (reader [6]).

**Closely adapt (Blackwell GEMM primitives):**
- `blackwell_helpers.py` — `make_trivial_tiled_mma` (`:13–30`), `gemm_ptx_partial` (`:396–614`, the tf32x3 K-loop into TMEM), `i64_to_i32x2` (`:110–112`), `make_smem_layout_a/b` swizzle/descriptor construction (`:428–447`). This is the §5 trailing GEMM.
- `flash_fwd_sm100.py` structural skeleton: warp dispatch + setmaxregister ladder (`:1138–1318`), `TmemAllocator` create/allocate/relinquish flow (`:875–891, 1187–1214`), the `mma()` overlap loop nesting (`:1656–1829` — the issue-prev-while-compute-current schedule, reader [1] GOTCHA "nesting order matters"), TMEM copy atoms `Ld32x32bOp`/`St32x32bOp` (`:1923–1943`).
- `softmax.py` — `SoftmaxSm100` consume/produce structure (`:244–440`) as the **panel warp template** (the §6 loop replaces the elementwise body with serial Householder).
- `utils.py` — `warp_reduce`/`warp_reduction_sum`/`warp_reduction_max` (`:318–334, 845–864`) for the FP32 column reductions; `make_warp_uniform` for every warp-role branch.

**Already in our tree to fuse in (the QR-specific bodies):**
- `experiments/stage0_regfile_panel.py::_panel_rowmagma` — 108-reg/0-spill FP32 panel → the §6 inner body.
- `experiments/stage1a_tf32x3_async.py` — tf32x3 3-pass into TMEM reference (validates the §5 limb chaining).
- `experiments/stage1c2_2gemm_apply.py` — compact-WY 2-GEMM apply.
- `cand_fused.py` / `fused_qr_slice.py` (persistent-engine branch) — V10 LARFT T16 (the §6 recurrence, the one piece with no FA4 analog) + the whole-kernel perf baseline every Mi must beat.
- `experiments/m4_overlap.py` / `m2_narrowapply.py` (engine-moonshot branch) — the look-ahead k-loop + `_narrow_apply` dependency-break (§2).
- `experiments/tri_grid_barrier.py` — atomic-counter fallback if `StaticPersistentTileScheduler` is awkward to wire (proven grep-clean).

---

## 8. Revised M-plan delta vs `docs/CUTEDSL_PORT_DESIGN.md`

The existing M-plan frames M6/K4 as occupancy-driven ("does the persistent queue restore 2–3 CTAs/SM"). The FA4 study refines this into a **pipeline-depth framing** and — critically — **reorders the build to answer the serial-panel-hideability question before the expensive infrastructure**, because that question (not occupancy) is what FA4 makes us realize is the true risk.

### Reframe 1 — occupancy → pipeline depth
The old K4 ("1 CTA/SM pins, panel exposed at 13294µs") treats occupancy as a binary recovery. Replace with a **pipeline-depth metric**:
- **Depth-1 (in-CTA overlap, §2):** panel[k+1] hidden behind far_apply[k] *within one matrix*. Bounded by `far_apply ≥ IB-serial-chain`. This is the FA4 `pipeline_s_p_o` mechanism. **Measurable at M4, no persistent queue needed.**
- **Depth-2 (cross-matrix, §4):** a second resident CTA's GEMM hides matrix-m's panel. Needs the persistent queue + dynamic-realloc reg footprint low enough for 2 CTAs/SM. **Measurable at M6.**

The new go/no-go is: **does Depth-1 alone get within striking distance, or do we strictly need Depth-2?** FA4 says Depth-1 *can* work (softmax is hidden depth-1) — but our serial panel may force Depth-2. We must find out cheaply.

### Reframe 2 — build the minimal overlap slice EARLY
The old order builds M1(serial)→M2(tcgen05)→M3(TMA)→M4(overlap)→M5(reg-lean)→M6(persistent). That sinks 3 milestones of infrastructure (tcgen05 TMEM, TMA ring) *before* the one risky question — can a serial panel be hidden at all — gets answered. **Reorder so the load-warp ‖ mma-warp ‖ panel-warp overlap triangle is built and measured by M2-equivalent**, using the *cheapest possible* GEMM (even plain cute MMA, no tcgen05 yet), to get a yes/no on hideability before investing in tcgen05/TMA.

### Revised milestone table

| M | Old framing | **New framing (FA4-informed)** | Decisive number |
|---|---|---|---|
| **M0** | Tooling + probe | unchanged: cute-dsl B200 image, probe submitted, 10-line TMEM+mbarrier snippet, grep-clean `cutlass.pipeline` imports | snippet runs |
| **M1** | Thinnest serial | unchanged: 1-CTA serial correctness floor (`fused_qr_slice` twin) | relerr<1e-4; whole <1.5× V10 (K2) |
| **M2 ⇄ NEW** | ~~tcgen05+TMEM~~ | **★ MINIMAL OVERLAP TRIANGLE (moved up).** load-warp ‖ mma-warp ‖ panel-warp in ONE CTA via `setmaxregister` realloc + `pl_panel_VT`/`pl_apply_done` handshake — but with a **cheap GEMM** (plain cute MMA, not tcgen05 yet) and direct SMEM loads (no TMA ring yet). Look-ahead `_narrow_apply` dependency-break. **This answers serial-panel-hideability (Depth-1) at minimum cost.** | **overlap eff (ov=1 vs ov=0)**; is panel[k+1] masked behind far_apply[k]? bit-identical to M1 serial. If Depth-1 dead here → the in-CTA prize is gone, jump to Depth-2 (persistent) diagnostic or STOP early — **saved 2 milestones of wasted tcgen05/TMA work.** |
| **M3** | TMA ring | **tcgen05 + TMEM-acc** (the old M2): swap the cheap GEMM for `tcgen05.MmaTF32Op` into `TmemAllocator`, tf32x3 3-pass. Only worth doing once M2 shows overlap has a pulse. | mixed@640 ≥1.83 (K5); apply-alone µs; ~2.0× serial floor |
| **M4** | overlap | **TMA producer ring** (`PipelineTmaUmma`, old M3) + reg-lean tune (old M5 folded in): SASS regs=0 spills, push IB toward latency-optimal | whole vs M3; regs/spills |
| **M5/M6** | reg-lean / persistent | **★ Depth-2 persistent queue** (the old M6, still THE load-bearing step): `StaticPersistentTileScheduler`, 148 CTAs, cross-matrix masking. **Now reached with overlap *already proven* at M2**, so this isolates the pure occupancy contribution. | **2–3 CTAs/SM? panel masked cross-matrix? whole vs V10 — first real 0.87× shot (K4)** |
| **M7/M8** | routing / 2-SM | unchanged: 12-shape routing geomean; optional `cta_group=2` for n=1024 | geomean of 12 vs V10 |

### The one-line delta
**Old:** infrastructure-first (tcgen05→TMA→overlap→persistent), risk-question answered late at M6.
**New:** **overlap-triangle-first** — build load‖mma‖panel with a throwaway-cheap GEMM at M2 to answer *"can a serial panel be hidden?"* (Depth-1) before spending days on tcgen05/TMA; keep M6 persistent (Depth-2 cross-matrix occupancy) as the load-bearing go/no-go, but reach it knowing overlap already has a pulse. The kill-criteria gain a cheap early exit: **if the minimal triangle shows the serial panel can't be hidden Depth-1 AND occupancy can't be coaxed to Depth-2, ship V10 having spent 2 milestones, not 6.**

---

*All FA4 line refs are from the 7-reader study of `flash_fwd_sm100.py`, `sm100_hd256_2cta_fmha_forward.py`, `flash_bwd_mla_sm100.py`, `pipeline.py`, `tile_scheduler.py`, `blackwell_helpers.py`, `softmax.py`, `utils.py`, `named_barrier.py`. Validate every reg budget via SASS `n_regs`/`n_spills`, every overlap via whole-kernel n=512 b640 vs V10, every precision touch via a real gpumode submission (never Modal alone).*

---

(Originally produced as a research note; persisted here as `docs/FA4_BLUEPRINT_FOR_QR.md`.)

**Key load-bearing decisions in this blueprint (so you can challenge them):**
1. **5-role machine, single panel-warpgroup** (not FA4's 6-role dual-softmax) — QR's LARFT is too cheap to justify a separate correction warpgroup in the minimal slice.
2. **The reorder (§8) is the highest-leverage change** — FA4 makes clear the real risk is *serial-panel hideability*, not occupancy, so I moved the load‖mma‖panel overlap triangle to M2 with a throwaway GEMM to answer it before the tcgen05/TMA spend. This contradicts the existing doc's "M6 is the go/no-go" framing by adding an *earlier* cheap kill-gate.
3. **Two depth levels** (in-CTA Depth-1 vs cross-matrix Depth-2) — the existing doc conflates these; separating them is what lets M2 be diagnostic.