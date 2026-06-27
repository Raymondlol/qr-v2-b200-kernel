# M3c Build Guide (generated 2026-06-27 from 6-agent design workflow)

# M3c Build Guide — tcgen05 TF32x3 BLOCKED Compact-WY Apply in cute-DSL

> **What M3c is.** Take M2b's validated per-matrix overlap CTA (panel warp ‖ apply warp, CTA-barrier
> delimited, bit-identical to geqrf) and **replace the warp-reduce far-apply with a tcgen05 tf32x3
> BLOCKED compact-WY apply** issued by one MMA warp into TMEM accumulators. The panel warp now also
> computes the LARFT T-factor. Math: `C -= V @ (T^T @ (V^T @ C))`, 2 GEMMs (W1=V^T·C, W2=T^T·W1, then
> C-=V·W2), tf32x3 3-limb into FP32 TMEM acc.
>
> **Honest framing up front.** M3c is single-CTA-per-matrix. At n=512 b640 this pins ~1 CTA/SM (TMEM
> ~250 cols + panel regs), so the serial panel is **NOT masked across matrices** — that's M6's job
> (persistent queue). The in-CTA `far_apply ‖ panel[k+1]` overlap can only hide panel latency behind a
> *shorter* tcgen05 shadow than the warp-reduce one was. **Realistic landing: ~1.0–1.2× V10 (NOT a
> 0.87× breakthrough).** The decision M3c answers: *does the tcgen05 apply at least not regress, and does
> the look-ahead shadow recover the panel?* Build it to learn that, not expecting a win at M3c alone.
>
> **Reuse-by-reference (do not rewrite):**
> - `experiments/cute_qr_m2b.py` — `_warp_reduce_add` (L26), `_panel_factor` (L63), `_apply_reflector`
>   (L33), `_apply_range` (L49), the CTA-barrier overlap loop (L115-142). **M3c-0 = m2b with the
>   far-apply body swapped.**
> - `experiments/cute_gemm_tf32x3.py` — the *entire* tcgen05 tf32x3 machinery: SharedStorage+mbar
>   (L30-34), `TmemAllocator` (L55-56,99), `MmaTF32Op`+`make_tiled_mma` (L148-150),
>   `make_smem_layout_a/b` (L151-152), `make_fragment_A/B/C` (L85-88), the **3-pass tf32x3 with
>   `tiled_mma.set(ACCUMULATE,True)`** (L128-131), TMEM read-back `Ld32x32bOp`+`make_tmem_copy`
>   (L106-108,137-140).
> - `experiments/cute_qr_m2a.py` — per-warp `setmaxregister_increase/decrease` (L108-111).
> - `experiments/cute_qr_m1b.py` — blocked QR convention (unit-lower V in stril(H), super-panel loop).
> - `experiments/fused_qr_slice.py` (dreamy-johnson worktree) — **the EXACT LARFT recurrence to port**
>   (L81-91) and the 2-GEMM apply math (L93-130) as the bit-for-bit Triton reference.
> - `experiments/stage0_regfile_panel.py` — 108-reg rowmagma panel (only if m2b's `_panel_factor`
>   register footprint blocks co-residence; default = reuse m2b's panel directly).

---

## 1. Architecture

**One CTA = one matrix.** Grid = `(B,1,1)`, block = `(64,1,1)` (2 warps) for M3c-0, extensible to
`(128,1,1)` (4 warps) if panel needs a warpgroup. Each CTA claims its matrix via `block_idx()[0]`,
exactly as m2b L97.

### Warp roles (M3c-0, 2 warps = 64 threads)

| Warp | Role | Regs | Reused from |
|------|------|------|-------------|
| **0 — PANEL** | FP32 Householder panel-factor + narrow-apply + **gram G=V^T·V (warp_reduce) + LARFT T → SMEM** | `setmaxregister_increase(192)` | m2b `_panel_factor` L63 + new `_compute_gram`/`_larft` |
| **1 — MMA** | tcgen05 tf32x3 blocked far-apply: W1=V^T·C, W2=T^T·W1, C-=V·W2 into TMEM acc, RMW C to gmem | `setmaxregister_decrease(128)` | gemm_tf32x3 3-pass L128-131 + new C-tile loop |

> Why panel computes G/T (not the MMA warp): the gram is `IB²=256` warp-reduces — cheap, serial,
> register-local, and reuses the *proven* `_warp_reduce_add`. The MMA warp stays lean (it only issues
> tcgen05 + does TMEM read-back). This keeps the MMA warp under 128 regs.

### Data flow

```
 gmem H[B,n,n] ──(panel warp factors in place, stril(H)=V reflectors, diag(H)=R, tau→gmem)──┐
                                                                                            │
 PANEL warp:  G = V^T·V  ──warp_reduce──► sG[IB,IB] (SMEM, FP32)                            │
              LARFT recurrence ──────────► sT[IB,IB] (SMEM, FP32, upper-tri)                │
                          │ CTA barrier (sT visible)                                         │
                          ▼                                                                  │
 MMA warp:  load V from stril(H)/gmem ──► SMEM sV (split hi/lo in-reg OR host) ◄────────────┘
            load C[trailing] from gmem ──► SMEM sC (split hi/lo)
            tcgen05 W1 = V^T·C   ─3-pass tf32x3─► TMEM acc[0]
            tcgen05 W2 = T^T·W1  ─3-pass tf32x3─► TMEM acc[1]   (T^T read from sT[j,i])
            tcgen05 dV = V·W2    ─3-pass tf32x3─► TMEM acc[2]   (or reuse acc[0])
            TMEM→reg read-back (Ld32x32bOp) ──► C_new = C_old − dV ──► RMW store to gmem
```

### TMEM accumulator usage

Three FP32 accumulators, allocated **once at kernel entry** (not per super-panel — `TmemAllocator`
in cute-DSL 4.5.2 is an allocate-once-free-at-end model per gemm_tf32x3 L56,143):

- `acc_W1` : `[IB, BW]` = `[16, 64]` — V^T·C
- `acc_W2` : `[IB, BW]` = `[16, 64]` — T^T·W1
- `acc_dV` : `[NBP, BW]` = `[64, 64]` — V·W2 (NBP≥64 = the tcgen05 TMEM row wall; we pad IB-rows of
  C-update output into a 64-row tile and only read back the live rows)

`tmem.allocate(512)` (the gemm_tf32x3 value) covers all three — B200 has 512 TMEM cols, so this is
the entire budget → **1 CTA/SM** (the documented occupancy reality; do not expect to fit 2).

### CUTLASS-lib + our-file pieces, exactly

| Need | Source | Lines |
|------|--------|-------|
| `MmaTF32Op((128,256,8), CtaGroup.ONE, OperandSource.SMEM, MajorMode.K, K)` + `make_tiled_mma` | gemm_tf32x3 | 148-150 |
| `make_smem_layout_a/b(tiled_mma, mma_tiler, Float32, ab_stages)` | gemm_tf32x3 | 151-152 |
| `SmemAllocator()` + `allocate_tensor(io_dtype, layout.outer, 128, swizzle=layout.inner)` | gemm_tf32x3 | 47,49-52 |
| `TmemAllocator(holding.ptr, barrier_for_retrieve=NamedBarrier(...))` + `.allocate/.wait_for_alloc/.retrieve_ptr/.free` | gemm_tf32x3 | 54-56,99,135,143 |
| 3-pass tf32x3: `gemm(hi,hi)`; `set(ACCUMULATE,True)`; `gemm(hi,lo)`; `gemm(lo,hi)` | gemm_tf32x3 | 128-131 |
| TMEM read-back: `Ld32x32bOp(Repetition.x64)` + `make_tmem_copy` + per-tile `cute.copy` | gemm_tf32x3 | 106-108,137-140 |
| `setmaxregister_increase/decrease` per warp | m2a | 108-111 |
| `_warp_reduce_add`, `_panel_factor`, `_apply_range`, CTA-barrier loop | m2b | 26,63,49,115-142 |
| LARFT recurrence (port to cute) | fused_qr_slice | 81-91 |

---

## 2. `@cute.struct SharedStorage` + SMEM byte budget (n=512, must fit 228 KB)

**The tf32x3-limb-doesn't-fit problem and its mitigation.** Staging all four limbs (V_hi, V_lo,
C_hi, C_lo) at a full `[128,128]` tile = ~260 KB → **blows 228 KB** (this is design-finding [2]'s
naive 325 KB result). **Chosen mitigation: stage V and C as full FP32 in SMEM, split into hi/lo
limbs *in-register* inside the MMA loop** (matches the Triton flow where `tl.dot(...,
input_precision="tf32x3")` does the split internally). This is the same `_split_tf32` arithmetic as
gemm_tf32x3 L164-168, applied to register fragments instead of host tensors.

> **Risk flag on in-register split (carry into M3c-0 testing):** gemm_tf32x3 currently splits on the
> **host** (clean, validated to ~1e-6). In-register `(x.view(i32) & ~0x1FFF).view(f32)` is the
> *unvalidated* variant. **If the M3c-0 mixed@640 margin lands < 1.83, fall back to host-split**: pass
> `V_hi,V_lo,C_hi,C_lo` pre-split, which doubles SMEM staging → use the **smaller tile (BW=64, BM=64,
> 1 ab_stage)** so 4 limbs still fit (~140 KB). Decide at the margin gate, not before.

### SharedStorage (M3c-0, in-register-split variant)

```python
NB   = 16      # IB super-panel / LARFT block width
BW   = 64      # trailing-column tile
BM   = 64      # K-loop row tile for gram + W1
NBP  = 64      # tcgen05 TMEM row floor for the C-update acc

@cute.struct
class SharedStorage:
    # --- pipeline / tmem bookkeeping (reuse gemm_tf32x3 L32-34) ---
    ab_mbar_ptr:        cute.struct.MemRange[cutlass.Int64, ab_stages * 2]  # if TMA used (M3c-2)
    acc_mbar_ptr:       cute.struct.MemRange[cutlass.Int64, acc_stage * 2]
    tmem_holding_buf:   cutlass.Int32
    # --- LARFT outputs (panel→mma handoff) ---
    sG:  cute.struct.MemRange[cutlass.Float32, NB * NB]   # gram V^T V  [16,16]
    sT:  cute.struct.MemRange[cutlass.Float32, NB * NB]   # T upper-tri [16,16]
    # --- tcgen05 operand staging (FP32, split in-reg) ---
    # sV/sC allocated as swizzled tensors via SmemAllocator (NOT in the struct) — see note
```

> **SMEM allocation split (important cute-DSL idiom):** the *swizzled* MMA operand tensors `sV`, `sC`
> are **not** struct members — they're allocated by `smem.allocate_tensor(io_dtype, layout.outer,
> 128, swizzle=layout.inner)` *after* `smem.allocate(SharedStorage)`, exactly as gemm_tf32x3 L49-52.
> The struct holds only barriers + the tiny FP32 G/T + the tmem holding int.

### Byte budget (n=512, BW=64, BM=64, in-register split)

| Component | Shape | Bytes |
|-----------|-------|------:|
| ab_mbar (2 stages × 2) | Int64×4 | 32 |
| acc_mbar (1 × 2) | Int64×2 | 16 |
| tmem_holding_buf | Int32 | 4 |
| sG | 16×16 FP32 | 1,024 |
| sT | 16×16 FP32 | 1,024 |
| sV (full FP32, swizzled) | NB-rows × BM-K staged = 64×64 | 16,384 |
| sC (full FP32, swizzled) | BM × BW = 64×64 | 16,384 |
| pipeline/swizzle padding (allocator) | — | ~2,048 |
| **Total** | | **≈ 37 KB** |

**≈ 37 KB ≪ 228 KB.** Even the host-split fallback (4 limb tiles + 1 ab_stage at 64×64 =
4×16 KB = 64 KB + overhead ≈ 80 KB) fits comfortably. **SMEM is not the binding constraint — TMEM
(1 CTA/SM) and registers are.**

> Note: M3c-0 can use **direct gmem loads** (no TMA producer warp) like m2b — then `ab_mbar` and the
> TMA atoms are unused and SMEM drops further. TMA staging is an M3c-2 optimization.

---

## 3. Apply tiling — the MMA warp's C-tile loop (2 GEMMs, tcgen05 tf32x3, RMW)

The MMA warp loops over trailing columns in BW-wide stripes. Per stripe it issues W1, W2, then the
C-update GEMM, reading back the result from TMEM and RMW-storing to gmem. This is the cute-DSL
transcription of `fused_qr_slice.py` L93-130.

```python
@cute.jit
def _apply_larft_tcgen05(Hm, c0, pend, n, lane,
                         tiled_mma, sV, sC, sT_smem,
                         acc_W1, acc_W2, acc_dV, tmem_copy):
    # super-panel V = stril(H[c0:n, c0:c0+NB]) unit-lower; T^T from sT_smem[j,i]
    M  = n - (c0 + NB)                 # trailing row count
    cc = c0 + NB                       # first trailing column
    while cc < n:
        cend = cc + BW if cc + BW < n else n

        # ============ GEMM-1 : W1 = V^T @ C   [NB, BW], K = M ============
        # K-loop over trailing rows in BM stripes (mirror fused_qr_slice L100-112)
        tiled_mma.set(tcgen05.Field.ACCUMULATE, False)   # first pass zero-inits
        first = True
        mb = c0 + NB
        while mb < n:
            mend = mb + BM if mb + BM < n else n
            # load V^T[NB, mlen] and C[mlen, BW] from gmem → sV, sC (FP32)
            #   V is unit-lower: mask V[i,j]=0 for global_row<=col, =1 on diag
            _load_Vt_unitlower(sV, Hm, c0, mb, mend, NB, n, lane)     # [NB, mlen]
            _load_C(sC, Hm, mb, mend, cc, cend, lane)                 # [mlen, BW]
            # in-register split → hi/lo fragments
            tCrVh, tCrVl = _frag_split(tiled_mma.make_fragment_A(sV))
            tCrCh, tCrCl = _frag_split(tiled_mma.make_fragment_B(sC))
            if not first:
                tiled_mma.set(tcgen05.Field.ACCUMULATE, True)
            cute.gemm(tiled_mma, acc_W1, tCrVh, tCrCh, acc_W1)        # hi*hi
            tiled_mma.set(tcgen05.Field.ACCUMULATE, True)
            cute.gemm(tiled_mma, acc_W1, tCrVh, tCrCl, acc_W1)        # hi*lo
            cute.gemm(tiled_mma, acc_W1, tCrVl, tCrCh, acc_W1)        # lo*hi
            first = False
            mb += BM
        # commit + read W1 back to SMEM/regs for the next GEMM's B-operand
        cute.arch.fence_view_async_tmem_store()          # PITFALL: fence before TMEM read
        _tmem_to_smem(acc_W1, sC_w1, tmem_copy)          # W1 [NB,BW] → SMEM (reuse sC region)

        # ============ GEMM-2 : W2 = T^T @ W1   [NB, BW], K = NB ============
        # T^T read from sT_smem[j,i]; single K-block (K=NB=16 ≤ inst K, no K-loop)
        _load_Tt(sV, sT_smem, NB)                         # sV ← T^T (reuse sV staging)
        tCrTh, tCrTl = _frag_split(tiled_mma.make_fragment_A(sV))
        tCrW1h, tCrW1l = _frag_split(tiled_mma.make_fragment_B(sC_w1))
        tiled_mma.set(tcgen05.Field.ACCUMULATE, False)
        cute.gemm(tiled_mma, acc_W2, tCrTh, tCrW1h, acc_W2)
        tiled_mma.set(tcgen05.Field.ACCUMULATE, True)
        cute.gemm(tiled_mma, acc_W2, tCrTh, tCrW1l, acc_W2)
        cute.gemm(tiled_mma, acc_W2, tCrTl, tCrW1h, acc_W2)
        cute.arch.fence_view_async_tmem_store()
        _tmem_to_smem(acc_W2, sC_w2, tmem_copy)          # W2 [NB,BW] → SMEM

        # ============ GEMM-3 + RMW : C -= V @ W2   [M, BW], K = NB ============
        mb = c0 + NB
        while mb < n:
            mend = mb + BM if mb + BM < n else n
            _load_V_unitlower(sV, Hm, c0, mb, mend, NB, n, lane)      # [mlen, NB]
            tCrVh, tCrVl = _frag_split(tiled_mma.make_fragment_A(sV))
            tCrW2h, tCrW2l = _frag_split(tiled_mma.make_fragment_B(sC_w2))
            tiled_mma.set(tcgen05.Field.ACCUMULATE, False)
            cute.gemm(tiled_mma, acc_dV, tCrVh, tCrW2h, acc_dV)
            tiled_mma.set(tcgen05.Field.ACCUMULATE, True)
            cute.gemm(tiled_mma, acc_dV, tCrVh, tCrW2l, acc_dV)
            cute.gemm(tiled_mma, acc_dV, tCrVl, tCrW2h, acc_dV)
            cute.arch.fence_view_async_tmem_store()
            # read dV back, RMW: C_new = C_old − dV  (both FP32)
            _rmw_C_minus_dV(Hm, acc_dV, mb, mend, cc, cend, tmem_copy, lane)
            mb += BM
        cc = cend
```

**Helper bodies** (small, mechanical):
- `_load_Vt_unitlower` / `_load_V_unitlower` — gmem→SMEM with the unit-lower mask from
  `fused_qr_slice` L72-73,77-78: `V = where(localrow > col, raw, where(localrow==col, 1.0, 0.0))`.
- `_frag_split(frag)` — `hi = bitcast(bitand(bitcast(frag,i32), ~0x1FFF), f32); lo = frag − hi`
  using cute register ops. **This is the unvalidated piece — gate on margin.**
- `_tmem_to_smem(acc, sdst, tmem_copy)` — the `Ld32x32bOp` read-back from gemm_tf32x3 L137-140,
  storing to SMEM instead of gmem.
- `_rmw_C_minus_dV` — read-back dV to regs, subtract from gmem-loaded C_old, store back.

**tiler note:** `mma_tiler_mnk=(128,256,32)` from gemm_tf32x3 is sized for big GEMMs. For the K-poor
`[16,BW]` and `[NBP,BW]` shapes here, **start with the gemm_tf32x3 tiler unchanged for M3c-0
correctness** (it will run at low peak — that's the documented K-poor TMEM wall, *expected*). Tune the
tiler only after correctness is locked (this is an OPEN, not an M3c-0 blocker).

---

## 4. LARFT T in cute-DSL (panel warp)

The panel warp computes `G = V^T·V` via warp-reduce, then the serial forward recurrence → `sT`. Direct
port of `fused_qr_slice.py` L61-91, using m2b's `_warp_reduce_add`.

```python
@cute.jit
def _compute_gram(Hm, sG, c0, lane, n):
    # G[i,j] = Σ_row V[row,i]·V[row,j], unit-lower V over rows [c0 .. n)
    for i in cutlass.range_constexpr(NB):
        for j in cutlass.range_constexpr(NB):     # 16×16 = 256 reduces (serial, cheap)
            p = cutlass.Float32(0.0)
            r = c0 + lane
            while r < n:
                vi = _v_unitlower(Hm, r, c0 + i, c0)   # H[r,c0+i] if r>c0+i; 1 if ==; 0 if <
                vj = _v_unitlower(Hm, r, c0 + j, c0)
                p = p + vi * vj
                r = r + 32
            g = _warp_reduce_add(p)
            if lane == 0:
                sG[i * NB + j] = g

@cute.jit
def _larft(sG, sT, tau, c0, lane, n):
    # port of fused_qr_slice L81-91; lane 0 does the serial recurrence (NB³ negligible)
    if lane == 0:
        for i in cutlass.range_constexpr(NB):
            for j in cutlass.range_constexpr(NB):
                sT[i * NB + j] = cutlass.Float32(0.0)
        sT[0] = tau[c0]                                  # T[0,0] = tau0
        for i in cutlass.range_constexpr(1, NB):
            ti = tau[c0 + i]
            # m[k] = Σ_{l<i} T[k,l] · G[l,i]   for k in [0,i)
            for k in cutlass.range_constexpr(i):
                m = cutlass.Float32(0.0)
                for l in cutlass.range_constexpr(i):
                    m = m + sT[k * NB + l] * sG[l * NB + i]
                sT[k * NB + i] = -ti * m                 # upper-right triangle
            sT[i * NB + i] = ti
```

**Convention checks (carry from finding [4] pitfalls):**
- T is **upper-triangular**; the apply needs `T^T` (lower). The MMA warp reads `sT[j*NB+i]` to get
  `T^T[i,j]` on the fly — **no explicit transpose** (matches `fused_qr_slice` storing `T` then using
  `tl.trans(T)` L91; we fold the transpose into the indexed read).
- Sign: `T[k,i] = -tau[i]·(T[:i,:i] @ G[:i,i])[k]`. **Cross-check against fused_qr_slice L89**
  (`newc = -ti*m`) — an off-by-one here silently corrupts orthogonality; validate `sT` bit-identical
  vs the Triton reference on a dense n=128 case *before* wiring the apply.
- Keep **G and T in FP32** (sign-sensitive). Only W1/W2/C-update go through tf32x3.

---

## 5. Panel ↔ MMA sync — M3c-0 (serial) and M3c-1 (overlap)

### M3c-0 — CTA barriers, serial (correctness-first)

Reuse the m2b loop verbatim (L115-142), swapping the far-apply body. Barriers give cross-warp
gmem/SMEM visibility; **`cute.arch.barrier()` synchronizes the whole CTA (both warps)** — confirmed by
m2b's working overlap. Per super-panel:

```
narrow_apply[k]            (warp1 — reuse m2b _apply_range, still warp-reduce, K-poor narrow cols)
barrier                    (next-panel cols finalized + visible)
─ serial region ─
  warp0: _compute_gram; barrier; _larft → sT         (G,T ready)
  barrier                                            (sT visible to MMA warp)
  warp1: _apply_larft_tcgen05[k]  (far cols [pend2,n))
  barrier
  warp0: _panel_factor[k+1]
barrier
```

**Do barriers suffice?** For **correctness, yes** (this is finding [3]'s verdict). The panel writes
reflectors to `H[rows>pend, c0:pend]` and `sT`; the MMA warp reads `V` from `H[c0:n, c0:pend]` (already
factored, disjoint from the next panel's columns) and `sT`. Barriers enforce the write-before-read
ordering. **TMEM ownership:** only the MMA warp retrieves the acc ptr; the panel warp never touches
TMEM. The `tmem.wait_for_alloc()` must be called by **all 64 threads** before any TMEM use (uniform
control) — call it once at kernel entry, before the super-panel loop.

### M3c-1 — overlap (panel[k+1] ‖ far_apply[k])

Toggle to the m2b `OV==1` structure: the far-apply (MMA warp) and panel-factor[k+1] (panel warp) run
**concurrently between two barriers** (disjoint columns → correct):

```
narrow_apply[k]; barrier
{ warp1: _apply_larft_tcgen05[k]   ‖   warp0: _panel_factor[k+1]; _compute_gram; _larft }
barrier
```

**Barriers vs pipelines:** CTA barriers are **correct but cap K-block pipelining** (both warps block
at the barrier). For M3c-1 start with barriers (debuggable). **Only if the whole-kernel time shows the
panel still exposed** (apply-shadow shorter than panel) consider upgrading the *intra-MMA* W1→W2→dV
chain to `PipelineUmmaAsync` (gemm_tf32x3 L70-74) for async TMEM commit — but that's an M4 lever, not
M3c. **cute-DSL 4.5.2 has no `warp_specialize`** (per memory `cutedsl-modal-loop.md`); the hand-rolled
barrier+phase-toggle (already proven in m2b) is the mechanism — this is low-risk.

---

## 6. Incremental sub-steps M3c-0 → M3c-2 (gates + kill-criteria)

| Step | Scope | Correctness gate | Perf gate | KILL if |
|------|-------|------------------|-----------|---------|
| **M3c-0a** | `_compute_gram`+`_larft` only; validate sT vs Triton | sT bit-identical to `fused_qr_slice` LARFT on n=128 dense (relerr<1e-5) | — | LARFT sign/index can't be matched → re-derive vs L81-91 |
| **M3c-0b** | Full serial kernel, **n=128 b20, n=256 b40** | H,tau vs `torch.geqrf` relerr **<1e-4**, all batches | whole-kernel **< 1.3× V10** (serial baseline) | tcgen05 apply wrong (in-reg split / TMEM read-back / V-mask) → fall back host-split, re-gate |
| **M3c-0c** | **n=512 b640**, mixed@640 | relerr<1e-4 **AND mixed@640 worst-of-640 margin ≥ 1.83** | whole-kernel measured (expect ~1.0–1.3× V10, panel unmasked) | margin <1.83 → host-split; still <1.83 → solve→fp32 robustness (+~1%) or **route n=512 to V10** |
| **M3c-1** | Add `OV=1` overlap | ov=1 == ov=0 **bit-identical** (== geqrf) | ov1/ov0 ≥ 1.05× (panel partially hides) | overlap eff ≤ 0 (panel longer than shadow) → ship M3c-0 routing or escalate to M6 |
| **M3c-2** | Tiler tune + optional TMA staging; route into submission | 22/22 via `modal_cute_lab.py --mode correctness` | geomean of n∈{≤512,B≥128} cases **≤ V10**; else fall back per-shape | net regresses vs V10 official → **do not ship**, keep V10 |

**Hard cross-cutting gates (every step):**
- `grep -niE "stream|graph" experiments/cute_qr_m3c.py` **must be empty** — re-grep after any
  `cutlass.pipeline` import (class names like `PipelineTmaUmma` are clean; verify the compiled file).
- Fresh A+Hg per config, one config per Modal run (grid=640 L2/DRAM-contention artifacts).
- 0 register spills (cuobjdump). If the MMA warp spills at 128 regs, drop BW 64→32 before host-split.
- **Test on the cute-DSL B200 replica (`modal_cute_lab.py`), then confirm the n=512 timing + margin on
  a real gpumode submission before shipping** (the project's hard-won lesson: Modal deltas <5% don't
  transfer; structural fusion+occupancy *does* — M3c is structural, so it should, but confirm).

> **Decision tree out of M3c:** if M3c-0c is correct and ~parity with V10, and M3c-1 recovers ≥5% →
> route the small-n cases through it and submit. If the panel stays exposed (likely at 1 CTA/SM), the
> real lever is **M6 persistent-queue occupancy**, not more M3c tuning — M3c's job was to prove the
> tcgen05 apply is correct + non-regressing in cute-DSL, which unblocks M6.

---

## 7. M3c-0 file skeleton — `experiments/cute_qr_m3c.py`

```python
"""M3c — tcgen05 TF32x3 BLOCKED compact-WY apply in cute-DSL (per-matrix overlap CTA).

Replaces M2b's warp-reduce far-apply with a tcgen05 tf32x3 apply C -= V @ (T^T @ (V^T @ C)) issued by
one MMA warp into TMEM accumulators; the panel warp adds gram G=V^T V (warp_reduce) + LARFT T → SMEM.
M3c-0 = serial (CTA barriers); M3c-1 = OV=1 overlap. Reuses cute_qr_m2b (panel/apply/warp_reduce/loop),
cute_gemm_tf32x3 (MmaTF32Op/TmemAllocator/3-pass tf32x3/TMEM read-back), fused_qr_slice (LARFT L81-91).
Route n<=512 → engine, else geqrf. Run: modal run modal_cute_lab.py::run_candidate --script cute_qr_m3c.py
(static-scan clean: no 'stream'/'graph'.)
"""
import torch
import cutlass
import cutlass.cute as cute
import cutlass.utils as utils
import cutlass.pipeline as pipeline
from cutlass.cute.nvgpu import tcgen05
import cutlass.utils.blackwell_helpers as sm100_utils
from cutlass.cute.runtime import from_dlpack

# ---- config ----
N_CUTE_MAX = 512
NB  = 16          # IB super-panel / LARFT block
BW  = 64          # trailing-col tile
BM  = 64          # K-loop row tile
NBP = 64          # tcgen05 TMEM row floor for C-update acc
mma_inst_shape_mnk = (128, 256, 8)
mma_tiler_mnk      = (128, 256, 32)
ab_stages = 2
acc_stage = 1
io_dtype  = cutlass.Float32
acc_dtype = cutlass.Float32

# ======================= reused-by-copy from m2b (do not re-derive) =======================
@cute.jit
def _warp_reduce_add(val):                       # m2b L26
    for i in cutlass.range_constexpr(5):
        val = val + cute.arch.shuffle_sync_bfly(val, offset=(1 << i))
    return val

# _apply_reflector (m2b L33), _apply_range (m2b L49), _panel_factor (m2b L63) — copy verbatim.
# (paste from cute_qr_m2b.py; used for narrow-apply + panel-factor + seed)

# ======================= NEW: LARFT (port of fused_qr_slice L81-91) =======================
@cute.jit
def _v_unitlower(Hm, r, col, c0):                # V[r,col] unit-lower (diag=1, above=0)
    return (Hm[r, col] if r > col else (cutlass.Float32(1.0) if r == col else cutlass.Float32(0.0)))

@cute.jit
def _compute_gram(Hm, sG, c0, lane, n: cutlass.Constexpr):   # §4
    ...

@cute.jit
def _larft(sG, sT, tm, c0, lane, n: cutlass.Constexpr):       # §4
    ...

# ======================= NEW: tcgen05 tf32x3 apply (§3) =======================
@cute.jit
def _frag_split(frag):
    hi = cute.bitcast(cute.bitwise_and(cute.bitcast(frag, cutlass.Int32), ~0x1FFF), cutlass.Float32)
    lo = frag - hi                                # PITFALL: unvalidated; gate on margin → host-split fallback
    return hi, lo

@cute.jit
def _apply_larft_tcgen05(Hm, c0, pend, n, lane, tiled_mma, sV, sC,
                         sT, acc_W1, acc_W2, acc_dV, tmem_copy):   # §3 body
    ...

# ======================= SharedStorage (§2) =======================
@cute.struct
class SharedStorage:
    acc_mbar_ptr:     cute.struct.MemRange[cutlass.Int64, acc_stage * 2]
    tmem_holding_buf: cutlass.Int32
    sG: cute.struct.MemRange[cutlass.Float32, NB * NB]
    sT: cute.struct.MemRange[cutlass.Float32, NB * NB]

# ======================= kernel =======================
@cute.kernel
def _qr_m3c_kernel(mH, mtau, a_sl, b_sl, n: cutlass.Constexpr, nb: cutlass.Constexpr,
                   OV: cutlass.Constexpr):
    bid, _, _ = cute.arch.block_idx()
    tidx, _, _ = cute.arch.thread_idx()
    warp = cute.arch.make_warp_uniform(cute.arch.warp_idx())
    lane = tidx % 32
    Hm = mH[bid, None, None]; tm = mtau[bid, None]

    if warp == 0: cute.arch.setmaxregister_increase(192)     # m2a L108-111
    else:         cute.arch.setmaxregister_decrease(128)

    smem = cutlass.utils.SmemAllocator()
    storage = smem.allocate(SharedStorage)
    sV = smem.allocate_tensor(io_dtype, a_sl.outer, 128, swizzle=a_sl.inner)   # gemm_tf32x3 L49
    sC = smem.allocate_tensor(io_dtype, b_sl.outer, 128, swizzle=b_sl.inner)

    # TMEM allocate ONCE (gemm_tf32x3 L54-56,99) — all 64 threads call wait_for_alloc
    tmem_bar = pipeline.NamedBarrier(barrier_id=1, num_threads=64)
    tmem = utils.TmemAllocator(storage.tmem_holding_buf.ptr, barrier_for_retrieve=tmem_bar)
    tmem.allocate(512)
    tmem.wait_for_alloc()
    base = tmem.retrieve_ptr(acc_dtype)
    acc_W1 = cute.make_tensor(base + 0,         _layout(NB,  BW))    # offsets pre-computed
    acc_W2 = cute.make_tensor(base + 0 + 256,   _layout(NB,  BW))
    acc_dV = cute.make_tensor(base + 0 + 512,   _layout(NBP, BW))
    tmem_copy = _make_tmem_copy(acc_dV)                              # Ld32x32bOp, gemm_tf32x3 L106-108

    # seed panel 0  (m2b L109-113)
    pend0 = nb if nb < n else n
    if warp == 0: _panel_factor(Hm, tm, 0, pend0, lane, n)
    cute.arch.barrier()

    c0 = cutlass.Int32(0)
    while c0 < n:
        pend  = c0 + nb if c0 + nb < n else n
        pend2 = pend + nb if pend + nb < n else n

        # NARROW apply (warp1, warp-reduce) — m2b L120-122
        if warp == 1: _apply_range(Hm, tm, c0, pend, pend, pend2, lane, n)
        cute.arch.barrier()

        # gram + LARFT (warp0) → sG, sT
        if warp == 0:
            _compute_gram(Hm, storage.sG, c0, lane, n)
        cute.arch.barrier()
        if warp == 0:
            _larft(storage.sG, storage.sT, tm, c0, lane, n)
        cute.arch.barrier()                                 # sT visible to MMA warp

        if OV == 1:                                         # M3c-1 overlap
            if warp == 1:
                _apply_larft_tcgen05(Hm, c0, pend, n, lane, tiled_mma, sV, sC,
                                     storage.sT, acc_W1, acc_W2, acc_dV, tmem_copy)
            if warp == 0 and pend < n:
                _panel_factor(Hm, tm, pend, pend2, lane, n)
            cute.arch.barrier()
        else:                                               # M3c-0 serial
            if warp == 1:
                _apply_larft_tcgen05(Hm, c0, pend, n, lane, tiled_mma, sV, sC,
                                     storage.sT, acc_W1, acc_W2, acc_dV, tmem_copy)
            cute.arch.barrier()
            if warp == 0 and pend < n:
                _panel_factor(Hm, tm, pend, pend2, lane, n)
            cute.arch.barrier()
        c0 = c0 + nb

    pipeline.sync(barrier_id=1)
    tmem.free(tmem.retrieve_ptr(acc_dtype))                 # gemm_tf32x3 L143

# ======================= host + routing =======================
@cute.jit
def _qr_host(mH, mtau, n: cutlass.Constexpr, nb: cutlass.Constexpr, OV: cutlass.Constexpr):
    op = tcgen05.MmaTF32Op(mma_inst_shape_mnk, tcgen05.CtaGroup.ONE, tcgen05.OperandSource.SMEM,
                           tcgen05.OperandMajorMode.K, tcgen05.OperandMajorMode.K)   # gemm L148
    tiled_mma = cute.make_tiled_mma(op)
    a_sl = sm100_utils.make_smem_layout_a(tiled_mma, mma_tiler_mnk, io_dtype, ab_stages)  # gemm L151
    b_sl = sm100_utils.make_smem_layout_b(tiled_mma, mma_tiler_mnk, io_dtype, ab_stages)
    B = cute.size(mH, mode=[0])
    _qr_m3c_kernel(mH, mtau, a_sl, b_sl, n, nb, OV).launch(grid=[B, 1, 1], block=[64, 1, 1])

_compiled = {}
def _run_cute(H, tau, ov):
    n = H.shape[-1]
    Ht = from_dlpack(H).mark_layout_dynamic(); tt = from_dlpack(tau).mark_layout_dynamic()
    key = (n, H.shape[0], ov)
    if key not in _compiled:
        _compiled[key] = cute.compile(_qr_host, Ht, tt, n, NB, ov)
    _compiled[key](Ht, tt)

def custom_kernel(data):
    A = data; B, n, _ = A.shape
    if n <= N_CUTE_MAX:
        try:
            H = A.clone().contiguous()
            tau = torch.zeros(B, n, device=A.device, dtype=A.dtype)
            _run_cute(H, tau, 1)            # OV=1
            torch.cuda.synchronize()
            return H, tau
        except Exception:
            pass
    return torch.geqrf(A)

if __name__ == "__main__":
    import traceback
    print("=== M3c serial/overlap: LARFT + tcgen05 apply correctness ===")
    torch.manual_seed(0)
    for B, n in [(1, 64), (1, 128), (4, 256), (1, 100), (4, 512)]:
        A = torch.randn(B, n, n, device="cuda", dtype=torch.float32)
        try:
            for ov in (0, 1):
                H = A.clone().contiguous(); tau = torch.zeros(B, n, device="cuda", dtype=torch.float32)
                _run_cute(H, tau, ov); torch.cuda.synchronize()
                Hg, _ = torch.geqrf(A)
                herr = (H - Hg).abs().max().item() / (Hg.abs().max().item() + 1e-30)
                print(f"  B={B} n={n:4d} ov={ov}: H={herr:.2e}  {'PASS' if herr<1e-4 else 'FAIL'}")
        except Exception:
            print(f"  B={B} n={n}: EXC\n" + traceback.format_exc()[-1800:])
    print("DONE")
```

**The undefined helpers (`_layout`, `_make_tmem_copy`, `_load_*`, `_tmem_to_smem`, `_rmw_C_minus_dV`,
`_apply_larft_tcgen05` body) are all direct transcriptions of gemm_tf32x3 L99-140 (TMEM tensor make +
read-back) and fused_qr_slice L72-128 (V mask + 2-GEMM + RMW).** Assembly = paste m2b's 4 reused
functions, port the LARFT (§4), fill the apply body (§3), then Modal-debug compile errors
mechanically. **First milestone to land: M3c-0a (LARFT bit-identical) — it needs no tcgen05 and
de-risks the sign/index convention before any GEMM wiring.**

---

## Risk summary (honest)

1. **1 CTA/SM at n=512 is structural** — TMEM 512 cols + panel 192 regs = no co-residence. The serial
   panel is exposed; M3c-1 overlap only hides it behind the (shorter) tcgen05 shadow, not across
   matrices. **Expect ~1.0–1.2× V10, not 0.87×.** The 0.87× / sub-3ms wins need M6 (persistent queue).
2. **In-register tf32x3 split is unvalidated** — the one piece not yet proven on B200. Margin gate at
   M3c-0c decides; **host-split fallback is pre-budgeted** (still fits SMEM at BW=64).
3. **Serial-panel-behind-shorter-shadow tension** — tcgen05 makes the far-apply *faster*, which
   *shrinks* the shadow the panel must hide behind → the overlap recovers *less* than warp-reduce did,
   even though total time drops. This is the key thing M3c-1's ov1/ov0 measurement reveals; don't be
   surprised if overlap eff is modest. The win (if any) is the faster apply, not the overlap.
4. **`grep stream|graph`** — re-run after every `cutlass.pipeline` import; gate before submit.
5. **K-poor tcgen05 at ~25% peak is the wall M3c does NOT escape** — do not chase higher trailing
   precision expecting speed; tf32x3 is the floor that's also fastest here.
```

Key files (all absolute):
- `/Users/raymond/Downloads/SubPY/experiments/cute_qr_m2b.py` — overlap loop + panel/apply/warp_reduce to copy
- `/Users/raymond/Downloads/SubPY/experiments/cute_gemm_tf32x3.py` — tcgen05 tf32x3 + TmemAllocator + 3-pass + TMEM read-back
- `/Users/raymond/Downloads/SubPY/experiments/cute_qr_m1b.py` — blocked QR convention
- `/Users/raymond/Downloads/SubPY/experiments/cute_qr_m2a.py` — per-warp setmaxregister
- `/Users/raymond/Downloads/SubPY/.claude/worktrees/dreamy-johnson-cc42cd/experiments/fused_qr_slice.py` — LARFT recurrence (L81-91) + 2-GEMM apply (L93-130), the bit-identical reference
- target to write: `/Users/raymond/Downloads/SubPY/experiments/cute_qr_m3c.py`

One correction to the design inputs worth flagging to the team: finding [1]'s SMEM table claims "sH[N,N]=164KB" but M3c uses **gmem H with small SMEM staging** (m2b reads reflectors straight from gmem) — the real SMEM budget is ~37 KB (in-reg split) / ~80 KB (host-split), not 168 KB; SMEM is not the binding constraint, TMEM (1 CTA/SM) is. The guide above uses the correct ~37 KB figure.