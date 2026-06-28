# Research plan: SMEM-resident streaming panel → unlock design-A overlap (~+6%)

> ## ⛔ STAGE A = NO-GO (2026-06-26). DESIGN A IS DEAD IN PRINCIPLE. Do not build Stages B–D below.
> Built the smem-streaming panel (`experiments/gluon_panel_smem.py`, `results/stageA_smem_panel_NOGO.txt`):
> bit-faithful (relerr 1e-7), DOES lower registers (RBLK=32 → **62 regs** ≤ the 64 gate), **BUT 2.56–6.74×
> SLOWER** than the register-resident panel (RBLK=256→2.56×, RBLK=32→6.74×). Structural: a smem-resident
> tile needs 2 passes/reflector (reduce w → rank-1 update) hitting smem twice; the register panel keeps the
> tile resident (~free). **The register gate and the ≤1.5× speed gate are MUTUALLY EXCLUSIVE.** So the panel
> must stay register-resident for speed → it blows the 16-warp budget co-resident with the trailing → design A
> (in-CTA warp_specialize) is closed. Implementation notes that DID work and are reusable: runtime `range` +
> 3D-smem `.index(runtime ib)` lowers regs (NOT `gl.static_range`, which unrolls and keeps all blocks live);
> `.slice(start)` needs a compile-time start. **Only remaining overlap path = design B (inter-CTA/persistent),
> a major multi-day build for the same bounded ~6%. Best official unchanged: V5 tf32x3 5915µs.**
>
> _(Original plan below — Stages B–D are now moot for design A; kept for the design-B fallback reasoning.)_

> **Branch `gluon-smem-panel`** (from the de-risk session commit). Read `CLAUDE.md` + the
> `docs/HANDOVER_NEXT_SESSION.md` banner + `[[qr-v2-gluon-warp-specialize]]` memory FIRST —
> this plan assumes the full Phase 0/1/1.5 diagnosis below.

## The single remaining unknown (everything else is de-risked)
Design A (in-CTA `gl.warp_specialize`: panel ‖ trailing) overlap was validated to **eff 0.64–0.73
at the 16-warp config** WHEN the default partition is low-register (`results/phase1.5_16warp_async.txt`).
The ONLY thing blocking it is: **the real Householder panel keeps its `[M, ib]` tile resident in
registers (~96–128 regs/thread); panel(8w, heavy) + async-trailing-worker(8w) = 16 warps blows the
64K register file → spill/occupancy collapse → WS up to 2.7× SLOWER.**

**Fix = make the panel LOW-REGISTER by keeping the tile in SHARED memory and streaming it in
row-blocks** (only one block in registers at a time). Then panel(8w, ~20–32 regs) + async-worker
(8w, 128 regs) ≈ 41K regs < 64K → fits 1 CTA/SM → overlaps like the proxy (eff ~0.65 → ~+6% geomean).

## What is ALREADY BUILT + VERIFIED (reuse these — do not rebuild)
- `experiments/gluon_panel.py` — register-resident Gluon port of `_panel_kernel`, **bit-faithful**
  to `_panel_factor` (relerr ~1e-7 incl M=512). This is the CORRECTNESS REFERENCE for the smem version.
- `experiments/gluon_panel_async.py::_async_trail_part` — **low-level async tcgen05 TILED trailing
  worker** (private per-tile mbarrier, low-register). Correct (trailing relerr ~8e-4 = tf32x3). This
  is the worker. **CRITICAL: do NOT use `tl_dot` in the worker** — its CTA-wide barrier couples the
  warp_specialize partitions and kills overlap (measured eff 0.06). Worker needs 8 warps / 128 regs.
- The async tcgen05 recipe + `gl.warp_specialize` mechanics (`gluon_lowlevel_dump.py`,
  `gluon_ws_dump.py`): `warp_specialize([(default_fn,args),(worker_fn,args)], [wk_warps],[wk_regs])`;
  partitions must NOT return tensors (write outputs directly); `tcgen05_mma(..., mbarriers=[bar])`
  (must pass mbarriers or it's synchronous); init bar count=NPASS.
- Test harness: `modal_microbench.py --script <x>` (single B200 script); `modal_lab.py` (correctness/
  compare). Modal path `/Users/raymond/Downloads/SubPY/.modalenv/bin/modal`.

## The smem-streaming panel design (the build)
Current per-j-iteration (register version) holds the whole `[BN, BCOLS]` tile in regs across all
BCOLS iterations. The streaming version keeps it in smem and re-streams row-blocks:
```
smem_tile = gl.allocate_shared_memory([BN, BCOLS], layout)   # tile lives in SMEM
load global -> smem_tile ; fence
for j in range(BCOLS):                    # BCOLS sequential reflectors (latency chain)
    colj = <load column j from smem>      # [BN] in regs (cheap, 1 column)
    alpha = reduce(colj==diag); xnorm2 = reduce(colj rows>j)   # scalars
    v = householder(colj, alpha, xnorm2)  # [BN] in regs
    # w = v^T @ tile  — stream row-blocks (only RBLK rows in regs at once):
    w = zeros([BCOLS])
    for rb in range(0, BN, RBLK):
        trb = <smem_load rows rb:rb+RBLK>            # [RBLK, BCOLS] in regs
        w += sum_rows(v[rb:rb+RBLK,None] * trb)
    # rank-1 update C[:,>j] -= tau*v⊗w — stream row-blocks, write back to smem:
    for rb in range(0, BN, RBLK):
        trb = <smem_load rows rb:rb+RBLK>
        trb = where(col>j, trb - tau*v[rb:rb+RBLK,None]*w[None,:], trb)
        <smem_store rows rb:rb+RBLK> trb
    <smem_store column j> (newcol = v + diag)
store smem_tile -> global ; store tau
```
Register peak ≈ one row-block `[RBLK, BCOLS]` (RBLK=32/64 → ~8–16 regs) + `colj`/`v` `[BN]` (~2 regs)
+ `w` `[BCOLS]`. Target ≤ ~32 regs/thread. **Cost risk:** 2 smem passes over the tile per j-iter
(w-reduce + update) = ~2× the smem traffic of the register version; the panel is SRAM-bandwidth-
bound, so this MAY slow the panel. Stage A measures exactly this.

Gluon smem APIs (confirmed present): `gl.allocate_shared_memory(dtype, shape, SwizzledSharedLayout/
NVMMASharedLayout)`; smem descriptor `.load(layout)` / `.store(value)` / `.index()` / `.slice(start,len)`
(seen in `gluon_lowlevel_dump.py` output). Use `.slice`/`.index` for row-blocks; `fence_async_shared()`
between global→smem and first read.

## STAGED GO/NO-GO plan (≤1.5 days to first verdict)

### Stage A — build the smem-streaming panel, measure register count + speed (THE gate). ~½ day.
1. Port `gluon_panel.py` to the streaming form above. Verify **bit-faithful** vs `panel_ref`
   (relerr <1e-4) on B=8, M∈{512,384,256,128}, b=64. (Reuse `gluon_panel.py`'s reference + harness.)
2. Report compiled **n_regs / n_spills** (`kernel.n_regs` from the Triton compiled handle) and the
   **panel time** vs the register-resident `gluon_panel.py` at grid=640.
- **GO**: regs ≤ ~64/thread AND panel time ≤ ~1.5× the register panel → Stage B.
- **NO-GO**: regs still high (streaming didn't lower peak) OR panel >2× slower (smem bandwidth wall)
  → design A dead even in principle. **Fall back to design B (inter-CTA) or ship V5 as-is.**

### Stage B — overlap the smem panel (8w) ‖ async trailing worker (8w) at grid=640. ~½ day.
Combine the Stage-A smem panel (default partition) with `_async_trail_part` (worker, 8w/128r) via
`gl.warp_specialize`, real n=512 shapes, sweep panel heights 384/256/128. Measure WS vs SEQ.
- **GO**: eff > 0.5 AND WS nets >1.1× vs SEQ on panel+trailing → Stage C.
- **NO-GO** (16-warp occupancy at grid=640 still serializes despite low regs) → design B / ship V5.

### Stage C — full fused factorization (multi-day, THE build).
Implement the look-ahead pipeline per super-step: do trailing(k) look-ahead block [k+nb:k+2nb] first,
then `warp_specialize(panel(k+1) ‖ trailing-rest(k))`. Fold gram (V^T V) + the T-factor solve into
the flow (these stay on the critical path — they do NOT overlap; keep solve fp32-safe per the
solve-tf32 artifact). Build incrementally: 1 matrix vs `torch.geqrf` → batched 640 → 22/22.
Keep panel reductions FP32 (low-prec panel corrupts reflectors — proven dead).

### Stage D — integrate, validate, submit. ~1 day.
Shape-route the Gluon overlap kernel to **n≤512 ONLY** (n=1024 single-CTA penalty = 2.31×; keep the
existing path for n≥1024). Gate: (a) 22/22 `modal_lab.py --mode correctness`; (b) **mixed@640
worst-of-640 margin ≥ 1.83 probed SEPARATELY** (the lab gate is batch-16 and MISSES it — use the
`microbench_validate.py`/`microbench_1x_reseed.py` pattern); (c) `--mode compare` geomean; (d)
`grep -niE "stream|graph" submission.py` empty. **SUBMIT to gpumode to confirm official** — Modal
codegen gains can fail to transfer (the fp16x3 lesson). Best official to beat: V5 tf32x3 **5915µs**.

## Honest payoff + risk
- **Realistic payoff ~+6% geomean** (eff~0.65 × trailing 26% on the 4 n=512 cases). 5915 → ~5560µs.
  Still ~4.3× off the leader (1292µs) — this is a bounded research win, not a leaderboard leap.
- **Top risk: Stage A** — the smem panel may be too slow (SRAM-bandwidth-bound, 2× passes). If so,
  the whole direction is dead and the fallback is design B (inter-CTA, separate full-register CTAs,
  no register sharing — premise untested, harder orchestration) or simply shipping V5.
- The 16-warp config forces ~1 CTA/SM → grid=640 runs ~4.3 waves; the async-worker proxy already
  showed this overlaps (eff 0.64–0.73), but the REAL smem panel's occupancy interaction is Stage B's
  residual risk.

## Do-NOT-retry (this session + prior) — see `docs/DEAD_ENDS.md`
tl_dot in the worker (CTA-barrier coupling, eff 0.06); register-resident panel co-resident (blows
16-warp budget); 3xcuBLAS trailing (0.866×); recursive/3-level K-rich blocking (0.976×, NB widening
eats it); fp8/fp4/mxfp; 1×TF32 trailing; fp16x3 (official wash); raw-PTX/load_inline (no nvcc);
streams/cooperative-launch (rule + substring scan); conditioning-routing/early-stop (reward-hack DQ).
