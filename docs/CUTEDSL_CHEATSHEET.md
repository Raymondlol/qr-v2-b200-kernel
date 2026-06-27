# cute-dsl 4.5.2 API Cheatsheet (B200 / sm_100a, CUDA 12.9)

Pinned to `nvidia-cutlass-dsl==4.5.2` (CUTLASS repo tag `v4.5.1`). All snippets are verbatim from
the v4.5.1 `examples/python/CuTeDSL/...` sources + the installed-wheel module introspection.

**Provenance of each snippet** (raw base path `examples/python/CuTeDSL/`):
- `EW` = `cute/ampere/kernel/elementwise/elementwise_add.py`
- `G0/G1/G3` = `cute/blackwell/tutorial/tutorial_gemm/fp16_gemm_{0,1,3}.py`
- `T0/T1` = `cute/blackwell/tutorial/tutorial_tma/tma_v{0,1}.py`
- `DG` = `cute/blackwell/kernel/dense_gemm/dense_gemm.py`
- `DGP` = `cute/blackwell/kernel/dense_gemm/dense_gemm_persistent.py`
- `GG` = `cute/hopper/kernel/grouped_gemm/grouped_gemm.py`
- `LIB:` = library source `python/CuTeDSL/cutlass/cute/arch/*.py` (verbatim signatures)

> ⚠️ HARD CONSTRAINT for this competition: the submission file must NOT contain the substrings
> `stream` or `graph` ANYWHERE (naive static scan). The `stream=`/`current_stream` plumbing shown
> in DG is **OPTIONAL** — the tutorial GEMMs (G0/G1/EW) compile + launch with no stream arg. Use the
> stream-free idiom. See §8.

---

## 1. Kernel skeleton (#1 deliverable — fixes the launch SIGSEGV)

### The two decorators, exact shapes

```python
# EW / G0 / G1  — device kernel
@cute.kernel
def kernel(
    tiled_mma: cute.TiledMma,
    tma_atom_a: cute.CopyAtom,
    mA_mkl: cute.Tensor,
    ...
    a_smem_layout: cute.ComposedLayout,   # ComposedLayout = layout+swizzle, NOT a plain Layout
):
    tidx, _, _ = cute.arch.thread_idx()
    bidx, bidy, _ = cute.arch.block_idx()
    ...   # no return

# G1 — host JIT entry. Takes cute.Tensor args, builds atoms/layouts, calls kernel(...).launch(...)
@cute.jit
def host_function(a: cute.Tensor, b: cute.Tensor, c: cute.Tensor):
    op = tcgen05.MmaF16BF16Op(...)
    tiled_mma = cute.make_tiled_mma(op)
    ...
    kernel(tiled_mma, a_tma_atom, a_tma_tensor, ...).launch(
        grid=grid_shape,
        block=[threads_per_cta, 1, 1],
        cluster=cluster_shape_mnk,   # OPTIONAL — omit if no cluster (G0 omits it)
    )
```

`@cute.kernel` is also callable with parens — `@cute.kernel()` (G1 uses this). A class method works too
(T1/DG): `@cute.jit def __call__(self, src, dst):` + `@cute.kernel def kernel(self, ...):`.

### The EXACT `.launch()` signature

The kernel call object returned by `kernel(*args)` has `.launch(...)`. Observed kwargs:

```python
# G0 — minimal, NO cluster, NO stream:
kernel(tiled_mma, a_tma_atom, a_tma_tensor, b_tma_atom, b_tma_tensor, c,
       a_smem_layout, b_smem_layout).launch(
    grid=grid_shape,                 # list/tuple [gx, gy, gz]  (or a Shape)
    block=(threads_per_cta, 1, 1),   # list/tuple, MUST be 3 entries
)

# G1 / T1 — with cluster:
... .launch(
    grid=grid_shape,
    block=[threads_per_cta, 1, 1],
    cluster=cluster_shape_mnk,        # 3-tuple, e.g. (2,1,1); REQUIRED if MMA/TMA use a cluster
)

# DG — with stream (DO NOT USE: trips the substring scanner):
... .launch(grid=..., block=..., cluster=..., stream=current_stream)   # ← avoid
```

- `grid` / `block` are **3-element** sequences. A bad block length or a grid that under-provisions vs the
  per-CTA tensor slicing is the classic launch SIGSEGV — see §2 for the per-CTA slice contract.
- `cluster=` must match the cluster the `tiled_mma`/TMA atoms were built for. Omit it entirely for 1-CTA
  kernels (G0). Mismatched cluster → illegal launch.

### `cute.compile` — what's baked vs passed at call time

```python
# EW: compile returns a callable; args passed at compile time become the *trace* signature.
compiled_func = cute.compile(elementwise_add, a_tensor, b_tensor, c_tensor,
                             options="--generate-line-info")
compiled_func(a_tensor, b_tensor, c_tensor)        # call with SAME positional arg list

# G1: compile is implicit — calling the @cute.jit fn directly traces+runs (no_cache=True forces retrace)
host_function(a_tensor, b_tensor, c_tensor, no_cache=True)
```

Rules:
- **`cutlass.Constexpr`-typed params and any non-`cute.Tensor` python value are BAKED at compile/trace
  time** (shapes, tile sizes, dtypes, layouts-as-constants). They specialize the compiled kernel.
- **`cute.Tensor` args are passed again at call time** — the compiled object is invoked with the same
  positional list it was compiled with. Only the data pointer/dynamic dims vary per call (those marked
  `.mark_layout_dynamic()` / `.mark_compact_shape_dynamic()`).
- `options="--generate-line-info"` is accepted; useful for debugging the SIGSEGV with `cuda-gdb`.
- Default arg with a Constexpr type: `def elementwise_add(mA, mB, mC, copy_bits: cutlass.Constexpr = 128):`

---

## 2. Tensors

### from_dlpack (the ONLY torch→cute bridge)

```python
from cutlass.cute.runtime import from_dlpack

# EW — simplest, fully dynamic:
a_tensor = from_dlpack(a).mark_layout_dynamic()

# G1 / T1 — K-major, aligned, with divisibility hints (needed for TMA + good codegen):
a_tensor = (
    from_dlpack(a, assumed_align=32)          # or 16; must match real alignment
    .mark_layout_dynamic(leading_dim=1)
    .mark_compact_shape_dynamic(mode=1, divisibility=k)
)
```

`assumed_align` must be a real alignment of the torch buffer or TMA will fault. For a `[B,n,n]` batched
QR input, align to at least 16B; if rows are contiguous, `leading_dim` is the last contiguous mode.

### Per-CTA slice of a [B, n, n] batch (one matrix per block) — THE QR PATTERN

You want `block_idx().x == batch index`, then a 2-D `[n,n]` view. Two idioms:

```python
# (A) EW-style zipped_divide + per-block coord. gA is [(tile), num_blocks]; pick column = bidx:
blk_coord = ((None, None), bidx)     # ((all-of-tile-mode), block_linear_idx)
blkA = gA[blk_coord]                 # the per-CTA tile, shape == tiler_mn

# (B) local_tile — slice mA[B,n,n] by giving a per-mode coord; None = keep whole mode:
#   treat batch as mode 0, give coord (bidx, None, None) -> a [n,n] sub-tensor
gMat = cute.local_tile(mA_bnn, tiler=(1, n, n), coord=(bidx, None, None))
# or the GEMM idiom (G1): project out modes you keep with proj=(...):
gA = cute.local_tile(mA_mkl, mma_tiler_mnk, mma_coord_mnk, proj=(1, None, 1))
```

The launch SIGSEGV is almost always: grid has fewer CTAs than the index you slice with, OR the slice
coord doesn't reduce rank as expected (a leftover singleton mode the MMA/TMA atom didn't expect).
`cute.printf("grid={} bidx={}", grid_shape, bidx)` and `print(cute.pretty_str(gMat))` at trace time.

### SMEM tensors

```python
# EW/T1: allocate a typed SMEM tensor through the allocator + a layout (+ optional swizzle):
smem = cutlass.utils.SmemAllocator()           # alias: cutlass.cute? no -> cutlass.utils.SmemAllocator
storage = smem.allocate(SharedStorage)         # SharedStorage = @cute.struct (see below)
sA = smem.allocate_tensor(
    element_type=io_dtype,
    layout=a_smem_layout.outer,                 # .outer = the plain layout part
    byte_alignment=128,
    swizzle=a_smem_layout.inner,                # .inner = the swizzle part of a ComposedLayout
)
# T1 alternative: get a tensor from a struct member range:
sA = storage.sA.get_tensor(smem_layout_sA.outer, swizzle=smem_layout_sA.inner)
```

`@cute.struct` declares the SMEM layout (sizes are compile-time):

```python
@cute.struct
class SharedStorage:
    ab_mbar_ptr:  cute.struct.MemRange[cutlass.Int64, ab_stages * 2]   # mbarrier storage
    acc_mbar_ptr: cute.struct.MemRange[cutlass.Int64, acc_stage * 2]
    tmem_dealloc_mbar: cutlass.Int64
    tmem_holding_buf:  cutlass.Int32
    sA: cute.struct.Align[cute.struct.MemRange[dtype, cute.cosize(smem_layout_sA)], 128]
    sB: cute.struct.Align[cute.struct.MemRange[dtype, cute.cosize(smem_layout_sB)], 128]
# access:  storage.ab_mbar_ptr.data_ptr()   storage.tmem_holding_buf.ptr   storage.sA.get_tensor(...)
```

### Layouts + tiling/partition vocabulary

```python
cute.make_layout((M, N), stride=(s0, s1))                # explicit
cute.make_ordered_layout((4, 32), order=(1, 0))          # EW: thread/value layouts
cute.make_layout_tv(thr_layout, val_layout)              # EW: -> (tiler_mn, tv_layout)
cute.make_identity_tensor(shape)                         # coordinate tensor (predication)
cute.zipped_divide(mA, tiler_mn)                         # EW: tile -> [(tile), num_tiles]
cute.local_tile(t, tiler, coord, proj=(...))             # G1: per-CTA tile with mode projection
cute.tiled_divide(layout, (tiled_mma.thr_id,))           # G1: build VMNK cta layout
cute.flat_divide(t, epi_tile)                            # G3 epilogue subtiling
cute.group_modes(t, 0, 3)                                # collapse modes [0,3) into one (TMA partition)
cute.select(layout, mode=[0,1,2])                        # pick a sub-layout
cute.make_fragment_like(t)                               # RMEM fragment matching a partition
cute.make_rmem_tensor(shape, dtype)                      # explicit RMEM tensor
cute.size(t, mode=[i]) ; cute.cosize(layout) ; cute.rank(t) ; cute.ceil_div(a,b) ; cute.round_up(a,b)
# slicing: t[(None, i)] keeps mode-0, fixes mode-1=i.  None == ":" (keep whole mode).
```

---

## 3. Thread / warp / block / sync

```python
# LIB: cutlass/cute/arch/nvvm_wrappers.py  — all return Int32 / 3-tuples of Int32
tidx, tidy, tidz = cute.arch.thread_idx()         # within CTA
bidx, bidy, bidz = cute.arch.block_idx()          # CTA id in grid
bdx,  bdy,  bdz  = cute.arch.block_dim()          # threads/CTA per dim
gdx,  gdy,  gdz  = cute.arch.grid_dim()           # CTAs/grid per dim
warp_idx = cute.arch.warp_idx()                   # warp index within CTA (tid//32 across x,y,z)
warp_idx = cute.arch.make_warp_uniform(warp_idx)  # REQUIRED before branching on warp_idx (uniformizes)
lane     = cute.arch.lane_idx()                   # 0..31
rank_in_cluster = cute.arch.block_idx_in_cluster()        # G1 (single int)
vmnk_rank       = cute.arch.block_idx_in_cluster()        # used as cluster coord source

# Barriers (LIB: nvvm_wrappers.py). Note: at exact CUDA 12.9 these lower to inline `bar.sync` asm.
cute.arch.barrier()                                       # plain __syncthreads (all threads in CTA)
cute.arch.barrier(barrier_id=1, number_of_threads=128)   # named barrier (a subset sync)
cute.arch.barrier_arrive(barrier_id=1, number_of_threads=128)  # arrive-only (no wait)
cute.arch.sync_threads()                                  # alias for full CTA barrier
cute.arch.sync_warp(mask=0xffffffff)                      # warp-wide

# Named barrier as an object (preferred for warp-spec; G1/G3/T1):
bar = pipeline.NamedBarrier(barrier_id=1, num_threads=128)
bar.arrive_and_wait()      # both sides
# (pipeline.sync(barrier_id=1) is a module-level helper used at teardown, G1)

# elect_one — single thread per warp; CONTEXT MANAGER (LIB: arch/elect.py):
with cute.arch.elect_one():
    cute.arch.mbarrier_init(barrier_ptr, 1)
    cute.arch.mbarrier_expect_tx(barrier_ptr, num_bytes)
# ⚠️ Do NOT wrap cute.copy(TMA...) or cute.gemm(...) in elect_one — they self-elect; wrapping deadlocks.
```

### mbarrier — exact signatures (LIB: `cutlass/cute/arch/mbar.py`)

```python
cute.arch.mbarrier_init(mbar_ptr: Pointer, cnt: Int)                  # arrival count
cute.arch.mbarrier_init_fence()                                       # after all inits, before use
cute.arch.mbarrier_expect_tx(mbar_ptr, bytes: Int, peer_cta_rank=None)
cute.arch.mbarrier_arrive_and_expect_tx(mbar_ptr, bytes, peer_cta_rank=None)
cute.arch.mbarrier_arrive(mbar_ptr, peer_cta_rank=None, arrive_count: Int = 1)
cute.arch.mbarrier_wait(mbar_ptr, phase: Int)                        # blocking; phase flips 0/1 each round
cute.arch.mbarrier_try_wait(mbar_ptr, phase) -> Boolean              # non-blocking
cute.arch.mbarrier_conditional_try_wait(cond: Boolean, mbar_ptr, phase)
```

Manual one-shot handshake (T0/T1 verbatim):

```python
if tidx == 0:
    cute.arch.mbarrier_init(load_mbar_ptr, 1)
    cute.arch.mbarrier_expect_tx(load_mbar_ptr, self.num_tma_load_bytes)
    cute.arch.mbarrier_init(store_mbar_ptr, len(self.trans_warp_id))
cute.arch.mbarrier_init_fence()
cute.arch.barrier()
# producer warp:
cute.copy(tma_atom_load, tAgA[(None,bidx,bidy)], tAsA[(None,0)], tma_bar_ptr=load_mbar_ptr)
with cute.arch.elect_one():
    cute.arch.mbarrier_arrive(load_mbar_ptr)
# consumer warps:
cute.arch.mbarrier_wait(load_mbar_ptr, 0)     # phase 0 first wait
```

---

## 4. GEMM

### (a) Simplest path for M1 — plain tile MMA (the `tl.dot` equivalent)

For a from-scratch fused-QR M1, the lowest-friction "multiply two SMEM/RMEM tiles into an accumulator"
is `cute.gemm(tiled_mma, acc, a_frag, b_frag, acc)`. Even the "plain" path on B200 goes through a
`tiled_mma`; there is no separate scalar dot. Build a tf32 `tiled_mma`, make A/B fragments, accumulate:

```python
# tf32 input precision is selected by the OP CLASS (MmaTF32Op) — operands are TFloat32, acc Float32.
from cutlass.cute.nvgpu import tcgen05
op = tcgen05.MmaTF32Op(                       # LIB: tcgen05/mma.py:609  (instruction K is fixed = 8)
    (M_inst, N_inst, 8),                      # instruction_shape; K-mode MUST be 8 for TF32
    tcgen05.CtaGroup.ONE,                     # or .TWO for 2-CTA
    tcgen05.OperandSource.SMEM,               # A from SMEM (or .TMEM)
    tcgen05.OperandMajorMode.K,               # a_major_mode
    tcgen05.OperandMajorMode.K,               # b_major_mode
)
tiled_mma = cute.make_tiled_mma(op)

# fragments + accumulator (G1 idiom):
tCrA = tiled_mma.make_fragment_A(sA)          # (MMA, MMA_M, MMA_K)
tCrB = tiled_mma.make_fragment_B(sB)          # (MMA, MMA_N, MMA_K)
acc_shape = tiled_mma.partition_shape_C(mma_tiler_mnk[:2])
tCtAcc = tiled_mma.make_fragment_C(acc_shape) # (MMA, MMA_M, MMA_N) — lives in TMEM after the swap (§4b)

# the MMA loop (G0/G1/G3 verbatim):
tiled_mma.set(tcgen05.Field.ACCUMULATE, False)          # first K-block overwrites
num_k_blocks = cute.size(tCrA, mode=[2])
for k_block_idx in cutlass.range_constexpr(num_k_blocks):
    k_block_coord = (None, None, k_block_idx, stage_index)
    cute.gemm(tiled_mma, tCtAcc, tCrA[k_block_coord], tCrB[k_block_coord], tCtAcc)
    tiled_mma.set(tcgen05.Field.ACCUMULATE, True)        # subsequent K-blocks accumulate
```

> tf32x3 (the project's precision floor): there is no single "3xtf32" op — emulate by splitting each
> fp32 operand into hi/mid/lo tf32 limbs and issuing 3 (or 6) `MmaTF32Op` passes accumulating into the
> same Float32 `tCtAcc`. The op fixes inputs to `TFloat32`, acc to `Float32`.

### (b) Full Blackwell tcgen05 path → TMEM accumulator

```python
# 1. build op + tiled_mma (fp16 example; swap MmaF16BF16Op->MmaTF32Op for tf32):
op = tcgen05.MmaF16BF16Op(io_dtype, acc_dtype, mma_inst_shape_mnk,
                          tcgen05.CtaGroup.ONE,           # .TWO + 2x1 cluster for 2-CTA MMA (G1)
                          tcgen05.OperandSource.SMEM,
                          tcgen05.OperandMajorMode.K, tcgen05.OperandMajorMode.K)
tiled_mma = cute.make_tiled_mma(op)

# 2. SMEM layouts via helpers (G1):
import cutlass.utils.blackwell_helpers as sm100_utils
a_smem_layout = sm100_utils.make_smem_layout_a(tiled_mma, mma_tiler_mnk, a.element_type, ab_stages)
b_smem_layout = sm100_utils.make_smem_layout_b(tiled_mma, mma_tiler_mnk, b.element_type, ab_stages)

# 3. allocate TMEM and SWAP the accumulator pointer into it (G1 verbatim):
tmem_alloc_barrier = pipeline.NamedBarrier(barrier_id=1, num_threads=threads_per_cta)
tmem = utils.TmemAllocator(
    storage.tmem_holding_buf.ptr,
    barrier_for_retrieve=tmem_alloc_barrier,
    is_two_cta=cute.size(cta_layout_vmnk, mode=[0]) > 1,
    two_cta_tmem_dealloc_mbar_ptr=storage.tmem_dealloc_mbar.ptr,
)
num_tmem_cols = 512
tmem.allocate(num_tmem_cols)            # only warp 0 actually allocs
tmem.wait_for_alloc()                   # CTA-wide sync before reading the ptr
tmem_ptr = tmem.retrieve_ptr(acc_dtype)
tCtAcc = cute.make_tensor(tmem_ptr, tCtAcc.layout)    # rebind acc tensor onto TMEM

# 4. mma loop: same cute.gemm loop as (a)

# 5. read TMEM acc back -> RMEM -> (convert) -> GMEM (G1 verbatim):
tmem_atom = cute.make_copy_atom(tcgen05.Ld32x32bOp(tcgen05.Repetition.x64), cutlass.Float32)
tmem_tiled_copy = tcgen05.make_tmem_copy(tmem_atom, tCtAcc_epi[None, 0])
tmem_thr_copy = tmem_tiled_copy.get_slice(tidx)
tDtC = tmem_thr_copy.partition_S(tCtAcc_epi)
tDgC = tmem_thr_copy.partition_D(gC_epi)
tCrAcc = cute.make_rmem_tensor(tDgC[None,None,0].shape, acc_dtype)
tCrC   = cute.make_rmem_tensor(tDgC[None,None,0].shape, io_dtype)
for i in cutlass.range(cute.size(tDtC, mode=[2])):
    cute.copy(tmem_tiled_copy, tDtC[None,None,i], tCrAcc)
    tCrC.store(tCrAcc.load().to(io_dtype))            # type conversion
    cute.autovec_copy(tCrC, tDgC[None,None,i])

# 6. teardown:
tmem.relinquish_alloc_permit()
pipeline.sync(barrier_id=1)
tmem.free(tmem_ptr)
```

Low-level alternative to `TmemAllocator` (LIB: `arch/tmem.py`): `cute.arch.alloc_tmem(num_columns,
smem_ptr_to_write_address, is_two_cta)`, `cute.arch.retrieve_tmem_ptr(element_type, alignment,
ptr_to_buffer_holding_addr)`, `cute.arch.relinquish_tmem_alloc_permit(is_two_cta)`,
`cute.arch.dealloc_tmem(tmem_ptr, num_columns, is_two_cta)`.

---

## 5. TMA (descriptor + async ring + handshake)

### Build the descriptor (atom) on the HOST

```python
from cutlass.cute.nvgpu import cpasync, tcgen05

# Simple (T0/T1) — single tile, no MMA coupling:
tma_atom_src, tma_tensor_src = cpasync.make_tiled_tma_atom(
    cpasync.CopyBulkTensorTileG2SOp(),    # GMEM->SMEM load   (S2G = store: CopyBulkTensorTileS2GOp)
    src,                                   # the from_dlpack cute.Tensor
    smem_layout,                           # one-stage SMEM layout
    (tile_m, tile_n),                      # cta tiler
)

# MMA-coupled A/B (G1) — returns atom + a reshaped tensor to slice with the mma coord:
op = cpasync.CopyBulkTensorTileG2SMulticastOp(tcgen05.CtaGroup.TWO)   # or non-multicast G2S for 1-CTA
a_tma_atom, a_tma_tensor = cute.nvgpu.make_tiled_tma_atom_A(
    op, a, a_smem_layout_one_stage, mma_tiler_mnk, tiled_mma, cta_layout_vmnk.shape)
b_tma_atom, b_tma_tensor = cute.nvgpu.make_tiled_tma_atom_B(
    op, b, b_smem_layout_one_stage, mma_tiler_mnk, tiled_mma, cta_layout_vmnk.shape)
```

### Partition + async copy (in-kernel)

```python
# prefetch the descriptor once (warp 0), G1:
if warp_idx == 0:
    cpasync.prefetch_descriptor(tma_atom_a)

# partition: maps the per-CTA SMEM tile <-> the gmem tensor through the atom (G1):
tAsA, tAgA = cute.nvgpu.cpasync.tma_partition(
    tma_atom_a,
    cta_in_cluster_coord_vmnk[2],                 # mcast coord (use 0 for 1-CTA, T0)
    cute.make_layout(cute.size(cta_layout_vmnk, mode=[2])),   # or cute.make_layout(1) for 1-CTA
    cute.group_modes(sA, 0, 3),                   # SMEM grouped
    cute.group_modes(tCgA, 0, 3),                 # gmem (mma-partitioned) grouped
)

# issue the load into stage `index`, signalling the stage's mbarrier with the byte count:
cute.copy(tma_atom_a,
          tAgA[(None, k_tile_idx)],               # src gmem slice
          tAsA[(None, stage_index)],              # dst smem stage
          tma_bar_ptr=ab_empty.barrier,           # the mbarrier to complete (expect_tx already set)
          mcast_mask=tma_mcast_mask_a)            # omit for 1-CTA
```

### Multi-stage ring with PipelineTmaUmma (G1 verbatim — producer=TMA, consumer=MMA)

```python
import cutlass.pipeline as pipeline
num_tma_copy_bytes = (cute.size_in_bytes(io_dtype, cute.select(a_smem_layout, mode=[0,1,2]))
                    + cute.size_in_bytes(io_dtype, cute.select(b_smem_layout, mode=[0,1,2]))
                     ) * cute.size(cta_layout_vmnk, mode=[0])

ab_producer, ab_consumer = pipeline.PipelineTmaUmma.create(
    num_stages=ab_stages,
    producer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread),
    consumer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread, num_tma_producer),
    tx_count=num_tma_copy_bytes,                  # bytes the mbarrier expects per stage
    barrier_storage=storage.ab_mbar_ptr.data_ptr(),
    cta_layout_vmnk=cta_layout_vmnk,
).make_participants()

# producer (TMA) loop:
ab_empty = ab_producer.acquire_and_advance()      # wait stage empty; returns handle{.index,.count,.barrier}
cute.copy(tma_atom_a, tAgA[(None, ab_empty.count)], tAsA[(None, ab_empty.index)],
          tma_bar_ptr=ab_empty.barrier, mcast_mask=tma_mcast_mask_a)
# ... after loop: ab_producer.tail()

# consumer (MMA) loop:
ab_full = ab_consumer.wait_and_advance()          # wait stage full
cute.gemm(tiled_mma, tCtAcc, tCrA[(None,None,kb,ab_full.index)], tCrB[...], tCtAcc)
ab_full.release()                                  # mark stage empty again
```

`PipelineUmmaAsync.create(...)` (same args minus `is_leader_cta` baked) is the MMA→epilogue accumulator
pipeline. `PipelineTmaStore.create(num_stages, producer_group)` is the epilogue SMEM→GMEM store pipeline
(G3): `.producer_acquire()/.producer_commit()/.producer_tail()`.

---

## 6. Persistent scheduler

```python
import cutlass.utils as utils

# HOST: build params from the C-tile grid, get max resident clusters from HW, size the grid:
tile_sched_params = utils.PersistentTileSchedulerParams(num_ctas_mnl, cluster_shape_mnl)
max_active_clusters = utils.HardwareInfo().get_max_active_clusters(
    cluster_shape_mn[0] * cluster_shape_mn[1])           # = #clusters that co-reside on the SMs
grid = utils.StaticPersistentTileScheduler.get_grid_shape(tile_sched_params, max_active_clusters)
#   -> a persistent grid (≈ num_persistent_clusters), NOT one-CTA-per-tile.
# num_ctas_mnl example (DGP): gc = cute.zipped_divide(c, tiler=c_shape); num_ctas_mnl = gc[(0,(None,None,None))].shape

# DEVICE: create scheduler from block/grid, then drive the work loop:
tile_sched = utils.StaticPersistentTileScheduler.create(
    tile_sched_params, cute.arch.block_idx(), cute.arch.grid_dim())
work_tile = tile_sched.initial_work_tile_info()
while work_tile.is_valid_tile:
    cur_tile_coord = work_tile.tile_idx               # (m_tile, n_tile, l) coord
    # ... factor / gemm this tile ...
    tile_sched.advance_to_next_work()
    work_tile = tile_sched.get_current_work()
```

`WorkTileInfo` fields used: `.is_valid_tile` (bool), `.tile_idx` (coord tuple). For a per-MATRIX QR
work-queue (one CTA factors one [n,n] matrix), set the C-tile == whole matrix so each work item is one
batch index; grid = persistent (#SMs-ish) and the loop hands out batch matrices.

---

## 7. Reg realloc + explicit warp roles (NO `warp_specialize` in 4.5.2)

Warp specialization = `make_warp_uniform(warp_idx)` then explicit `if warp_idx == ...:` branches, with
each role doing its own reg realloc. Verbatim role constants + branch shape (G3 / DGP):

```python
epilogue_warp_ids = (0, 1, 2, 3)
mma_warp_id = 4
tma_warp_id = 5

warp_idx = cute.arch.warp_idx()
warp_idx = cute.arch.make_warp_uniform(warp_idx)        # MANDATORY before branching

if warp_idx == tma_warp_id:
    cpasync.prefetch_descriptor(tma_atom_a)
    while work_tile.is_valid_tile:
        handle = ab_producer.acquire_and_advance()
        cute.copy(tma_atom_a, ..., tma_bar_ptr=handle.barrier, mcast_mask=...)
        tile_sched.advance_to_next_work(); work_tile = tile_sched.get_current_work()
    ab_producer.tail()
elif warp_idx == mma_warp_id:
    tmem.wait_for_alloc(); tmem_ptr = tmem.retrieve_ptr(acc_dtype)
    tCtAcc_base = cute.make_tensor(tmem_ptr, tCtAcc_fake.layout)
    while work_tile.is_valid_tile:
        if is_leader_cta:
            acc_empty = acc_producer.acquire_and_advance()
            tCtAcc = tCtAcc_base[(None, None, None, acc_empty.index)]
            tiled_mma.set(tcgen05.Field.ACCUMULATE, False)
            for k_tile_idx in range(num_k_tiles):
                handle = ab_consumer.wait_and_advance()
                for kb in cutlass.range_constexpr(cute.size(tCrA, mode=[2])):
                    cute.gemm(tiled_mma, tCtAcc, tCrA[(None,None,kb,handle.index)], tCrB[...], tCtAcc)
                    tiled_mma.set(tcgen05.Field.ACCUMULATE, True)
                handle.release()
            acc_empty.commit()
        tile_sched.advance_to_next_work(); work_tile = tile_sched.get_current_work()
    acc_producer.tail()
elif warp_idx < mma_warp_id:           # epilogue warps 0..3
    tmem.allocate(num_tmem_cols=512); tmem.wait_for_alloc()
    ...                                # TMEM->RMEM->SMEM->GMEM, see §4b/§5
    tmem.relinquish_alloc_permit(); tmem.free(tmem_ptr)
```

### Reg realloc (LIB + GG verbatim) — give heavy warps more registers, DMA warps fewer

```python
# constants (GG): self.load_register_requirement = 40 ; self.mma_register_requirement = 232
is_dma_warp_group = warp_group_idx < self.num_dma_warp_groups
if is_dma_warp_group:
    cute.arch.warpgroup_reg_dealloc(self.load_register_requirement)   # -> 40 regs (TMA/producer)
if not is_dma_warp_group:
    cute.arch.warpgroup_reg_alloc(self.mma_register_requirement)      # -> 232 regs (MMA + epilogue)
```

Verbatim signatures (LIB: `arch/nvvm_wrappers.py`):
```python
cute.arch.warpgroup_reg_alloc(reg_count: int)      # nvvm.setmaxregister(.., increase)
cute.arch.warpgroup_reg_dealloc(reg_count: int)    # nvvm.setmaxregister(.., decrease)  ⚠ @deprecated
```
> `warpgroup_reg_dealloc` is marked **`@deprecated("use setmaxregister_decrease instead")`** in 4.5.2 —
> it still works (it just calls `setmaxregister(.., decrease)`), but prefer the non-deprecated path if a
> `setmaxregister_decrease` wrapper is exported. `reg_count` must be a multiple of 8, per-warp budget; the
> alloc total across warps must fit 64K regs/SM. For QR: panel-reduction warps = alloc-high (≈232),
> tcgen05-issuer/TMA warps = dealloc-low (≈40).

---

## 8. Gotchas

**`stream` / `graph` substrings (HARD: trips the submission scanner).**
- `cute.compile(...)` and `.launch(...)` are fully usable WITHOUT any stream arg — G0/G1/EW pass none.
  Do NOT copy DG's stream plumbing (`torch.cuda.current_stream()`, `cuda.CUstream`, `.launch(stream=...)`,
  `compiled(... , current_stream)`). Those literally contain `stream`.
- `cute.arch.sm_id()` docstring says "Streaming Multiprocessor" — docstrings/comments DO count in a naive
  scan. If you call it, the import is fine but don't paste the docstring; and `grep -niE "stream|graph"`
  the final file regardless.
- No CuTeDSL public API you need here has `stream`/`graph` in the *name* — `cpasync`, `tcgen05`,
  `pipeline`, `tma_*`, `mbarrier_*` are all clean. The only leak vector is the stream-launch convenience
  pattern and docstrings. **Always run `grep -niE "stream|graph" submission.py` before submit.**

**float32 / tf32 handling.**
- tf32 path = `tcgen05.MmaTF32Op` — it FORCES operands to `TFloat32` and accumulator to `Float32`;
  instruction K-mode is fixed to **8** (it raises if you pass another K). There is no native 3xtf32 op;
  do tf32x3 by limb-splitting fp32 → 3 tf32 passes into one Float32 acc.
- Convert on store with `frag.store(other.load().to(dtype))` (G1/G3); element type is set at
  `make_rmem_tensor(shape, dtype)` / fragment creation time.
- Panel reductions must stay fp32 (project constraint) — keep those in `Float32` RMEM, not via the tf32 op.

**Constexpr rules.**
- Anything typed `cutlass.Constexpr` (or any plain python int/shape/dtype/layout passed at trace) is BAKED
  and specializes the kernel. Tensor shapes you want dynamic must come through `from_dlpack(...).mark_*`.
- Use `cutlass.const_expr(cond)` for compile-time branches (T1: `if cutlass.const_expr(a.dtype != b.dtype)`).
- Loops: `cutlass.range(n)` (runtime), `cutlass.range_constexpr(n)` (unrolled/compile-time, for K-blocks),
  `cutlass.range(n, prefetch_stages=...)` for SW-pipelined loops (G1).

**CUDA 12.9 / experimental module.**
- `cutlass.cute.experimental` is **NOT importable under CUDA < 13.1** — do not `import` or reference it.
  (4.5.2 is pinned at CUDA 12.9 here.) Stick to `cutlass.cute`, `cutlass.cute.nvgpu.{cpasync,tcgen05}`,
  `cutlass.pipeline`, `cutlass.utils`, `cutlass.cute.arch`.
- At exact 12.9 the `cute.arch.barrier`/`barrier_arrive` lower to inline `bar.sync`/`bar.arrive` PTX asm
  (version-gated in the lib) — behavior is correct, just be aware named-barrier semantics route through asm.
- `make_warp_uniform(warp_idx)` is REQUIRED before any `if warp_idx == ...` warp-role branch, else the
  branch is not treated as uniform and codegen/sync can be wrong (a plausible SIGSEGV/hang source).
- `tmem.wait_for_alloc()` MUST precede `retrieve_ptr` in every warp that reads the acc; reading the TMEM
  pointer before the allocating warp finished = garbage pointer = fault.
