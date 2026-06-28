"""M3c probe #3a — apply GEMM-1 W1=V^T C with a K-LOOP (M>32) + per-tile drain.

Extends the validated single-tile GEMM-1 (cute_apply_gemm1.py, rel 8.25e-7) to M up to 512: the
contraction K=M is loaded in BM=32-row tiles, looping. The SMEM-overwrite race (async UMMA still
reading a tile's SMEM while the next tile refills it) is avoided by **per-tile readback + register
accumulate**: each K-tile MMAs a FRESH acc (ACC=False), the acc-pipeline readback DRAINS the UMMA
(acc_consumer.wait), the partial is added into a persistent register fragment W1acc, then the stage is
released and SMEM refilled. Uses only validated primitives. If this PASSES at M=128/512 ~1e-6, the
K-loop is locked and the full 3-GEMM chain is just more of the same.

Run: modal run modal_cute_lab.py::run_candidate --script cute_apply_gemm1_kloop.py   (static-scan clean.)
"""
import argparse
import cutlass
import cutlass.cute as cute
import cutlass.utils as utils
import cutlass.pipeline as pipeline
from cutlass.cute.nvgpu import tcgen05
import cutlass.utils.blackwell_helpers as sm100_utils
from cutlass.cute.runtime import from_dlpack

io_dtype = cutlass.Float32
acc_dtype = cutlass.Float32
mma_inst_shape_mnk = (128, 256, 8)
mma_tiler_mnk = (128, 256, 32)
threads_per_cta = 128
ab_stages = 1
acc_stage = 1
IB = 16
BW = 64
BM = 32                                # K-tile = tiler K-mode


@cute.struct
class SharedStorage:
    acc_mbar_ptr: cute.struct.MemRange[cutlass.Int64, acc_stage * 2]
    tmem_holding_buf: cutlass.Int32


@cute.jit
def _split_hi(x: cutlass.Float32) -> cutlass.Float32:
    xi = x.bitcast(cutlass.Int32)
    return (xi & cutlass.Int32(-8192)).bitcast(cutlass.Float32)


@cute.kernel
def kernel(tiled_mma: cute.TiledMma, mV: cute.Tensor, mC: cute.Tensor, mW: cute.Tensor,
           a_sl: cute.ComposedLayout, b_sl: cute.ComposedLayout, M: cutlass.Constexpr):
    tidx, _, _ = cute.arch.thread_idx()
    warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())

    smem = cutlass.utils.SmemAllocator()
    storage = smem.allocate(SharedStorage)
    sVh = smem.allocate_tensor(io_dtype, a_sl.outer, 128, swizzle=a_sl.inner)
    sVl = smem.allocate_tensor(io_dtype, a_sl.outer, 128, swizzle=a_sl.inner)
    sCh = smem.allocate_tensor(io_dtype, b_sl.outer, 128, swizzle=b_sl.inner)
    sCl = smem.allocate_tensor(io_dtype, b_sl.outer, 128, swizzle=b_sl.inner)

    tmem_alloc_barrier = pipeline.NamedBarrier(barrier_id=1, num_threads=threads_per_cta)
    tmem = utils.TmemAllocator(storage.tmem_holding_buf.ptr, barrier_for_retrieve=tmem_alloc_barrier)
    tmem.allocate(512)
    acc_producer, acc_consumer = pipeline.PipelineUmmaAsync.create(
        num_stages=acc_stage,
        producer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread),
        consumer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread, threads_per_cta),
        barrier_storage=storage.acc_mbar_ptr.data_ptr()).make_participants()

    gW = cute.local_tile(mW, mma_tiler_mnk, (0, 0, None), proj=(1, 1, None))
    thr_mma = tiled_mma.get_slice(0)
    tCgW = thr_mma.partition_C(gW)
    tVh = tiled_mma.make_fragment_A(sVh); tVl = tiled_mma.make_fragment_A(sVl)
    tCh = tiled_mma.make_fragment_B(sCh); tCl = tiled_mma.make_fragment_B(sCl)
    acc_shape = tiled_mma.partition_shape_C(mma_tiler_mnk[:2])
    tCtAcc = tiled_mma.make_fragment_C(acc_shape)
    tmem.wait_for_alloc()
    tmem_ptr = tmem.retrieve_ptr(acc_dtype)
    tCtAcc = cute.make_tensor(tmem_ptr, tCtAcc.layout)

    subtile = 4                                   # subtile-0 = cols [0,64) = BW
    epi_tiler = ((cute.size(tCtAcc, mode=[0, 0]), cute.size(tCtAcc, mode=[0, 1]) // subtile),)
    tCtAcc_epi = cute.zipped_divide(tCtAcc, epi_tiler)
    gW_epi = cute.zipped_divide(tCgW, epi_tiler)
    tmem_atom = cute.make_copy_atom(tcgen05.Ld32x32bOp(tcgen05.Repetition.x64), cutlass.Float32)
    tmem_tiled_copy = tcgen05.make_tmem_copy(tmem_atom, tCtAcc_epi[None, 0])
    tmem_thr_copy = tmem_tiled_copy.get_slice(tidx)
    tDtC = tmem_thr_copy.partition_S(tCtAcc_epi)
    tDgC = tmem_thr_copy.partition_D(gW_epi)
    tCrAcc = cute.make_rmem_tensor(tDgC[None, None, 0].shape, acc_dtype)
    W1acc = cute.make_rmem_tensor(tDgC[None, None, 0].shape, acc_dtype)
    for e in cutlass.range_constexpr(cute.size(W1acc)):
        W1acc[e] = cutlass.Float32(0.0)

    num_kt = (M + BM - 1) // BM
    kt = 0
    while kt < num_kt:
        r0 = kt * BM
        i = tidx
        while i < IB * 32:                        # A = V^T : sVh[(ii,ki),0,kb,0] = mask(V[r0+kk, ii])
            ii = i // 32; kk = i - ii * 32; kb = kk // 8; ki = kk - kb * 8; r = r0 + kk
            v = cutlass.Float32(0.0)
            if r < M:
                vraw = mV[r, ii]
                v = vraw if r > ii else (cutlass.Float32(1.0) if r == ii else cutlass.Float32(0.0))
            hi = _split_hi(v)
            sVh[(ii, ki), 0, kb, 0] = hi
            sVl[(ii, ki), 0, kb, 0] = v - hi
            i = i + threads_per_cta
        i = tidx
        while i < BW * 32:                        # B = C^T : sCh[(ww,ki),0,kb,0] = C[r0+kk, ww]
            ww = i // 32; kk = i - ww * 32; kb = kk // 8; ki = kk - kb * 8; r = r0 + kk
            cval = cutlass.Float32(0.0)
            if r < M:
                cval = mC[r, ww]
            hi = _split_hi(cval)
            sCh[(ww, ki), 0, kb, 0] = hi
            sCl[(ww, ki), 0, kb, 0] = cval - hi
            i = i + threads_per_cta
        cute.arch.barrier()

        if warp_idx == 0:
            acc_empty = acc_producer.acquire_and_advance()
            tiled_mma.set(tcgen05.Field.ACCUMULATE, False)
            nkb = cute.size(tVh, mode=[2])
            for kb in cutlass.range_constexpr(nkb):
                kc = (None, None, kb, 0)
                cute.gemm(tiled_mma, tCtAcc, tVh[kc], tCh[kc], tCtAcc)
                tiled_mma.set(tcgen05.Field.ACCUMULATE, True)
                cute.gemm(tiled_mma, tCtAcc, tVh[kc], tCl[kc], tCtAcc)
                cute.gemm(tiled_mma, tCtAcc, tVl[kc], tCh[kc], tCtAcc)
            acc_empty.commit()
        acc_full = acc_consumer.wait_and_advance()
        cute.copy(tmem_tiled_copy, tDtC[None, None, 0], tCrAcc)      # drains UMMA
        for e in cutlass.range_constexpr(cute.size(W1acc)):
            W1acc[e] = W1acc[e] + tCrAcc[e]
        acc_full.release()
        cute.arch.barrier()
        kt = kt + 1

    tmem.relinquish_alloc_permit()
    tCrW = cute.make_rmem_tensor(tDgC[None, None, 0].shape, io_dtype)
    tCrW.store(W1acc.load().to(io_dtype))
    cute.autovec_copy(tCrW, tDgC[None, None, 0])
    pipeline.sync(barrier_id=1)
    tmem.free(tmem_ptr)


@cute.jit
def host_function(v: cute.Tensor, c: cute.Tensor, w: cute.Tensor, M: cutlass.Constexpr):
    op = tcgen05.MmaTF32Op(mma_inst_shape_mnk, tcgen05.CtaGroup.ONE, tcgen05.OperandSource.SMEM,
                           tcgen05.OperandMajorMode.K, tcgen05.OperandMajorMode.K)
    tiled_mma = cute.make_tiled_mma(op)
    a_sl = sm100_utils.make_smem_layout_a(tiled_mma, mma_tiler_mnk, io_dtype, ab_stages)
    b_sl = sm100_utils.make_smem_layout_b(tiled_mma, mma_tiler_mnk, io_dtype, ab_stages)
    kernel(tiled_mma, v, c, w, a_sl, b_sl, M).launch(grid=[1, 1, 1], block=(threads_per_cta, 1, 1))


def run(M, tol=1e-5):
    import torch
    print(f"=== probe#3a GEMM-1 K-LOOP  M={M}  ({(M+BM-1)//BM} k-tiles)  IB={IB} BW={BW} ===")
    torch.manual_seed(3)
    V = torch.randn(M, IB, device="cuda", dtype=torch.float32)
    C = torch.randn(M, BW, device="cuda", dtype=torch.float32)
    W = torch.zeros(128, 256, device="cuda", dtype=torch.float32)
    T = lambda x, d: (from_dlpack(x, assumed_align=16).mark_layout_dynamic(leading_dim=1)
                      .mark_compact_shape_dynamic(mode=1, divisibility=d))
    host_function(T(V, IB), T(C, BW), T(W, 256), M, no_cache=True)
    torch.cuda.synchronize()
    rows = torch.arange(M, device="cuda").view(M, 1); cols = torch.arange(IB, device="cuda").view(1, IB)
    Vm = torch.where(rows > cols, V, torch.where(rows == cols, torch.ones_like(V), torch.zeros_like(V)))
    ref = Vm.t() @ C
    rel = (W[:IB, :BW] - ref).abs().max().item() / (ref.abs().max().item() + 1e-30)
    tag = "PASS tf32x3" if rel < tol else ("1xtf32?" if rel < 2e-3 else "FAIL")
    print(f"  M={M:4d}: max rel err = {rel:.2e}   -> {tag}")
    return rel


if __name__ == "__main__":
    import traceback
    p = argparse.ArgumentParser(); p.add_argument("--M", type=int, default=0)
    args = p.parse_args()
    Ms = [args.M] if args.M else [32, 64, 128, 256, 512]
    for M in Ms:
        try:
            run(M)
        except Exception:
            print(f"  M={M}: EXC\n" + traceback.format_exc()[-2500:])
    print("DONE")
