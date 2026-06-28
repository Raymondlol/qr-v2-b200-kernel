"""M3c probe #3b — FULL 3-GEMM WY apply chain: dV = V @ (T^T @ (V^T @ C)), tcgen05 tf32x3.

The decisive apply-chain validation before integration. One super-panel, one BW=64 stripe, M rows
(G3 single output tile, M<=NBP=64). Controlled synthetic V (unit-lower mask), T (upper-tri), C; compares
the kernel dV to the reference V_masked @ (T^T @ (V_masked^T @ C)). Validates the NEW pieces beyond the
proven GEMM-1: transposed restage of W1/W2 (B[w,k]=W1[k,w]); the K=NB=16 GEMMs with the tiler-K=32 TAIL
zeroed (cols [16,32)); T^T index-swap read (mT[k,i]). W1/W2 staged through gmem scratch (the validated
readback->gmem path; SMEM staging is a later perf opt). The C-=dV subtraction is trivial -> on host.

Reuses: swizzle-fill sX[(r,ki),0,kb,0], in-reg split (x.bitcast(i32)&-8192), acc-pipeline readback drain.
Run: modal run modal_cute_lab.py::run_candidate --script cute_apply_chain.py   (static-scan clean.)
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
BM = 32
NBP = 64


@cute.struct
class SharedStorage:
    acc_mbar_ptr: cute.struct.MemRange[cutlass.Int64, acc_stage * 2]
    tmem_holding_buf: cutlass.Int32


@cute.jit
def _hi(x: cutlass.Float32) -> cutlass.Float32:
    xi = x.bitcast(cutlass.Int32)
    return (xi & cutlass.Int32(-8192)).bitcast(cutlass.Float32)


@cute.kernel
def kernel(tiled_mma: cute.TiledMma, mV, mT, mC, mW1, mW2, mDV,
           a_sl: cute.ComposedLayout, b_sl: cute.ComposedLayout, M: cutlass.Constexpr):
    tidx, _, _ = cute.arch.thread_idx()
    warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())

    smem = cutlass.utils.SmemAllocator()
    storage = smem.allocate(SharedStorage)
    sAh = smem.allocate_tensor(io_dtype, a_sl.outer, 128, swizzle=a_sl.inner)
    sAl = smem.allocate_tensor(io_dtype, a_sl.outer, 128, swizzle=a_sl.inner)
    sBh = smem.allocate_tensor(io_dtype, b_sl.outer, 128, swizzle=b_sl.inner)
    sBl = smem.allocate_tensor(io_dtype, b_sl.outer, 128, swizzle=b_sl.inner)

    tmem_alloc_barrier = pipeline.NamedBarrier(barrier_id=1, num_threads=threads_per_cta)
    tmem = utils.TmemAllocator(storage.tmem_holding_buf.ptr, barrier_for_retrieve=tmem_alloc_barrier)
    tmem.allocate(512)
    acc_producer, acc_consumer = pipeline.PipelineUmmaAsync.create(
        num_stages=acc_stage,
        producer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread),
        consumer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread, threads_per_cta),
        barrier_storage=storage.acc_mbar_ptr.data_ptr()).make_participants()

    thr_mma = tiled_mma.get_slice(0)
    tAh = tiled_mma.make_fragment_A(sAh); tAl = tiled_mma.make_fragment_A(sAl)
    tBh = tiled_mma.make_fragment_B(sBh); tBl = tiled_mma.make_fragment_B(sBl)
    acc_shape = tiled_mma.partition_shape_C(mma_tiler_mnk[:2])
    tCtAcc = tiled_mma.make_fragment_C(acc_shape)
    tmem.wait_for_alloc()
    tmem_ptr = tmem.retrieve_ptr(acc_dtype)
    tCtAcc = cute.make_tensor(tmem_ptr, tCtAcc.layout)

    subtile = 4
    epi_tiler = ((cute.size(tCtAcc, mode=[0, 0]), cute.size(tCtAcc, mode=[0, 1]) // subtile),)
    tCtAcc_epi = cute.zipped_divide(tCtAcc, epi_tiler)
    tmem_atom = cute.make_copy_atom(tcgen05.Ld32x32bOp(tcgen05.Repetition.x64), cutlass.Float32)
    tmem_tiled_copy = tcgen05.make_tmem_copy(tmem_atom, tCtAcc_epi[None, 0])
    tmem_thr_copy = tmem_tiled_copy.get_slice(tidx)
    tDtC = tmem_thr_copy.partition_S(tCtAcc_epi)
    nkb = cute.size(tAh, mode=[2])

    def dst_part(mOut):                          # partition_D over a [128,256] gmem out tile, subtile 0
        g = cute.local_tile(mOut, mma_tiler_mnk, (0, 0, None), proj=(1, 1, None))
        return tmem_thr_copy.partition_D(cute.zipped_divide(thr_mma.partition_C(g), epi_tiler))

    tDgW1 = dst_part(mW1); tDgW2 = dst_part(mW2); tDgDV = dst_part(mDV)
    rb = cute.make_rmem_tensor(tDgDV[None, None, 0].shape, acc_dtype)

    # =================== G1: W1 = V^T @ C  (K-loop over M rows) -> mW1 ===================
    W1acc = cute.make_rmem_tensor(tDgDV[None, None, 0].shape, acc_dtype)
    for e in cutlass.range_constexpr(cute.size(W1acc)):
        W1acc[e] = cutlass.Float32(0.0)
    num_kt = (M + BM - 1) // BM
    kt = 0
    while kt < num_kt:
        r0 = kt * BM
        i = tidx
        while i < IB * 32:
            ii = i // 32; kk = i - ii * 32; kb = kk // 8; ki = kk - kb * 8; r = r0 + kk
            v = cutlass.Float32(0.0)
            if r < M:
                vraw = mV[r, ii]
                v = vraw if r > ii else (cutlass.Float32(1.0) if r == ii else cutlass.Float32(0.0))
            h = _hi(v); sAh[(ii, ki), 0, kb, 0] = h; sAl[(ii, ki), 0, kb, 0] = v - h
            i = i + threads_per_cta
        i = tidx
        while i < BW * 32:
            ww = i // 32; kk = i - ww * 32; kb = kk // 8; ki = kk - kb * 8; r = r0 + kk
            cv = cutlass.Float32(0.0)
            if r < M:
                cv = mC[r, ww]
            h = _hi(cv); sBh[(ww, ki), 0, kb, 0] = h; sBl[(ww, ki), 0, kb, 0] = cv - h
            i = i + threads_per_cta
        cute.arch.barrier()
        if warp_idx == 0:
            acc_empty = acc_producer.acquire_and_advance()
            tiled_mma.set(tcgen05.Field.ACCUMULATE, False)
            for kb in cutlass.range_constexpr(nkb):
                kc = (None, None, kb, 0)
                cute.gemm(tiled_mma, tCtAcc, tAh[kc], tBh[kc], tCtAcc)
                tiled_mma.set(tcgen05.Field.ACCUMULATE, True)
                cute.gemm(tiled_mma, tCtAcc, tAh[kc], tBl[kc], tCtAcc)
                cute.gemm(tiled_mma, tCtAcc, tAl[kc], tBh[kc], tCtAcc)
            acc_empty.commit()
        acc_full = acc_consumer.wait_and_advance()
        cute.copy(tmem_tiled_copy, tDtC[None, None, 0], rb)
        for e in cutlass.range_constexpr(cute.size(W1acc)):
            W1acc[e] = W1acc[e] + rb[e]
        acc_full.release()
        cute.arch.barrier()
        kt = kt + 1
    w1io = cute.make_rmem_tensor(tDgDV[None, None, 0].shape, io_dtype)
    w1io.store(W1acc.load().to(io_dtype))
    cute.autovec_copy(w1io, tDgW1[None, None, 0])           # W1 -> gmem mW1
    cute.arch.barrier()

    # =================== G2: W2 = T^T @ W1  (K=NB=16, tail-zeroed) -> mW2 ===================
    i = tidx
    while i < IB * 32:                           # A = T^T : A[ii,k] = T[k,ii] (k<16 else 0)
        ii = i // 32; k = i - ii * 32; kb = k // 8; ki = k - kb * 8
        a = mT[k, ii] if k < IB else cutlass.Float32(0.0)
        h = _hi(a); sAh[(ii, ki), 0, kb, 0] = h; sAl[(ii, ki), 0, kb, 0] = a - h
        i = i + threads_per_cta
    i = tidx
    while i < BW * 32:                           # B = W1 : B[ww,k] = W1[k,ww] (k<16 else 0)
        ww = i // 32; k = i - ww * 32; kb = k // 8; ki = k - kb * 8
        b = mW1[k, ww] if k < IB else cutlass.Float32(0.0)
        h = _hi(b); sBh[(ww, ki), 0, kb, 0] = h; sBl[(ww, ki), 0, kb, 0] = b - h
        i = i + threads_per_cta
    cute.arch.barrier()
    if warp_idx == 0:
        acc_empty = acc_producer.acquire_and_advance()
        tiled_mma.set(tcgen05.Field.ACCUMULATE, False)
        for kb in cutlass.range_constexpr(nkb):
            kc = (None, None, kb, 0)
            cute.gemm(tiled_mma, tCtAcc, tAh[kc], tBh[kc], tCtAcc)
            tiled_mma.set(tcgen05.Field.ACCUMULATE, True)
            cute.gemm(tiled_mma, tCtAcc, tAh[kc], tBl[kc], tCtAcc)
            cute.gemm(tiled_mma, tCtAcc, tAl[kc], tBh[kc], tCtAcc)
        acc_empty.commit()
    acc_full = acc_consumer.wait_and_advance()
    cute.copy(tmem_tiled_copy, tDtC[None, None, 0], rb)
    w2io = cute.make_rmem_tensor(tDgDV[None, None, 0].shape, io_dtype)
    w2io.store(rb.load().to(io_dtype))
    cute.autovec_copy(w2io, tDgW2[None, None, 0])           # W2 -> gmem mW2
    acc_full.release()
    cute.arch.barrier()

    # =================== G3: dV = V @ W2  (K=NB=16, single output tile rows [0,M)) ===================
    i = tidx
    while i < NBP * 32:                          # A = V : A[rr,k] = mask(V[rr,k]) (k<16 else 0)
        rr = i // 32; k = i - rr * 32; kb = k // 8; ki = k - kb * 8
        a = cutlass.Float32(0.0)
        if k < IB and rr < M:
            a = mV[rr, k] if rr > k else (cutlass.Float32(1.0) if rr == k else cutlass.Float32(0.0))
        h = _hi(a); sAh[(rr, ki), 0, kb, 0] = h; sAl[(rr, ki), 0, kb, 0] = a - h
        i = i + threads_per_cta
    i = tidx
    while i < BW * 32:                           # B = W2 : B[ww,k] = W2[k,ww] (k<16 else 0)
        ww = i // 32; k = i - ww * 32; kb = k // 8; ki = k - kb * 8
        b = mW2[k, ww] if k < IB else cutlass.Float32(0.0)
        h = _hi(b); sBh[(ww, ki), 0, kb, 0] = h; sBl[(ww, ki), 0, kb, 0] = b - h
        i = i + threads_per_cta
    cute.arch.barrier()
    if warp_idx == 0:
        acc_empty = acc_producer.acquire_and_advance()
        tiled_mma.set(tcgen05.Field.ACCUMULATE, False)
        for kb in cutlass.range_constexpr(nkb):
            kc = (None, None, kb, 0)
            cute.gemm(tiled_mma, tCtAcc, tAh[kc], tBh[kc], tCtAcc)
            tiled_mma.set(tcgen05.Field.ACCUMULATE, True)
            cute.gemm(tiled_mma, tCtAcc, tAh[kc], tBl[kc], tCtAcc)
            cute.gemm(tiled_mma, tCtAcc, tAl[kc], tBh[kc], tCtAcc)
        acc_empty.commit()
    acc_full = acc_consumer.wait_and_advance()
    cute.copy(tmem_tiled_copy, tDtC[None, None, 0], rb)
    dvio = cute.make_rmem_tensor(tDgDV[None, None, 0].shape, io_dtype)
    dvio.store(rb.load().to(io_dtype))
    cute.autovec_copy(dvio, tDgDV[None, None, 0])           # dV -> gmem mDV
    acc_full.release()
    tmem.relinquish_alloc_permit()
    pipeline.sync(barrier_id=1)
    tmem.free(tmem_ptr)


@cute.jit
def host_function(v, t, c, w1, w2, dv, M: cutlass.Constexpr):
    op = tcgen05.MmaTF32Op(mma_inst_shape_mnk, tcgen05.CtaGroup.ONE, tcgen05.OperandSource.SMEM,
                           tcgen05.OperandMajorMode.K, tcgen05.OperandMajorMode.K)
    tiled_mma = cute.make_tiled_mma(op)
    a_sl = sm100_utils.make_smem_layout_a(tiled_mma, mma_tiler_mnk, io_dtype, ab_stages)
    b_sl = sm100_utils.make_smem_layout_b(tiled_mma, mma_tiler_mnk, io_dtype, ab_stages)
    kernel(tiled_mma, v, t, c, w1, w2, dv, a_sl, b_sl, M).launch(grid=[1, 1, 1], block=(threads_per_cta, 1, 1))


def run(M=64, tol=1e-4):
    import torch
    print(f"=== probe#3b FULL apply chain dV=V(T^T(V^T C))  M={M} IB={IB} BW={BW} ===")
    torch.manual_seed(5)
    V = torch.randn(M, IB, device="cuda", dtype=torch.float32)
    T = torch.triu(torch.randn(IB, IB, device="cuda", dtype=torch.float32))
    C = torch.randn(M, BW, device="cuda", dtype=torch.float32)
    W1 = torch.zeros(128, 256, device="cuda", dtype=torch.float32)
    W2 = torch.zeros(128, 256, device="cuda", dtype=torch.float32)
    DV = torch.zeros(128, 256, device="cuda", dtype=torch.float32)
    Tn = lambda x, d: (from_dlpack(x, assumed_align=16).mark_layout_dynamic(leading_dim=1)
                       .mark_compact_shape_dynamic(mode=1, divisibility=d))
    host_function(Tn(V, IB), Tn(T, IB), Tn(C, BW), Tn(W1, 256), Tn(W2, 256), Tn(DV, 256), M, no_cache=True)
    torch.cuda.synchronize()
    rows = torch.arange(M, device="cuda").view(M, 1); cols = torch.arange(IB, device="cuda").view(1, IB)
    Vm = torch.where(rows > cols, V, torch.where(rows == cols, torch.ones_like(V), torch.zeros_like(V)))
    refW1 = Vm.t() @ C
    refdV = Vm @ (T.t() @ refW1)
    rW1 = (W1[:IB, :BW] - refW1).abs().max().item() / (refW1.abs().max().item() + 1e-30)
    rDV = (DV[:M, :BW] - refdV).abs().max().item() / (refdV.abs().max().item() + 1e-30)
    print(f"  M={M:4d}: W1 rel={rW1:.2e}  dV rel={rDV:.2e}   -> {'PASS tf32x3' if rDV < tol else 'FAIL'}")
    return rDV


if __name__ == "__main__":
    import traceback
    p = argparse.ArgumentParser(); p.add_argument("--M", type=int, default=64)
    args = p.parse_args()
    try:
        run(args.M)
    except Exception:
        print("EXC\n" + traceback.format_exc()[-3000:])
    print("DONE")
