"""M6 per-matrix engine — panel+gram+LARFT (warp0) + ASYNC-RING tcgen05 apply (warps0-2 ‖ warp3).

The full integration: cute_ring_chain's async warp-spec apply (G1 ring + G2/G3 warp-spec, NO per-K-tile
barrier) put into the per-matrix QR loop with the validated panel/gram/LARFT and RMW H-=dV. To keep the
phase bookkeeping simple across the nested super-panel × cc-stripe × output-tile loops, the ring mbarriers
are RE-INIT per cc-stripe (one cheap barrier/stripe — the per-K-tile barriers are what we removed).
Route n<=512 -> engine, else torch.geqrf.
Run: modal run modal_cute_lab.py::run_candidate_quick --script cute_qr_m6.py   (150s kill on hang).

STATUS (2026-06-28): CORRECT at n<=64 (single cc-stripe AND single G3 output-tile: n=32/48/64 PASS
rel ~1e-6). HEISENBUG at n>=128 (multi-stripe / multi-G3-tile): hangs WITHOUT device prints but PASSES
WITH a `cute.printf` (the print's timing delay masks a mbarrier-handshake race). gmem fences did NOT fix
it -> it's a TIMING race, not a gmem RAW. ROOT CAUSE = the per-cc-stripe mbarrier RE-INIT racing the
ring's async phase timing. FIX = init the mbarriers ONCE (no re-init) + RUNNING-PARITY phase bookkeeping
(pe[2],pf[2],pacc,pgf,pgd XORed per use, empty primed) across the nested super-panel × cc-stripe ×
output-tile loops. The ring mechanism itself is proven (cute_ring_g1 1.58x, cute_ring_chain correct).
"""
import torch
import cutlass
import cutlass.cute as cute
import cutlass.utils as utils
import cutlass.pipeline as pipeline
from cutlass.cute.nvgpu import tcgen05
import cutlass.utils.blackwell_helpers as sm100_utils
from cutlass.cute.runtime import from_dlpack

N_CUTE_MAX = 512
NB = 16
BW = 64
BM = 32
NBP = 64
io_dtype = cutlass.Float32
acc_dtype = cutlass.Float32
mma_inst_shape_mnk = (128, 256, 8)
mma_tiler_mnk = (128, 256, 32)
threads_per_cta = 128
ab_stages = 2
NPROD = 96


@cute.jit
def _wreduce(val: cutlass.Float32) -> cutlass.Float32:
    for i in cutlass.range_constexpr(5):
        val = val + cute.arch.shuffle_sync_bfly(val, offset=(1 << i))
    return val


@cute.jit
def _vlow(Hm, r: cutlass.Int32, col: cutlass.Int32) -> cutlass.Float32:
    return Hm[r, col] if r > col else (cutlass.Float32(1.0) if r == col else cutlass.Float32(0.0))


@cute.jit
def _hi(x: cutlass.Float32) -> cutlass.Float32:
    return (x.bitcast(cutlass.Int32) & cutlass.Int32(-8192)).bitcast(cutlass.Float32)


@cute.jit
def _panel_factor(Hm, tm, c0: cutlass.Int32, pend: cutlass.Int32, lane: cutlass.Int32, n: cutlass.Constexpr):
    col = c0
    while col < pend:
        alpha = Hm[col, col]
        partial = cutlass.Float32(0.0)
        r = col + 1 + lane
        while r < n:
            hv = Hm[r, col]; partial = partial + hv * hv; r = r + 32
        xnorm2 = _wreduce(partial)
        need = xnorm2 > 0.0
        normfull = cute.math.sqrt(alpha * alpha + xnorm2, fastmath=True)
        beta = -normfull if alpha >= 0.0 else normfull
        tau_j = (beta - alpha) / beta if need else cutlass.Float32(0.0)
        scale = 1.0 / (alpha - beta) if need else cutlass.Float32(0.0)
        r = col + 1 + lane
        while r < n:
            if need:
                Hm[r, col] = Hm[r, col] * scale
            r = r + 32
        if lane == 0:
            Hm[col, col] = beta if need else alpha
            tm[col] = tau_j
        if need:
            cc = col + 1
            while cc < pend:
                pw = cutlass.Float32(0.0)
                r = col + 1 + lane
                while r < n:
                    pw = pw + Hm[r, col] * Hm[r, cc]; r = r + 32
                w = _wreduce(pw) + Hm[col, cc]
                if lane == 0:
                    Hm[col, cc] = Hm[col, cc] - tau_j * w
                r = col + 1 + lane
                while r < n:
                    Hm[r, cc] = Hm[r, cc] - tau_j * Hm[r, col] * w; r = r + 32
                cc = cc + 1
        col = col + 1


@cute.jit
def _compute_gram(Hm, sG, c0: cutlass.Int32, lane: cutlass.Int32, n: cutlass.Constexpr):
    for i in cutlass.range_constexpr(NB):
        for j in cutlass.range_constexpr(NB):
            p = cutlass.Float32(0.0)
            r = c0 + lane
            while r < n:
                p = p + _vlow(Hm, r, c0 + i) * _vlow(Hm, r, c0 + j); r = r + 32
            g = _wreduce(p)
            if lane == 0:
                sG[i * NB + j] = g


@cute.jit
def _larft(sG, sT, tm, c0: cutlass.Int32, plen: cutlass.Int32, lane: cutlass.Int32):
    if lane == 0:
        for a in cutlass.range_constexpr(NB * NB):
            sT[a] = cutlass.Float32(0.0)
        sT[0] = tm[c0]
        for i in cutlass.range_constexpr(1, NB):
            if i < plen:
                ti = tm[c0 + i]
                for k in cutlass.range_constexpr(NB):
                    if k < i:
                        m = cutlass.Float32(0.0)
                        for l in cutlass.range_constexpr(NB):
                            if l < i:
                                m = m + sT[k * NB + l] * sG[l * NB + i]
                        sT[k * NB + i] = -ti * m
                sT[i * NB + i] = ti


@cute.struct
class SS:
    full_mbar: cute.struct.MemRange[cutlass.Int64, ab_stages]
    empty_mbar: cute.struct.MemRange[cutlass.Int64, ab_stages]
    acc_done: cute.struct.MemRange[cutlass.Int64, 1]
    g23_full: cute.struct.MemRange[cutlass.Int64, 1]
    g23_done: cute.struct.MemRange[cutlass.Int64, 1]
    tmem_holding_buf: cutlass.Int32
    sG: cute.struct.MemRange[cutlass.Float32, NB * NB]
    sT: cute.struct.MemRange[cutlass.Float32, NB * NB]


@cute.kernel
def _qr_kernel(tiled_mma: cute.TiledMma, mH, mtau, mW1, mW2, mDV, a_sl, b_sl,
               n: cutlass.Constexpr, nb: cutlass.Constexpr):
    bid, _, _ = cute.arch.block_idx()
    tidx, _, _ = cute.arch.thread_idx()
    warp = cute.arch.make_warp_uniform(cute.arch.warp_idx())
    lane = tidx % 32
    Hm = mH[bid, None, None]; tm = mtau[bid, None]
    W1m = mW1[bid, None, None]; W2m = mW2[bid, None, None]; DVm = mDV[bid, None, None]

    smem = cutlass.utils.SmemAllocator()
    st = smem.allocate(SS)
    sG = st.sG.get_tensor(cute.make_layout(NB * NB)); sT = st.sT.get_tensor(cute.make_layout(NB * NB))
    sVh = smem.allocate_tensor(io_dtype, a_sl.outer, 128, swizzle=a_sl.inner)
    sVl = smem.allocate_tensor(io_dtype, a_sl.outer, 128, swizzle=a_sl.inner)
    sCh = smem.allocate_tensor(io_dtype, b_sl.outer, 128, swizzle=b_sl.inner)
    sCl = smem.allocate_tensor(io_dtype, b_sl.outer, 128, swizzle=b_sl.inner)
    full0 = st.full_mbar.data_ptr(); empty0 = st.empty_mbar.data_ptr(); accd = st.acc_done.data_ptr()
    g23f = st.g23_full.data_ptr(); g23d = st.g23_done.data_ptr()

    tmem_bar = pipeline.NamedBarrier(barrier_id=1, num_threads=threads_per_cta)
    tmem = utils.TmemAllocator(st.tmem_holding_buf.ptr, barrier_for_retrieve=tmem_bar)
    tmem.allocate(512)
    cute.arch.barrier()

    thr_mma = tiled_mma.get_slice(0)
    tVh = tiled_mma.make_fragment_A(sVh); tVl = tiled_mma.make_fragment_A(sVl)
    tCh = tiled_mma.make_fragment_B(sCh); tCl = tiled_mma.make_fragment_B(sCl)
    acc_shape = tiled_mma.partition_shape_C(mma_tiler_mnk[:2])
    tCtAcc = tiled_mma.make_fragment_C(acc_shape)
    tmem.wait_for_alloc()
    tCtAcc = cute.make_tensor(tmem.retrieve_ptr(acc_dtype), tCtAcc.layout)
    subtile = 4
    epi_tiler = ((cute.size(tCtAcc, mode=[0, 0]), cute.size(tCtAcc, mode=[0, 1]) // subtile),)
    tCtAcc_epi = cute.zipped_divide(tCtAcc, epi_tiler)
    tmem_atom = cute.make_copy_atom(tcgen05.Ld32x32bOp(tcgen05.Repetition.x64), cutlass.Float32)
    tmem_tiled_copy = tcgen05.make_tmem_copy(tmem_atom, tCtAcc_epi[None, 0])
    tmem_thr_copy = tmem_tiled_copy.get_slice(tidx)
    tDtC = tmem_thr_copy.partition_S(tCtAcc_epi)
    nkb = cute.size(tVh, mode=[2])

    def dst_part(mOut):
        g = cute.local_tile(mOut, mma_tiler_mnk, (0, 0, None), proj=(1, 1, None))
        return tmem_thr_copy.partition_D(cute.zipped_divide(thr_mma.partition_C(g), epi_tiler))
    tDgW1 = dst_part(W1m); tDgW2 = dst_part(W2m); tDgDV = dst_part(DVm)
    rb = cute.make_rmem_tensor(tDgDV[None, None, 0].shape, acc_dtype)
    rbio = cute.make_rmem_tensor(tDgDV[None, None, 0].shape, io_dtype)

    c0 = cutlass.Int32(0)
    while c0 < n:
        pend = c0 + nb if c0 + nb < n else n
        plen = pend - c0
        if warp == 0:
            _panel_factor(Hm, tm, c0, pend, lane, n)
        cute.arch.barrier()
        if pend < n:
            if warp == 0:
                _compute_gram(Hm, sG, c0, lane, n)
            cute.arch.barrier()
            if warp == 0:
                _larft(sG, sT, tm, c0, plen, lane)
            cute.arch.barrier()

            cc = pend
            while cc < n:
                cend = cc + BW if cc + BW < n else n
                # ---- re-init ring mbarriers for this stripe ----
                if tidx == 0:
                    cute.arch.mbarrier_init(full0 + 0, NPROD); cute.arch.mbarrier_init(full0 + 1, NPROD)
                    cute.arch.mbarrier_init(empty0 + 0, 1); cute.arch.mbarrier_init(empty0 + 1, 1)
                    cute.arch.mbarrier_init(accd + 0, 1)
                    cute.arch.mbarrier_init(g23f + 0, NPROD); cute.arch.mbarrier_init(g23d + 0, 1)
                    cute.arch.mbarrier_arrive(empty0 + 0); cute.arch.mbarrier_arrive(empty0 + 1)
                cute.arch.mbarrier_init_fence()
                cute.arch.barrier()

                # ===== G1 RING: W1 = V^T C[:,cc:cend], K-loop rows [c0,n) =====
                num_kt = (n - c0 + BM - 1) // BM
                if warp < 3:
                    kt = 0
                    while kt < num_kt:
                        s = kt % 2; ph = (kt // 2) % 2
                        cute.arch.mbarrier_wait(empty0 + s, ph)
                        r0 = c0 + kt * BM
                        i = tidx
                        while i < NB * 32:
                            ii = i // 32; kk = i - ii * 32; kb = kk // 8; ki = kk - kb * 8; r = r0 + kk
                            v = cutlass.Float32(0.0)
                            if r < n:
                                v = _vlow(Hm, r, c0 + ii)
                            h = _hi(v); sVh[(ii, ki), 0, kb, s] = h; sVl[(ii, ki), 0, kb, s] = v - h
                            i = i + NPROD
                        i = tidx
                        while i < BW * 32:
                            ww = i // 32; kk = i - ww * 32; kb = kk // 8; ki = kk - kb * 8
                            r = r0 + kk; col = cc + ww
                            cv = cutlass.Float32(0.0)
                            if r < n and col < cend:
                                cv = Hm[r, col]
                            h = _hi(cv); sCh[(ww, ki), 0, kb, s] = h; sCl[(ww, ki), 0, kb, s] = cv - h
                            i = i + NPROD
                        cute.arch.fence_view_async_shared()
                        cute.arch.mbarrier_arrive(full0 + s)
                        kt = kt + 1
                if warp == 3:
                    tiled_mma.set(tcgen05.Field.ACCUMULATE, False)
                    kt = 0
                    while kt < num_kt:
                        s = kt % 2; ph = (kt // 2) % 2
                        cute.arch.mbarrier_wait(full0 + s, ph)
                        for kb in cutlass.range_constexpr(nkb):
                            kc = (None, None, kb, s)
                            cute.gemm(tiled_mma, tCtAcc, tVh[kc], tCh[kc], tCtAcc)
                            tiled_mma.set(tcgen05.Field.ACCUMULATE, True)
                            cute.gemm(tiled_mma, tCtAcc, tVh[kc], tCl[kc], tCtAcc)
                            cute.gemm(tiled_mma, tCtAcc, tVl[kc], tCh[kc], tCtAcc)
                        if lane == 0:
                            tcgen05.commit(empty0 + s)
                        kt = kt + 1
                    if lane == 0:
                        tcgen05.commit(accd + 0)
                cute.arch.mbarrier_wait(accd + 0, 0)
                cute.copy(tmem_tiled_copy, tDtC[None, None, 0], rb)
                rbio.store(rb.load().to(io_dtype))
                cute.autovec_copy(rbio, tDgW1[None, None, 0])
                cute.arch.fence_acq_rel_cta()
                cute.arch.barrier()

                # ===== G2: W2 = T^T W1 (warp-spec single tile) =====
                if warp < 3:
                    i = tidx
                    while i < NB * 32:
                        ii = i // 32; k = i - ii * 32; kb = k // 8; ki = k - kb * 8
                        a = sT[k * NB + ii] if k < NB else cutlass.Float32(0.0)
                        h = _hi(a); sVh[(ii, ki), 0, kb, 0] = h; sVl[(ii, ki), 0, kb, 0] = a - h
                        i = i + NPROD
                    i = tidx
                    while i < BW * 32:
                        ww = i // 32; k = i - ww * 32; kb = k // 8; ki = k - kb * 8
                        b = W1m[k, ww] if k < NB else cutlass.Float32(0.0)
                        h = _hi(b); sCh[(ww, ki), 0, kb, 0] = h; sCl[(ww, ki), 0, kb, 0] = b - h
                        i = i + NPROD
                    cute.arch.fence_view_async_shared()
                    cute.arch.mbarrier_arrive(g23f + 0)
                if warp == 3:
                    cute.arch.mbarrier_wait(g23f + 0, 0)
                    tiled_mma.set(tcgen05.Field.ACCUMULATE, False)
                    for kb in cutlass.range_constexpr(nkb):
                        kc = (None, None, kb, 0)
                        cute.gemm(tiled_mma, tCtAcc, tVh[kc], tCh[kc], tCtAcc)
                        tiled_mma.set(tcgen05.Field.ACCUMULATE, True)
                        cute.gemm(tiled_mma, tCtAcc, tVh[kc], tCl[kc], tCtAcc)
                        cute.gemm(tiled_mma, tCtAcc, tVl[kc], tCh[kc], tCtAcc)
                    if lane == 0:
                        tcgen05.commit(g23d + 0)
                cute.arch.mbarrier_wait(g23d + 0, 0)
                cute.copy(tmem_tiled_copy, tDtC[None, None, 0], rb)
                rbio.store(rb.load().to(io_dtype))
                cute.autovec_copy(rbio, tDgW2[None, None, 0])
                cute.arch.fence_acq_rel_cta()
                cute.arch.barrier()

                # ===== G3: dV = V W2 (output-M tiles); RMW H -= dV. Simple all-128 + warp3 MMA =====
                # W2 (B-operand) is constant across output tiles -> fill sCh/sCl ONCE.
                i = tidx
                while i < BW * 32:
                    ww = i // 32; k = i - ww * 32; kb = k // 8; ki = k - kb * 8
                    b = W2m[k, ww] if k < NB else cutlass.Float32(0.0)
                    h = _hi(b); sCh[(ww, ki), 0, kb, 0] = h; sCl[(ww, ki), 0, kb, 0] = b - h
                    i = i + threads_per_cta
                cute.arch.barrier()
                r0 = c0
                while r0 < n:
                    if tidx == 0:                          # fresh g23d (phase 0) per output tile
                        cute.arch.mbarrier_init(g23d + 0, 1)
                    cute.arch.mbarrier_init_fence()
                    cute.arch.barrier()
                    i = tidx
                    while i < NBP * 32:
                        rr = i // 32; k = i - rr * 32; kb = k // 8; ki = k - kb * 8; r = r0 + rr
                        a = cutlass.Float32(0.0)
                        if k < NB and r < n:
                            a = _vlow(Hm, r, c0 + k)
                        h = _hi(a); sVh[(rr, ki), 0, kb, 0] = h; sVl[(rr, ki), 0, kb, 0] = a - h
                        i = i + threads_per_cta
                    cute.arch.barrier()
                    if warp == 3:
                        tiled_mma.set(tcgen05.Field.ACCUMULATE, False)
                        for kb in cutlass.range_constexpr(nkb):
                            kc = (None, None, kb, 0)
                            cute.gemm(tiled_mma, tCtAcc, tVh[kc], tCh[kc], tCtAcc)
                            tiled_mma.set(tcgen05.Field.ACCUMULATE, True)
                            cute.gemm(tiled_mma, tCtAcc, tVh[kc], tCl[kc], tCtAcc)
                            cute.gemm(tiled_mma, tCtAcc, tVl[kc], tCh[kc], tCtAcc)
                        if lane == 0:
                            tcgen05.commit(g23d + 0)
                    cute.arch.mbarrier_wait(g23d + 0, 0)
                    cute.copy(tmem_tiled_copy, tDtC[None, None, 0], rb)
                    rbio.store(rb.load().to(io_dtype))
                    cute.autovec_copy(rbio, tDgDV[None, None, 0])
                    cute.arch.fence_acq_rel_cta()
                    cute.arch.barrier()
                    idx = tidx
                    while idx < NBP * BW:
                        rr = idx // BW; w = idx - rr * BW; r = r0 + rr; col = cc + w
                        if r < n and col < cend:
                            Hm[r, col] = Hm[r, col] - DVm[rr, w]
                        idx = idx + threads_per_cta
                    cute.arch.barrier()
                    r0 = r0 + NBP
                cc = cend
        c0 = c0 + nb

    pipeline.sync(barrier_id=1)
    tmem.free(tmem.retrieve_ptr(acc_dtype))


@cute.jit
def _qr_host(mH, mtau, mW1, mW2, mDV, n: cutlass.Constexpr, nb: cutlass.Constexpr):
    op = tcgen05.MmaTF32Op(mma_inst_shape_mnk, tcgen05.CtaGroup.ONE, tcgen05.OperandSource.SMEM,
                           tcgen05.OperandMajorMode.K, tcgen05.OperandMajorMode.K)
    tiled_mma = cute.make_tiled_mma(op)
    a_sl = sm100_utils.make_smem_layout_a(tiled_mma, mma_tiler_mnk, io_dtype, ab_stages)
    b_sl = sm100_utils.make_smem_layout_b(tiled_mma, mma_tiler_mnk, io_dtype, ab_stages)
    B = cute.size(mH, mode=[0])
    _qr_kernel(tiled_mma, mH, mtau, mW1, mW2, mDV, a_sl, b_sl, n, nb).launch(
        grid=[B, 1, 1], block=[threads_per_cta, 1, 1])


_compiled = {}


def _scr(B, dev):
    return tuple(torch.zeros(B, 128, 256, device=dev, dtype=torch.float32) for _ in range(3))


def _run_cute(H, tau, scr):
    n = H.shape[-1]
    Ht = from_dlpack(H).mark_layout_dynamic(); tt = from_dlpack(tau).mark_layout_dynamic()
    Tn = lambda x: from_dlpack(x).mark_layout_dynamic()
    W1, W2, DV = scr
    key = (n, H.shape[0])
    if key not in _compiled:
        _compiled[key] = cute.compile(_qr_host, Ht, tt, Tn(W1), Tn(W2), Tn(DV), n, NB)
    _compiled[key](Ht, tt, Tn(W1), Tn(W2), Tn(DV))


def custom_kernel(data):
    A = data; B, n, _ = A.shape
    if n <= N_CUTE_MAX:
        try:
            H = A.clone().contiguous(); tau = torch.zeros(B, n, device=A.device, dtype=A.dtype)
            _run_cute(H, tau, _scr(B, A.device)); torch.cuda.synchronize(); return H, tau
        except Exception:
            pass
    return torch.geqrf(A)


def _time(fn, it=15):
    for _ in range(3):
        fn()
    torch.cuda.synchronize()
    ts = []
    for _ in range(it):
        torch.empty((8, 1024, 1024), dtype=torch.int64, device="cuda").fill_(0)
        torch.cuda.synchronize()
        s = torch.cuda.Event(enable_timing=True); e = torch.cuda.Event(enable_timing=True)
        s.record(); fn(); e.record(); torch.cuda.synchronize()
        ts.append(s.elapsed_time(e))
    ts.sort(); return ts[len(ts) // 2] * 1000


if __name__ == "__main__":
    import traceback
    print("=== M6 engine (CORRECT n<=64; heisenbug n>=128, see docstring) ===")
    torch.manual_seed(0)
    for nn in [32, 48, 64]:
        print(f"--- n={nn} ---")
        A = torch.randn(1, nn, nn, device="cuda", dtype=torch.float32)
        try:
            H = A.clone().contiguous(); tau = torch.zeros(1, nn, device="cuda", dtype=torch.float32)
            _run_cute(H, tau, _scr(1, "cuda")); torch.cuda.synchronize()
            Hg, _ = torch.geqrf(A)
            he = (H - Hg).abs().max().item() / (Hg.abs().max().item() + 1e-30)
            print(f"  n={nn}: H={he:.2e}  {'PASS' if he < 1e-4 else 'FAIL'}")
        except Exception:
            print("EXC\n" + traceback.format_exc()[-1200:])
    print("DONE")
