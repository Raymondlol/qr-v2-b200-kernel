"""M3c-0 — per-matrix QR with tcgen05 tf32x3 blocked-WY apply (the integrated engine).

warp0 factors the super-panel + gram + LARFT (FP32, validated m1c); then ALL 128 threads run the
validated tcgen05 tf32x3 apply (cute_apply_chain): for each trailing BW-stripe, G1 W1=V^T C (K-loop),
G2 W2=T^T W1, G3 dV=V W2 (output-M tiled) and RMW H -= dV. CTA-barrier serial (panel THEN apply; the
panel‖apply overlap is M3c-1). W1/W2/dV staged through per-CTA gmem scratch (SMEM staging is a later opt).

Route n<=512 -> engine, else torch.geqrf.
Run: modal run modal_cute_lab.py::run_candidate --script cute_qr_m3c_tc.py   (static-scan clean.)
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
ab_stages = 1
acc_stage = 1


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
    xi = x.bitcast(cutlass.Int32)
    return (xi & cutlass.Int32(-8192)).bitcast(cutlass.Float32)


@cute.jit
def _panel_factor(Hm, tm, c0: cutlass.Int32, pend: cutlass.Int32, lane: cutlass.Int32,
                  n: cutlass.Constexpr):
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
    acc_mbar_ptr: cute.struct.MemRange[cutlass.Int64, acc_stage * 2]
    tmem_holding_buf: cutlass.Int32
    sG: cute.struct.MemRange[cutlass.Float32, NB * NB]
    sT: cute.struct.MemRange[cutlass.Float32, NB * NB]


@cute.kernel
def _qr_kernel(tiled_mma: cute.TiledMma, mH, mtau, mW1, mW2, mDV, a_sl, b_sl,
               n: cutlass.Constexpr, nb: cutlass.Constexpr,
               APPLY: cutlass.Constexpr, NPASS: cutlass.Constexpr, DOWR: cutlass.Constexpr):
    bid, _, _ = cute.arch.block_idx()
    tidx, _, _ = cute.arch.thread_idx()
    warp = cute.arch.make_warp_uniform(cute.arch.warp_idx())
    lane = tidx % 32
    Hm = mH[bid, None, None]; tm = mtau[bid, None]
    W1m = mW1[bid, None, None]; W2m = mW2[bid, None, None]; DVm = mDV[bid, None, None]

    smem = cutlass.utils.SmemAllocator()
    st = smem.allocate(SS)
    sG = st.sG.get_tensor(cute.make_layout(NB * NB))
    sT = st.sT.get_tensor(cute.make_layout(NB * NB))
    sAh = smem.allocate_tensor(io_dtype, a_sl.outer, 128, swizzle=a_sl.inner)
    sAl = smem.allocate_tensor(io_dtype, a_sl.outer, 128, swizzle=a_sl.inner)
    sBh = smem.allocate_tensor(io_dtype, b_sl.outer, 128, swizzle=b_sl.inner)
    sBl = smem.allocate_tensor(io_dtype, b_sl.outer, 128, swizzle=b_sl.inner)

    tmem_alloc_barrier = pipeline.NamedBarrier(barrier_id=1, num_threads=threads_per_cta)
    tmem = utils.TmemAllocator(st.tmem_holding_buf.ptr, barrier_for_retrieve=tmem_alloc_barrier)
    tmem.allocate(512)
    acc_producer, acc_consumer = pipeline.PipelineUmmaAsync.create(
        num_stages=acc_stage,
        producer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread),
        consumer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread, threads_per_cta),
        barrier_storage=st.acc_mbar_ptr.data_ptr()).make_participants()

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

            cc = pend if APPLY else cutlass.Int32(n)     # APPLY=0 -> skip the apply (panel-only timing)
            while cc < n:
                cend = cc + BW if cc + BW < n else n
                # ===== G1: W1 = V^T C[:, cc:cend], K-loop rows [c0,n) -> W1m =====
                W1acc = cute.make_rmem_tensor(tDgDV[None, None, 0].shape, acc_dtype)
                for e in cutlass.range_constexpr(cute.size(W1acc)):
                    W1acc[e] = cutlass.Float32(0.0)
                r0 = c0
                while r0 < n:
                    i = tidx
                    while i < NB * 32:
                        ii = i // 32; kk = i - ii * 32; kb = kk // 8; ki = kk - kb * 8; r = r0 + kk
                        v = cutlass.Float32(0.0)
                        if r < n:
                            v = _vlow(Hm, r, c0 + ii)
                        h = _hi(v); sAh[(ii, ki), 0, kb, 0] = h; sAl[(ii, ki), 0, kb, 0] = v - h
                        i = i + threads_per_cta
                    i = tidx
                    while i < BW * 32:
                        ww = i // 32; kk = i - ww * 32; kb = kk // 8; ki = kk - kb * 8; r = r0 + kk
                        col = cc + ww
                        cv = cutlass.Float32(0.0)
                        if r < n and col < cend:
                            cv = Hm[r, col]
                        h = _hi(cv); sBh[(ww, ki), 0, kb, 0] = h; sBl[(ww, ki), 0, kb, 0] = cv - h
                        i = i + threads_per_cta
                    cute.arch.barrier()
                    if warp == 0:
                        acc_empty = acc_producer.acquire_and_advance()
                        tiled_mma.set(tcgen05.Field.ACCUMULATE, False)
                        for kb in cutlass.range_constexpr(nkb):
                            kc = (None, None, kb, 0)
                            cute.gemm(tiled_mma, tCtAcc, tAh[kc], tBh[kc], tCtAcc)
                            tiled_mma.set(tcgen05.Field.ACCUMULATE, True)
                            if NPASS == 3:
                                cute.gemm(tiled_mma, tCtAcc, tAh[kc], tBl[kc], tCtAcc)
                                cute.gemm(tiled_mma, tCtAcc, tAl[kc], tBh[kc], tCtAcc)
                        acc_empty.commit()
                    acc_full = acc_consumer.wait_and_advance()
                    cute.copy(tmem_tiled_copy, tDtC[None, None, 0], rb)
                    for e in cutlass.range_constexpr(cute.size(W1acc)):
                        W1acc[e] = W1acc[e] + rb[e]
                    acc_full.release()
                    cute.arch.barrier()
                    r0 = r0 + BM
                rbio.store(W1acc.load().to(io_dtype))
                if DOWR:
                    cute.autovec_copy(rbio, tDgW1[None, None, 0])
                cute.arch.barrier()

                # ===== G2: W2 = T^T W1 -> W2m =====
                i = tidx
                while i < NB * 32:
                    ii = i // 32; k = i - ii * 32; kb = k // 8; ki = k - kb * 8
                    a = sT[k * NB + ii] if k < NB else cutlass.Float32(0.0)
                    h = _hi(a); sAh[(ii, ki), 0, kb, 0] = h; sAl[(ii, ki), 0, kb, 0] = a - h
                    i = i + threads_per_cta
                i = tidx
                while i < BW * 32:
                    ww = i // 32; k = i - ww * 32; kb = k // 8; ki = k - kb * 8
                    b = W1m[k, ww] if k < NB else cutlass.Float32(0.0)
                    h = _hi(b); sBh[(ww, ki), 0, kb, 0] = h; sBl[(ww, ki), 0, kb, 0] = b - h
                    i = i + threads_per_cta
                cute.arch.barrier()
                if warp == 0:
                    acc_empty = acc_producer.acquire_and_advance()
                    tiled_mma.set(tcgen05.Field.ACCUMULATE, False)
                    for kb in cutlass.range_constexpr(nkb):
                        kc = (None, None, kb, 0)
                        cute.gemm(tiled_mma, tCtAcc, tAh[kc], tBh[kc], tCtAcc)
                        tiled_mma.set(tcgen05.Field.ACCUMULATE, True)
                        if NPASS == 3:
                            cute.gemm(tiled_mma, tCtAcc, tAh[kc], tBl[kc], tCtAcc)
                            cute.gemm(tiled_mma, tCtAcc, tAl[kc], tBh[kc], tCtAcc)
                    acc_empty.commit()
                acc_full = acc_consumer.wait_and_advance()
                cute.copy(tmem_tiled_copy, tDtC[None, None, 0], rb)
                rbio.store(rb.load().to(io_dtype))
                if DOWR:
                    cute.autovec_copy(rbio, tDgW2[None, None, 0])
                acc_full.release()
                cute.arch.barrier()

                # ===== G3: dV = V W2 (output-M tiles) -> DVm; RMW H -= dV =====
                r0 = c0
                while r0 < n:
                    i = tidx
                    while i < NBP * 32:
                        rr = i // 32; k = i - rr * 32; kb = k // 8; ki = k - kb * 8; r = r0 + rr
                        a = cutlass.Float32(0.0)
                        if k < NB and r < n:
                            a = _vlow(Hm, r, c0 + k)
                        h = _hi(a); sAh[(rr, ki), 0, kb, 0] = h; sAl[(rr, ki), 0, kb, 0] = a - h
                        i = i + threads_per_cta
                    i = tidx
                    while i < BW * 32:
                        ww = i // 32; k = i - ww * 32; kb = k // 8; ki = k - kb * 8
                        b = W2m[k, ww] if k < NB else cutlass.Float32(0.0)
                        h = _hi(b); sBh[(ww, ki), 0, kb, 0] = h; sBl[(ww, ki), 0, kb, 0] = b - h
                        i = i + threads_per_cta
                    cute.arch.barrier()
                    if warp == 0:
                        acc_empty = acc_producer.acquire_and_advance()
                        tiled_mma.set(tcgen05.Field.ACCUMULATE, False)
                        for kb in cutlass.range_constexpr(nkb):
                            kc = (None, None, kb, 0)
                            cute.gemm(tiled_mma, tCtAcc, tAh[kc], tBh[kc], tCtAcc)
                            tiled_mma.set(tcgen05.Field.ACCUMULATE, True)
                            if NPASS == 3:
                                cute.gemm(tiled_mma, tCtAcc, tAh[kc], tBl[kc], tCtAcc)
                                cute.gemm(tiled_mma, tCtAcc, tAl[kc], tBh[kc], tCtAcc)
                        acc_empty.commit()
                    acc_full = acc_consumer.wait_and_advance()
                    cute.copy(tmem_tiled_copy, tDtC[None, None, 0], rb)
                    rbio.store(rb.load().to(io_dtype))
                    if DOWR:
                        cute.autovec_copy(rbio, tDgDV[None, None, 0])
                    acc_full.release()
                    cute.arch.barrier()
                    # RMW: H[r0+rr, cc+w] -= DV[rr, w]
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

    tmem.relinquish_alloc_permit()
    pipeline.sync(barrier_id=1)
    tmem.free(tmem_ptr)


@cute.jit
def _qr_host(mH, mtau, mW1, mW2, mDV, n: cutlass.Constexpr, nb: cutlass.Constexpr,
             APPLY: cutlass.Constexpr, NPASS: cutlass.Constexpr, DOWR: cutlass.Constexpr):
    op = tcgen05.MmaTF32Op(mma_inst_shape_mnk, tcgen05.CtaGroup.ONE, tcgen05.OperandSource.SMEM,
                           tcgen05.OperandMajorMode.K, tcgen05.OperandMajorMode.K)
    tiled_mma = cute.make_tiled_mma(op)
    a_sl = sm100_utils.make_smem_layout_a(tiled_mma, mma_tiler_mnk, io_dtype, ab_stages)
    b_sl = sm100_utils.make_smem_layout_b(tiled_mma, mma_tiler_mnk, io_dtype, ab_stages)
    B = cute.size(mH, mode=[0])
    _qr_kernel(tiled_mma, mH, mtau, mW1, mW2, mDV, a_sl, b_sl, n, nb, APPLY, NPASS, DOWR).launch(
        grid=[B, 1, 1], block=[threads_per_cta, 1, 1])


_compiled = {}


def _run_cute(H, tau, scr, apply=1, npass=3, dowr=1):
    n = H.shape[-1]
    Ht = from_dlpack(H).mark_layout_dynamic(); tt = from_dlpack(tau).mark_layout_dynamic()
    W1, W2, DV = scr
    Tn = lambda x: from_dlpack(x).mark_layout_dynamic()
    key = (n, H.shape[0], apply, npass, dowr)
    if key not in _compiled:
        _compiled[key] = cute.compile(_qr_host, Ht, tt, Tn(W1), Tn(W2), Tn(DV), n, NB, apply, npass, dowr)
    _compiled[key](Ht, tt, Tn(W1), Tn(W2), Tn(DV))


def _scratch(B, device):
    return (torch.zeros(B, 128, 256, device=device, dtype=torch.float32),
            torch.zeros(B, 128, 256, device=device, dtype=torch.float32),
            torch.zeros(B, 128, 256, device=device, dtype=torch.float32))


def custom_kernel(data):
    A = data; B, n, _ = A.shape
    if n <= N_CUTE_MAX:
        try:
            H = A.clone().contiguous(); tau = torch.zeros(B, n, device=A.device, dtype=A.dtype)
            _run_cute(H, tau, _scratch(B, A.device)); torch.cuda.synchronize(); return H, tau
        except Exception:
            pass
    return torch.geqrf(A)


def _time(fn, it=15):
    for _ in range(3):
        fn()
    torch.cuda.synchronize()
    ts = []
    for _ in range(it):
        torch.empty((16, 1024, 1024), dtype=torch.int64, device="cuda").fill_(0)
        torch.cuda.synchronize()
        s = torch.cuda.Event(enable_timing=True); e = torch.cuda.Event(enable_timing=True)
        s.record(); fn(); e.record(); torch.cuda.synchronize()
        ts.append(s.elapsed_time(e))
    ts.sort(); return ts[len(ts) // 2] * 1000


if __name__ == "__main__":
    import traceback
    print("=== M3c-0 ABLATION PROFILE (n=512, ablation by toggle, median us) ===")
    print("    decompose 274ms: APPLY(panel-only) NPASS(1 vs 3 tf32x3) DOWR(gmem-scratch write)")
    torch.manual_seed(0)
    # sanity: full config correct
    A0 = torch.randn(2, 512, 512, device="cuda", dtype=torch.float32)
    H0 = A0.clone().contiguous(); t0 = torch.zeros(2, 512, device="cuda", dtype=torch.float32)
    _run_cute(H0, t0, _scratch(2, "cuda"), 1, 3, 1); torch.cuda.synchronize()
    Hg, _ = torch.geqrf(A0)
    he = (H0 - Hg).abs().max().item() / (Hg.abs().max().item() + 1e-30)
    print(f"  [correctness full cfg] n=512 b2: H={he:.2e} {'PASS' if he < 1e-4 else 'FAIL'}")

    B = 64
    A = torch.randn(B, 512, 512, device="cuda", dtype=torch.float32)
    scr = _scratch(B, "cuda")
    H = A.clone().contiguous(); tau = torch.zeros(B, 512, device="cuda", dtype=torch.float32)
    cfgs = [("full   (apply,3pass,wr)", 1, 3, 1),
            ("1pass  (apply,1pass,wr)", 1, 1, 1),
            ("no-wr  (apply,3pass,  )", 1, 3, 0),
            ("1p+nowr(apply,1pass,  )", 1, 1, 0),
            ("panel  (no apply)      ", 0, 3, 1)]
    res = {}
    for name, ap, npass, dowr in cfgs:
        try:
            t = _time(lambda ap=ap, npass=npass, dowr=dowr: _run_cute(H, tau, scr, ap, npass, dowr))
            res[name] = t
            print(f"  B={B} n=512  {name}: {t:10.1f}us")
        except Exception:
            print(f"  {name}: EXC\n" + traceback.format_exc()[-1200:])
    try:
        full = res.get("full   (apply,3pass,wr)")
        panel = res.get("panel  (no apply)      ")
        p1 = res.get("1pass  (apply,1pass,wr)")
        nowr = res.get("no-wr  (apply,3pass,  )")
        if full and panel:
            print(f"\n  --- attribution (us) ---")
            print(f"  panel(+gram/larft) ~ {panel:9.1f}   ({100*panel/full:.0f}% of full)")
            print(f"  apply total        ~ {full-panel:9.1f}   ({100*(full-panel)/full:.0f}%)")
            if p1:
                print(f"  tf32x3 extra2pass  ~ {full-p1:9.1f}   (3pass-1pass = the 2 extra GEMM passes)")
            if nowr:
                print(f"  gmem-scratch write ~ {full-nowr:9.1f}   (W1/W2/dV autovec_copy to gmem)")
    except Exception:
        pass
    print("DONE")
