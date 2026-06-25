"""Hard-engine recon: (A) what does the B200/Modal Triton toolchain support, and
(B) can ANY advanced pure-Triton tf32x3 GEMM beat the current fused kernel's ~10%
ceiling on the real trailing shapes -- or is the only path raw-PTX/TLX?

Tests on the dominant trailing shapes: n=512 fat (640,512,128,384) tf32x3-required,
n=1024 fat (60,1024,256,768). Variants: current-fused vs PERSISTENT (grid=#SMs,
tile-loop, better scheduling) vs persistent+big-tile vs (if available) TMA. Reports
TFLOP/s vs the tf32x3 ceiling (~peak/3) and vs cuBLAS-1xTF32. No banned substrings.
"""
import torch, triton, triton.language as tl

torch.backends.cuda.matmul.allow_tf32 = True
PEAK = 1100.0


def caps():
    print("=== CAPABILITY REPORT ===")
    print("torch", torch.__version__, "| triton", triton.__version__)
    cc = torch.cuda.get_device_capability(0)
    print("device", torch.cuda.get_device_name(0), "| compute capability", cc,
          "| #SMs", torch.cuda.get_device_properties(0).multi_processor_count)
    feats = ['make_tensor_descriptor', '_experimental_make_tensor_descriptor',
             '_experimental_descriptor_load', '_experimental_descriptor_store',
             'async_task', 'dot_scaled']
    have = {f: hasattr(tl, f) for f in feats}
    print("tl features:", have)
    # warp specialization / consumer groups (autotune/Config kwargs vary by version)
    import inspect
    try:
        sig = inspect.signature(triton.Config.__init__)
        print("triton.Config params:", list(sig.parameters))
    except Exception as e:
        print("Config sig err", e)
    for mod in ('triton_tlx', 'tlx', 'triton.tlx'):
        try:
            __import__(mod); print(f"IMPORT OK: {mod}")
        except Exception as e:
            print(f"import {mod}: {type(e).__name__}")
    print()


def clear_l2():
    torch.empty((32, 1024, 1024), dtype=torch.int64, device="cuda").fill_(0)


def time_fn(fn, iters=40):
    for _ in range(5):
        try: fn()
        except Exception as e: return ("ERR:" + type(e).__name__)
    torch.cuda.synchronize(); ts = []
    for _ in range(iters):
        clear_l2(); torch.cuda.synchronize()
        s = torch.cuda.Event(enable_timing=True); e = torch.cuda.Event(enable_timing=True)
        s.record(); fn(); e.record(); torch.cuda.synchronize(); ts.append(s.elapsed_time(e))
    ts.sort(); return ts[len(ts) // 2] * 1000.0


# ---- current fused tf32x3 (baseline) ----
@triton.autotune(configs=[
    triton.Config({'BM': 64, 'BN': 64, 'BK': 32}, num_warps=4, num_stages=3),
    triton.Config({'BM': 128, 'BN': 64, 'BK': 32}, num_warps=4, num_stages=3),
    triton.Config({'BM': 64, 'BN': 128, 'BK': 32}, num_warps=4, num_stages=3),
    triton.Config({'BM': 128, 'BN': 128, 'BK': 32}, num_warps=8, num_stages=3),
], key=['M', 'N', 'K'])
@triton.jit
def _fused(A, B, C, M, N, K, sab, sam, sak, sbb, sbk, sbn, scb, scm, scn,
          BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
    pb = tl.program_id(0); pm = tl.program_id(1); pn = tl.program_id(2)
    rm = pm * BM + tl.arange(0, BM); rn = pn * BN + tl.arange(0, BN); rk = tl.arange(0, BK)
    ap = A + pb * sab + (rm[:, None] * sam + rk[None, :] * sak)
    bp = B + pb * sbb + (rk[:, None] * sbk + rn[None, :] * sbn)
    acc = tl.zeros((BM, BN), dtype=tl.float32)
    for k0 in range(0, K, BK):
        a = tl.load(ap, mask=(rm[:, None] < M) & (rk[None, :] + k0 < K), other=0.0)
        b = tl.load(bp, mask=(rk[:, None] + k0 < K) & (rn[None, :] < N), other=0.0)
        acc += tl.dot(a, b, input_precision="tf32x3")
        ap += BK * sak; bp += BK * sbk
    cp = C + pb * scb + (rm[:, None] * scm + rn[None, :] * scn)
    tl.store(cp, acc, mask=(rm[:, None] < M) & (rn[None, :] < N))


def fused(A, B):
    Bb, M, K = A.shape; N = B.shape[2]
    C = torch.empty((Bb, M, N), device=A.device, dtype=torch.float32)
    grid = lambda m: (Bb, triton.cdiv(M, m['BM']), triton.cdiv(N, m['BN']))
    _fused[grid](A, B, C, M, N, K, *A.stride(), *B.stride(), *C.stride())
    return C


# ---- PERSISTENT tf32x3: grid = #SMs, each program strides over (batch*mtile*ntile) ----
def _persistent_factory(NSM):
    @triton.autotune(configs=[
        triton.Config({'BM': 64, 'BN': 128, 'BK': 32}, num_warps=4, num_stages=4),
        triton.Config({'BM': 128, 'BN': 128, 'BK': 32}, num_warps=8, num_stages=4),
        triton.Config({'BM': 128, 'BN': 128, 'BK': 64}, num_warps=8, num_stages=3),
        triton.Config({'BM': 64, 'BN': 256, 'BK': 32}, num_warps=8, num_stages=4),
        triton.Config({'BM': 128, 'BN': 256, 'BK': 64}, num_warps=8, num_stages=3),
    ], key=['M', 'N', 'K'])
    @triton.jit
    def _persist(A, B, C, M, N, K, NB_GRID, sab, sam, sak, sbb, sbk, sbn, scb, scm, scn,
                 BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr, NUM: tl.constexpr):
        pid = tl.program_id(0)
        mtiles = tl.cdiv(M, BM); ntiles = tl.cdiv(N, BN)
        tiles_per_b = mtiles * ntiles
        total = NB_GRID * tiles_per_b
        for t in range(pid, total, NUM):
            pb = t // tiles_per_b
            r = t % tiles_per_b
            pm = r // ntiles; pn = r % ntiles
            rm = pm * BM + tl.arange(0, BM); rn = pn * BN + tl.arange(0, BN); rk = tl.arange(0, BK)
            ap = A + pb * sab + (rm[:, None] * sam + rk[None, :] * sak)
            bp = B + pb * sbb + (rk[:, None] * sbk + rn[None, :] * sbn)
            acc = tl.zeros((BM, BN), dtype=tl.float32)
            for k0 in range(0, K, BK):
                a = tl.load(ap, mask=(rm[:, None] < M) & (rk[None, :] + k0 < K), other=0.0)
                b = tl.load(bp, mask=(rk[:, None] + k0 < K) & (rn[None, :] < N), other=0.0)
                acc += tl.dot(a, b, input_precision="tf32x3")
                ap += BK * sak; bp += BK * sbk
            cp = C + pb * scb + (rm[:, None] * scm + rn[None, :] * scn)
            tl.store(cp, acc, mask=(rm[:, None] < M) & (rn[None, :] < N))
    def run(A, B):
        Bb, M, K = A.shape; N = B.shape[2]
        C = torch.empty((Bb, M, N), device=A.device, dtype=torch.float32)
        _persist[(NUM_SMS,)](A, B, C, M, N, K, Bb, *A.stride(), *B.stride(), *C.stride(), NUM=NUM_SMS)
        return C
    return run


NUM_SMS = 148


def relerr(C, ref): return (C - ref).abs().max().item() / (ref.abs().max().item() + 1e-9)


def bench():
    print("=== ADVANCED tf32x3 GEMM on trailing shapes (vs ~367 TFLOPs tf32x3 ceiling) ===")
    persist = _persistent_factory(NUM_SMS)
    for name, Bb, M, K, N in [("n512 fat 640,512,128,384", 640, 512, 128, 384),
                              ("n1024 fat 60,1024,256,768", 60, 1024, 256, 768)]:
        A = torch.randn(Bb, M, K, device="cuda"); B = torch.randn(Bb, K, N, device="cuda")
        flops = 2.0 * Bb * M * N * K; ceil = PEAK / 3.0
        ref = torch.matmul(A.double(), B.double()).float()
        print(f"\n{name}:")
        for lbl, fn in [("fused (current)", fused), ("persistent", persist)]:
            t = time_fn(lambda fn=fn: fn(A, B))
            if isinstance(t, str): print(f"  {lbl:18s} {t}"); continue
            tf = flops / (t * 1e-6) / 1e12
            print(f"  {lbl:18s} {t:8.1f} us  {tf:6.0f} TFLOPs  {100*tf/ceil:4.0f}% ceil  relerr={relerr(fn(A,B),ref):.1e}")
        tcu = time_fn(lambda: torch.matmul(A, B)); tfu = flops/(tcu*1e-6)/1e12
        print(f"  {'cuBLAS-1xTF32':18s} {tcu:8.1f} us  {tfu:6.0f} TFLOPs  {100*tfu/PEAK:4.0f}% peak (loose precision)")


def main():
    caps()
    bench()


if __name__ == "__main__":
    main()
