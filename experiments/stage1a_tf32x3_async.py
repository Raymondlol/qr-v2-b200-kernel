"""STAGE 1 increment A — a REAL tf32x3 trailing GEMM via async tcgen05.

The existing _async_trail_part is only a PROXY (NPASS copies of the SAME A@B summed, ref=3*A@B).
The trailing apply at n<=512 needs true tf32x3 (relerr ~1e-6). Build it as an async tcgen05
worker: split A->Ah,Al and B->Bh,Bl (tf32 hi/lo), then 3 dependent mma passes into one TMEM
acc with use_acc chaining (Ah@Bh, +Ah@Bl, +Al@Bh) and one mbarrier(count=3). This is the
trailing partition's real compute for design A; must be (a) correct vs fp64 / matching _bmm3,
(b) not much slower than the current fused tf32x3 Triton _bmm3. No banned substrings.
"""
import torch, triton
import triton.language as tl
from triton.experimental import gluon
from triton.experimental.gluon import language as gl
from triton.experimental.gluon.language.nvidia.blackwell import (
    allocate_tensor_memory, tcgen05_mma, TensorMemoryLayout, mbarrier,
    fence_async_shared, get_tmem_reg_layout,
)
from triton.tools.triton_to_gluon_translater.translator_helpers import (
    get_shared_memory_mma_operand, default_blocked_layout,
)


@gluon.jit
def _split_hi(x):
    # tf32 high part: clear the low 13 mantissa bits (mask -8192 = 0xFFFFE000), bitcast back.
    xi = x.to(gl.int32, bitcast=True)
    hi = (xi & -8192).to(gl.float32, bitcast=True)
    return hi


@gluon.jit
def _tf32x3_gemm(A, B, C, sab, sam, sak, sbb, sbk, sbn, scb, scm, scn,
                 M, N, K,
                 BM: gl.constexpr, BN: gl.constexpr, TK: gl.constexpr):
    pid_b = gl.program_id(0); pid_m = gl.program_id(1); pid_n = gl.program_id(2)
    a_blk: gl.constexpr = default_blocked_layout([BM, TK], gl.num_warps())
    b_blk: gl.constexpr = default_blocked_layout([TK, BN], gl.num_warps())
    rm = pid_m * BM + gl.arange(0, BM, layout=gl.SliceLayout(1, a_blk))[:, None]
    rka = gl.arange(0, TK, layout=gl.SliceLayout(0, a_blk))[None, :]
    rkb = gl.arange(0, TK, layout=gl.SliceLayout(1, b_blk))[:, None]
    rn = pid_n * BN + gl.arange(0, BN, layout=gl.SliceLayout(0, b_blk))[None, :]
    a = gl.load(A + pid_b * sab + rm * sam + rka * sak, mask=(rm < M) & (rka < K), other=0.0)
    b = gl.load(B + pid_b * sbb + rkb * sbk + rn * sbn, mask=(rkb < K) & (rn < N), other=0.0)
    ah = _split_hi(a); al = a - ah
    bh = _split_hi(b); bl = b - bh
    ah_s = get_shared_memory_mma_operand(ah, 0, False)
    al_s = get_shared_memory_mma_operand(al, 0, False)
    bh_s = get_shared_memory_mma_operand(bh, 1, False)
    bl_s = get_shared_memory_mma_operand(bl, 1, False)
    m: gl.constexpr = 128 if BM >= 128 else 64
    n: gl.constexpr = 256 if BN >= 256 else BN
    col_stride: gl.constexpr = 32 // gl.float32.primitive_bitwidth
    acc_layout: gl.constexpr = TensorMemoryLayout([m, n], col_stride=col_stride)
    reg_layout: gl.constexpr = get_tmem_reg_layout(gl.float32, (BM, BN), acc_layout, gl.num_warps())
    acc0 = gl.zeros([BM, BN], gl.float32, layout=reg_layout)
    acc_tmem = allocate_tensor_memory(gl.float32, [BM, BN], acc_layout, acc0)
    fence_async_shared()
    bar = gl.allocate_shared_memory(gl.int64, [1], mbarrier.MBarrierLayout())
    mbarrier.init(bar, count=3)
    tcgen05_mma(ah_s, bh_s, acc_tmem, use_acc=False, mbarriers=[bar])
    tcgen05_mma(ah_s, bl_s, acc_tmem, use_acc=True, mbarriers=[bar])
    tcgen05_mma(al_s, bh_s, acc_tmem, use_acc=True, mbarriers=[bar])
    mbarrier.wait(bar, phase=0)
    mbarrier.invalidate(bar)
    out = acc_tmem.load(reg_layout)
    c_blk: gl.constexpr = default_blocked_layout([BM, BN], gl.num_warps())
    out = gl.convert_layout(out, c_blk)
    cm = pid_m * BM + gl.arange(0, BM, layout=gl.SliceLayout(1, c_blk))[:, None]
    cn = pid_n * BN + gl.arange(0, BN, layout=gl.SliceLayout(0, c_blk))[None, :]
    gl.store(C + pid_b * scb + cm * scm + cn * scn, out, mask=(cm < M) & (cn < N))


def tf32x3_async(A, B, BM=64, BN=128, nw=8):
    Bb, M, K = A.shape; N = B.shape[2]
    C = torch.empty((Bb, M, N), device=A.device, dtype=torch.float32)
    grid = (Bb, triton.cdiv(M, BM), triton.cdiv(N, BN))
    k = _tf32x3_gemm[grid](A, B, C, *A.stride(), *B.stride(), *C.stride(),
                           M, N, K, BM=BM, BN=BN, TK=triton.next_power_of_2(K), num_warps=nw)
    return C, k


# ----- current-path tf32x3 reference (mirror submission._bmm3) -----
@triton.jit
def _bmm_x3_kernel(A, B, C, M, N, K, sab, sam, sak, sbb, sbk, sbn, scb, scm, scn,
                   BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
    pid_b = tl.program_id(0); pid_m = tl.program_id(1); pid_n = tl.program_id(2)
    rm = pid_m * BM + tl.arange(0, BM); rn = pid_n * BN + tl.arange(0, BN); rk = tl.arange(0, BK)
    a_ptrs = A + pid_b * sab + (rm[:, None] * sam + rk[None, :] * sak)
    b_ptrs = B + pid_b * sbb + (rk[:, None] * sbk + rn[None, :] * sbn)
    acc = tl.zeros((BM, BN), dtype=tl.float32)
    for k0 in range(0, K, BK):
        a = tl.load(a_ptrs, mask=(rm[:, None] < M) & (rk[None, :] + k0 < K), other=0.0)
        b = tl.load(b_ptrs, mask=(rk[:, None] + k0 < K) & (rn[None, :] < N), other=0.0)
        acc += tl.dot(a, b, input_precision="tf32x3")
        a_ptrs += BK * sak; b_ptrs += BK * sbk
    c_ptrs = C + pid_b * scb + (rm[:, None] * scm + rn[None, :] * scn)
    tl.store(c_ptrs, acc, mask=(rm[:, None] < M) & (rn[None, :] < N))


def bmm3(A, B, BM=128, BN=64, BK=32, nw=4):
    Bb, M, K = A.shape; N = B.shape[2]
    C = torch.empty((Bb, M, N), device=A.device, dtype=torch.float32)
    grid = (Bb, triton.cdiv(M, BM), triton.cdiv(N, BN))
    _bmm_x3_kernel[grid](A, B, C, M, N, K, *A.stride(), *B.stride(), *C.stride(),
                         BM=BM, BN=BN, BK=BK, num_warps=nw)
    return C


def clear_l2():
    torch.empty((32, 1024, 1024), dtype=torch.int64, device="cuda").fill_(0)


def time_fn(fn, it=50):
    for _ in range(6):
        try: fn()
        except Exception as e: return "ERR:" + repr(e)[:200]
    torch.cuda.synchronize()
    ts = []
    for _ in range(it):
        clear_l2(); torch.cuda.synchronize()
        s = torch.cuda.Event(enable_timing=True); e = torch.cuda.Event(enable_timing=True)
        s.record(); fn(); e.record(); torch.cuda.synchronize()
        ts.append(s.elapsed_time(e))
    ts.sort(); return ts[len(ts) // 2] * 1000


def main():
    print("dev", torch.cuda.get_device_name(0), "triton", triton.__version__)
    torch.manual_seed(0)

    print("\n=== correctness (async tf32x3 vs fp64 and vs _bmm3) ===")
    for Bb, M, K, N in [(8, 128, 128, 128), (8, 512, 128, 384), (8, 256, 16, 256)]:
        A = torch.randn(Bb, M, K, device="cuda")
        Bm = torch.randn(Bb, K, N, device="cuda")
        ref = (A.double() @ Bm.double()).float()
        try:
            C, k = tf32x3_async(A, Bm)
            torch.cuda.synchronize()
        except Exception as e:
            import traceback; print(f"  {Bb}x{M}x{K}x{N}: FAIL\n" + traceback.format_exc()[-1800:]); return
        Cb = bmm3(A, Bm)
        torch.cuda.synchronize()
        e_async = (C - ref).abs().max().item() / (ref.abs().max().item() + 1e-9)
        e_bmm3 = (Cb - ref).abs().max().item() / (ref.abs().max().item() + 1e-9)
        nreg = getattr(k, "n_regs", "?"); nsp = getattr(k, "n_spills", "?")
        print(f"  {Bb}x{M}x{K}x{N}: async relerr={e_async:.2e}  _bmm3 relerr={e_bmm3:.2e}  "
              f"n_regs={nreg} spills={nsp}  {'OK' if e_async < 5e-6 else 'CHECK'}")

    print("\n=== speed: fat trailing shape (B=640, M=512, K=128, N=384) ===")
    A = torch.randn(640, 512, 128, device="cuda")
    Bm = torch.randn(640, 128, 384, device="cuda")
    t_bmm3 = time_fn(lambda: bmm3(A, Bm))
    print(f"  _bmm3 (current tf32x3 Triton): {t_bmm3 if isinstance(t_bmm3,str) else f'{t_bmm3:.1f} us'}  (BASELINE)")
    for BM, BN in [(64, 128), (128, 128), (64, 256), (128, 256)]:
        t = time_fn(lambda BM=BM, BN=BN: tf32x3_async(A, Bm, BM=BM, BN=BN))
        if isinstance(t, str): print(f"  async BM={BM} BN={BN}: {t}"); continue
        r = t / t_bmm3 if not isinstance(t_bmm3, str) else float('nan')
        print(f"  async tcgen05 BM={BM} BN={BN}: {t:7.1f} us  {r:.2f}x bmm3")
    print("\nSTAGE 1A DONE (need: async relerr ~1e-6 AND speed <= ~1.3x _bmm3 -> trailing partition viable)")


if __name__ == "__main__":
    main()
