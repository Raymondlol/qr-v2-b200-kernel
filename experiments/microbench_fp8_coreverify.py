"""VERIFY we actually hit B200's dedicated fp8 tensor cores in the low-precision
tests (the user's challenge). Two checks:
(A) PTX inspection: does our plain `tl.dot(fp8,fp8)` kernel emit native fp8 MMA
    (e4m3 in mma/wgmma/tcgen05) or does Triton UPCAST to fp16/tf32? Dump+grep PTX
    for the fp8 kernel vs the tf32x3 and fp16x3 kernels.
(B) torch._scaled_mm (cuBLAS fp8 = GUARANTEED dedicated cores) as the ground-truth
    fp8 ceiling: at a large friendly shape (must reach ~fp8 peak if cores work) AND
    at our real per-matrix trailing shape (the K-poor ceiling even with perfect
    cuBLAS fp8). If _scaled_mm >> our Triton fp8, our verdict measured a bad path.

Run: modal run modal_microbench.py --script microbench_fp8_coreverify.py
"""
import os, glob, re
os.environ["TRITON_CACHE_DIR"] = "/tmp/ttc_fp8verify"
import torch, triton, triton.language as tl
torch.set_grad_enabled(False)
torch.backends.cuda.matmul.allow_tf32 = True
DEV = "cuda"

def clear_l2(): torch.empty((32,1024,1024),dtype=torch.int64,device=DEV).fill_(0)
def time_fn(fn, it=30):
    for _ in range(6):
        try: fn()
        except Exception as e: return "ERR:"+type(e).__name__+":"+str(e)[:120]
    torch.cuda.synchronize(); ts=[]
    for _ in range(it):
        clear_l2(); torch.cuda.synchronize()
        s=torch.cuda.Event(enable_timing=True);e=torch.cuda.Event(enable_timing=True)
        s.record(); fn(); e.record(); torch.cuda.synchronize(); ts.append(s.elapsed_time(e))
    ts.sort(); return ts[len(ts)//2]*1000.0

# ---- kernels (2D, single program-grid友好) ----
@triton.jit
def _fp8(A,B,C,M,N,K,sam,sak,sbk,sbn,scm,scn,BM:tl.constexpr,BN:tl.constexpr,BK:tl.constexpr):
    pm=tl.program_id(0);pn=tl.program_id(1)
    rm=pm*BM+tl.arange(0,BM);rn=pn*BN+tl.arange(0,BN);rk=tl.arange(0,BK)
    ap=A+(rm[:,None]*sam+rk[None,:]*sak);bp=B+(rk[:,None]*sbk+rn[None,:]*sbn)
    acc=tl.zeros((BM,BN),dtype=tl.float32)
    for k0 in range(0,K,BK):
        a=tl.load(ap,mask=(rm[:,None]<M)&(rk[None,:]+k0<K));b=tl.load(bp,mask=(rk[:,None]+k0<K)&(rn[None,:]<N))
        acc+=tl.dot(a,b,out_dtype=tl.float32);ap+=BK*sak;bp+=BK*sbk
    tl.store(C+(rm[:,None]*scm+rn[None,:]*scn),acc,mask=(rm[:,None]<M)&(rn[None,:]<N))
@triton.jit
def _f16(A,B,C,M,N,K,sam,sak,sbk,sbn,scm,scn,BM:tl.constexpr,BN:tl.constexpr,BK:tl.constexpr):
    pm=tl.program_id(0);pn=tl.program_id(1)
    rm=pm*BM+tl.arange(0,BM);rn=pn*BN+tl.arange(0,BN);rk=tl.arange(0,BK)
    ap=A+(rm[:,None]*sam+rk[None,:]*sak);bp=B+(rk[:,None]*sbk+rn[None,:]*sbn)
    acc=tl.zeros((BM,BN),dtype=tl.float32)
    for k0 in range(0,K,BK):
        a=tl.load(ap,mask=(rm[:,None]<M)&(rk[None,:]+k0<K));b=tl.load(bp,mask=(rk[:,None]+k0<K)&(rn[None,:]<N))
        acc+=tl.dot(a,b,out_dtype=tl.float32);ap+=BK*sak;bp+=BK*sbk
    tl.store(C+(rm[:,None]*scm+rn[None,:]*scn),acc,mask=(rm[:,None]<M)&(rn[None,:]<N))
@triton.jit
def _tf32x3(A,B,C,M,N,K,sam,sak,sbk,sbn,scm,scn,BM:tl.constexpr,BN:tl.constexpr,BK:tl.constexpr):
    pm=tl.program_id(0);pn=tl.program_id(1)
    rm=pm*BM+tl.arange(0,BM);rn=pn*BN+tl.arange(0,BN);rk=tl.arange(0,BK)
    ap=A+(rm[:,None]*sam+rk[None,:]*sak);bp=B+(rk[:,None]*sbk+rn[None,:]*sbn)
    acc=tl.zeros((BM,BN),dtype=tl.float32)
    for k0 in range(0,K,BK):
        a=tl.load(ap,mask=(rm[:,None]<M)&(rk[None,:]+k0<K));b=tl.load(bp,mask=(rk[:,None]+k0<K)&(rn[None,:]<N))
        acc+=tl.dot(a,b,input_precision="tf32x3");ap+=BK*sak;bp+=BK*sbk
    tl.store(C+(rm[:,None]*scm+rn[None,:]*scn),acc,mask=(rm[:,None]<M)&(rn[None,:]<N))

def all_ptx_mma_report():
    files = glob.glob("/tmp/ttc_fp8verify/**/*.ptx", recursive=True)
    print(f"  found {len(files)} compiled .ptx in triton cache")
    seen = {}
    for fp in files:
        try: txt = open(fp).read()
        except Exception: continue
        mmas = sorted(set(re.findall(r'\b(?:wgmma\.mma_async|tcgen05\.mma|mma\.sync)[.\w]*', txt)))
        for m in mmas:
            seen[m] = seen.get(m, 0) + 1
    print("  ALL distinct MMA instructions emitted across the fp8/fp16/tf32x3 kernels:")
    for m in sorted(seen):
        tag = ""
        if 'e4m3' in m or 'e5m2' in m: tag = "  <-- NATIVE FP8 core"
        elif 'tf32' in m: tag = "  <-- tf32 core"
        elif 'f16' in m or 'bf16' in m: tag = "  <-- fp16/bf16 core"
        print(f"    {m}{tag}")
    if not seen:
        print("    (no recognizable mma.sync/wgmma/tcgen05 ops found — printing raw dtype hits)")
        blob = "".join(open(f).read() for f in files[:6])
        for t in ('e4m3','e5m2','tf32','.f16','bf16'):
            print(f"      token {t}: {'present' if t in blob else 'absent'}")

def run_kernel(kernel, A, B, prec=None, BM=128, BN=128, BK=64):
    M,K=A.shape;N=B.shape[1];C=torch.empty((M,N),device=DEV,dtype=torch.float32)
    kernel[(triton.cdiv(M,BM),triton.cdiv(N,BN))](A,B,C,M,N,K,*A.stride(),*B.stride(),*C.stride(),
        BM=BM,BN=BN,BK=BK,num_warps=4,num_stages=3);return C

def main():
    print("torch",torch.__version__,"triton",triton.__version__,"| dev",torch.cuda.get_device_name(0))
    cc=torch.cuda.get_device_capability(0); print("compute capability",cc)

    # 2D per-matrix trailing shape (one matrix of the b=640 batch) and a large friendly shape
    print("\n=== (A) PTX: did tl.dot(fp8) emit native fp8 MMA, or upcast? ===")
    M,K,N=512,128,384
    Af=torch.randn(M,K,device=DEV); Bf=torch.randn(K,N,device=DEV)
    A8=Af.clamp(-448,448).to(torch.float8_e4m3fn); B8=Bf.clamp(-448,448).to(torch.float8_e4m3fn)
    A16=Af.to(torch.float16); B16=Bf.to(torch.float16)
    run_kernel(_fp8,A8,B8); run_kernel(_f16,A16,B16); run_kernel(_tf32x3,Af,Bf)
    try: all_ptx_mma_report()
    except Exception as e: print("  ptx report failed:", type(e).__name__, str(e)[:100])

    print("\n=== (B) torch._scaled_mm (cuBLAS fp8 = dedicated cores) ceiling ===")
    def scaled_mm(a8,b8):
        s=torch.tensor(1.0,device=DEV)
        return torch._scaled_mm(a8, b8, scale_a=s, scale_b=s, out_dtype=torch.bfloat16)
    print(f"{'shape':18s} {'fp8_scaledmm':>13s} {'TFLOPs':>8s} {'%fp8pk':>7s}  {'tritfp8':>9s} {'cuTF32':>8s}")
    for name,M,K,N in [("per-matrix 512^",512,128,384),("medium 2048^",2048,2048,2048),("large 4096^",4096,4096,4096)]:
        a=torch.randn(M,K,device=DEV).clamp(-448,448).to(torch.float8_e4m3fn)
        bcol=torch.randn(N,K,device=DEV).clamp(-448,448).to(torch.float8_e4m3fn).t()  # [K,N] col-major
        flops=2.0*M*N*K
        ts=time_fn(lambda:scaled_mm(a,bcol))
        # our triton fp8 + cuBLAS tf32 for the same shape
        a8=torch.randn(M,K,device=DEV).clamp(-448,448).to(torch.float8_e4m3fn)
        b8=torch.randn(K,N,device=DEV).clamp(-448,448).to(torch.float8_e4m3fn)
        tt=time_fn(lambda:run_kernel(_fp8,a8,b8))
        af=torch.randn(M,K,device=DEV); bf=torch.randn(K,N,device=DEV)
        tc=time_fn(lambda:torch.matmul(af,bf))
        def tf(t): return flops/(t*1e-6)/1e12 if isinstance(t,float) else float('nan')
        def f(x): return f"{x:.1f}" if isinstance(x,float) else str(x)[:13]
        pk=1900.0
        print(f"{name:18s} {f(ts):>13s} {tf(ts):>8.0f} {100*tf(ts)/pk:>6.0f}% {f(tt):>9s} {f(tc):>8s}",flush=True)
    print("\nIf _scaled_mm reaches ~1500-2700 TFLOPs on large shapes -> cuBLAS DOES use fp8 cores.")
    print("If per-matrix _scaled_mm is also slow (~similar TFLOPs to triton fp8) -> the K-poor")
    print("batched shape is the wall, NOT the core/api: fp8's 4x can't manifest here.")

if __name__=="__main__":
    main()
