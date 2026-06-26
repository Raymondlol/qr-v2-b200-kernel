"""How K-rich must the trailing GEMM be to FEED the tensor cores? Fixed batch=640,
M=512, N=384; sweep K (= the block width NB = trailing contraction dim). Reports
realized TFLOPs + %-of-dtype-peak for cuBLAS-tf32, fused fp16x3, batched fp8.
Answers: (a) the feeding curve (efficiency vs K), (b) does fp8 ever overtake
tf32/fp16, and at what K. (In the real QR, bigger K = wider panel = more sequential
panel cost -- this microbench isolates only the GEMM-side payoff.)

Run: modal run modal_microbench.py --script microbench_feed_gemm.py
"""
import torch, triton, triton.language as tl
torch.set_grad_enabled(False)
torch.backends.cuda.matmul.allow_tf32 = True
DEV = "cuda"
PEAK = {"tf32": 482.0, "fp16": 965.0, "fp8": 1900.0}

def clear_l2(): torch.empty((32,1024,1024),dtype=torch.int64,device=DEV).fill_(0)
def time_fn(fn, it=30):
    for _ in range(6):
        try: fn()
        except Exception as e: return "ERR:"+type(e).__name__+":"+str(e)[:80]
    torch.cuda.synchronize(); ts=[]
    for _ in range(it):
        clear_l2(); torch.cuda.synchronize()
        s=torch.cuda.Event(enable_timing=True);e=torch.cuda.Event(enable_timing=True)
        s.record(); fn(); e.record(); torch.cuda.synchronize(); ts.append(s.elapsed_time(e))
    ts.sort(); return ts[len(ts)//2]*1000.0

@triton.autotune(configs=[triton.Config({'BM':bm,'BN':bn,'BK':bk},num_warps=nw,num_stages=ns)
    for bm in (64,128) for bn in (64,128) for bk in (32,64) for nw in (4,8) for ns in (3,4)
    if bm*bn*bk<=128*128*64], key=['M','N','K'])
@triton.jit
def _f16x3(A,B,C,M,N,K,sab,sam,sak,sbb,sbk,sbn,scb,scm,scn,BM:tl.constexpr,BN:tl.constexpr,BK:tl.constexpr):
    pb=tl.program_id(0);pm=tl.program_id(1);pn=tl.program_id(2)
    rm=pm*BM+tl.arange(0,BM);rn=pn*BN+tl.arange(0,BN);rk=tl.arange(0,BK)
    ap=A+pb*sab+(rm[:,None]*sam+rk[None,:]*sak);bp=B+pb*sbb+(rk[:,None]*sbk+rn[None,:]*sbn)
    acc=tl.zeros((BM,BN),dtype=tl.float32)
    for k0 in range(0,K,BK):
        a=tl.load(ap,mask=(rm[:,None]<M)&(rk[None,:]+k0<K),other=0.);b=tl.load(bp,mask=(rk[:,None]+k0<K)&(rn[None,:]<N),other=0.)
        ah=a.to(tl.float16);al=(a-ah.to(tl.float32)).to(tl.float16);bh=b.to(tl.float16);bl=(b-bh.to(tl.float32)).to(tl.float16)
        acc+=tl.dot(ah,bh,out_dtype=tl.float32)+tl.dot(ah,bl,out_dtype=tl.float32)+tl.dot(al,bh,out_dtype=tl.float32)
        ap+=BK*sak;bp+=BK*sbk
    tl.store(C+pb*scb+(rm[:,None]*scm+rn[None,:]*scn),acc,mask=(rm[:,None]<M)&(rn[None,:]<N))

@triton.autotune(configs=[triton.Config({'BM':bm,'BN':bn,'BK':bk},num_warps=nw,num_stages=ns)
    for bm in (64,128) for bn in (64,128) for bk in (64,128) for nw in (4,8) for ns in (3,4)
    if bm*bn*bk<=128*128*128], key=['M','N','K'])
@triton.jit
def _fp8(A,B,C,M,N,K,sab,sam,sak,sbb,sbk,sbn,scb,scm,scn,BM:tl.constexpr,BN:tl.constexpr,BK:tl.constexpr):
    pb=tl.program_id(0);pm=tl.program_id(1);pn=tl.program_id(2)
    rm=pm*BM+tl.arange(0,BM);rn=pn*BN+tl.arange(0,BN);rk=tl.arange(0,BK)
    ap=A+pb*sab+(rm[:,None]*sam+rk[None,:]*sak);bp=B+pb*sbb+(rk[:,None]*sbk+rn[None,:]*sbn)
    acc=tl.zeros((BM,BN),dtype=tl.float32)
    for k0 in range(0,K,BK):
        a=tl.load(ap,mask=(rm[:,None]<M)&(rk[None,:]+k0<K));b=tl.load(bp,mask=(rk[:,None]+k0<K)&(rn[None,:]<N))
        acc+=tl.dot(a,b,out_dtype=tl.float32);ap+=BK*sak;bp+=BK*sbk
    tl.store(C+pb*scb+(rm[:,None]*scm+rn[None,:]*scn),acc,mask=(rm[:,None]<M)&(rn[None,:]<N))

def run(kern,A,B):
    Bb,M,K=A.shape;N=B.shape[2];C=torch.empty((Bb,M,N),device=DEV,dtype=torch.float32)
    kern[lambda m:(Bb,triton.cdiv(M,m['BM']),triton.cdiv(N,m['BN']))](A,B,C,M,N,K,*A.stride(),*B.stride(),*C.stride());return C

def main():
    print("torch",torch.__version__,"| dev",torch.cuda.get_device_name(0))
    Bb,M,N=640,512,384
    print(f"batch={Bb} M={M} N={N}; sweep K (= trailing block width NB).")
    print(f"{'K':>5s} | {'tf32 us':>8s} {'TFLOP':>6s} {'%pk':>4s} | {'fp16x3 us':>9s} {'TFLOP':>6s} {'%pk':>4s} | "
          f"{'fp8 us':>7s} {'TFLOP':>6s} {'%pk':>4s} | {'fp8/tf32 eff':>12s}")
    for K in [32,64,128,256,512,1024]:
        A=torch.randn(Bb,M,K,device=DEV);B=torch.randn(Bb,K,N,device=DEV)
        A8=A.clamp(-448,448).to(torch.float8_e4m3fn);B8=B.clamp(-448,448).to(torch.float8_e4m3fn)
        flops=2.0*Bb*M*N*K
        t_tf=time_fn(lambda:torch.matmul(A,B))            # cuBLAS tf32 bmm
        t_16=time_fn(lambda:run(_f16x3,A,B))              # fused fp16x3 (=tf32x3-class accuracy)
        t_8=time_fn(lambda:run(_fp8,A8,B8))               # batched fp8 (1 dot)
        def tf(t): return flops/(t*1e-6)/1e12 if isinstance(t,float) else float('nan')
        def pk(t,d): return 100*tf(t)/PEAK[d] if isinstance(t,float) else float('nan')
        def f(x): return f"{x:.0f}" if isinstance(x,float) else str(x)[:8]
        # fp8 effective vs tf32: 1 fp8 dot's TFLOPs vs tf32's (both per-dot)
        eff = (tf(t_8)/tf(t_tf)) if (isinstance(t_8,float) and isinstance(t_tf,float)) else float('nan')
        print(f"{K:>5d} | {f(t_tf):>8s} {tf(t_tf):>6.0f} {pk(t_tf,'tf32'):>3.0f}% | "
              f"{f(t_16):>9s} {tf(t_16):>6.0f} {pk(t_16,'fp16'):>3.0f}% | "
              f"{f(t_8):>7s} {tf(t_8):>6.0f} {pk(t_8,'fp8'):>3.0f}% | {eff:>11.2f}x",flush=True)
    print("\n%pk = fraction of that dtype's dense B200 peak (tf32 482, fp16 965, fp8 1900 TFLOPs).")
    print("fp8/tf32 eff = realized fp8 TFLOPs / realized tf32 TFLOPs (one dot). >1 means fp8 faster/dot.")
    print("In QR: K=NB is the panel width; bigger K = more sequential panel cost (not shown here).")

if __name__=="__main__":
    main()
