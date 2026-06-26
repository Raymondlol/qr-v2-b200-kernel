"""#3: does mxfp8 (block-scaled tl.dot_scaled, tcgen05) get BETTER hardware
utilization than plain fp8 (tl.dot) on the real K-poor trailing shapes? HW research
said ~equal; this measures it directly. Single dot, same tiling, real shapes.
Compares: tf32x3 (ref) | fp16x3 (ref) | plain-fp8 tl.dot | mxfp8 tl.dot_scaled.
Captures full CompilationError if dot_scaled won't build.

Run: modal run modal_microbench.py --script microbench_mxfp8_vs_fp8.py
"""
import torch, triton, triton.language as tl
torch.set_grad_enabled(False)
torch.backends.cuda.matmul.allow_tf32 = True
DEV = "cuda"

def clear_l2(): torch.empty((32,1024,1024),dtype=torch.int64,device=DEV).fill_(0)
def time_fn(fn, it=30):
    for _ in range(6):
        try: fn()
        except Exception as e: return "ERR:"+type(e).__name__+":"+str(e)[:160]
    torch.cuda.synchronize(); ts=[]
    for _ in range(it):
        clear_l2(); torch.cuda.synchronize()
        s=torch.cuda.Event(enable_timing=True);e=torch.cuda.Event(enable_timing=True)
        s.record(); fn(); e.record(); torch.cuda.synchronize(); ts.append(s.elapsed_time(e))
    ts.sort(); return ts[len(ts)//2]*1000.0

# plain fp8 batched single dot
@triton.jit
def _fp8(A,B,C,M,N,K,sab,sam,sak,sbb,sbk,sbn,scb,scm,scn,
         BM:tl.constexpr,BN:tl.constexpr,BK:tl.constexpr):
    pb=tl.program_id(0);pm=tl.program_id(1);pn=tl.program_id(2)
    rm=pm*BM+tl.arange(0,BM);rn=pn*BN+tl.arange(0,BN);rk=tl.arange(0,BK)
    ap=A+pb*sab+(rm[:,None]*sam+rk[None,:]*sak);bp=B+pb*sbb+(rk[:,None]*sbk+rn[None,:]*sbn)
    acc=tl.zeros((BM,BN),dtype=tl.float32)
    for k0 in range(0,K,BK):
        a=tl.load(ap,mask=(rm[:,None]<M)&(rk[None,:]+k0<K));b=tl.load(bp,mask=(rk[:,None]+k0<K)&(rn[None,:]<N))
        acc+=tl.dot(a,b,out_dtype=tl.float32);ap+=BK*sak;bp+=BK*sbk
    tl.store(C+pb*scb+(rm[:,None]*scm+rn[None,:]*scn),acc,mask=(rm[:,None]<M)&(rn[None,:]<N))
def fp8dot(A8,B8,BM=128,BN=128,BK=64):
    Bb,M,K=A8.shape;N=B8.shape[2];C=torch.empty((Bb,M,N),device=DEV,dtype=torch.float32)
    _fp8[(Bb,triton.cdiv(M,BM),triton.cdiv(N,BN))](A8,B8,C,M,N,K,*A8.stride(),*B8.stride(),*C.stride(),
        BM=BM,BN=BN,BK=BK,num_warps=4,num_stages=3);return C

# mxfp8 block-scaled single dot (e8m0 group-32 scales)
@triton.jit
def _mxfp8(A,B,SA,SB,C,M,N,K,sab,sam,sak,sbb,sbk,sbn,ssam,ssak,ssbn,ssbk,scb,scm,scn,
          BM:tl.constexpr,BN:tl.constexpr,BK:tl.constexpr):
    pb=tl.program_id(0);pm=tl.program_id(1);pn=tl.program_id(2)
    rm=pm*BM+tl.arange(0,BM);rn=pn*BN+tl.arange(0,BN);rk=tl.arange(0,BK)
    rks=tl.arange(0,BK//32)
    acc=tl.zeros((BM,BN),dtype=tl.float32)
    for k0 in range(0,K,BK):
        ap=A+pb*sab+(rm[:,None]*sam+(rk[None,:]+k0)*sak)
        bp=B+pb*sbb+((rk[:,None]+k0)*sbk+rn[None,:]*sbn)
        a=tl.load(ap,mask=(rm[:,None]<M)&(rk[None,:]+k0<K));b=tl.load(bp,mask=(rk[:,None]+k0<K)&(rn[None,:]<N))
        # scales: SA[M, K//32] -> [BM, BK//32];  SB[N, K//32] -> [BN, BK//32]
        sap=SA+pb*(M*(K//32))+(rm[:,None]*(K//32)+(rks[None,:]+k0//32))
        sbp=SB+pb*(N*(K//32))+(rn[:,None]*(K//32)+(rks[None,:]+k0//32))
        sa=tl.load(sap,mask=rm[:,None]<M,other=127); sb=tl.load(sbp,mask=rn[:,None]<N,other=127)
        acc=tl.dot_scaled(a,sa,"e4m3",b,sb,"e4m3",acc)
    tl.store(C+pb*scb+(rm[:,None]*scm+rn[None,:]*scn),acc,mask=(rm[:,None]<M)&(rn[None,:]<N))
def mxfp8dot(A8,B8,SA,SB,M,N,K,BM=128,BN=128,BK=64):
    Bb=A8.shape[0];C=torch.empty((Bb,M,N),device=DEV,dtype=torch.float32)
    _mxfp8[(Bb,triton.cdiv(M,BM),triton.cdiv(N,BN))](A8,B8,SA,SB,C,M,N,K,*A8.stride(),*B8.stride(),
        SA.stride(1),SA.stride(2),SB.stride(1),SB.stride(2),*C.stride(),
        BM=BM,BN=BN,BK=BK,num_warps=4,num_stages=3);return C

def main():
    print("torch",torch.__version__,"| dev",torch.cuda.get_device_name(0),"| dot_scaled:",hasattr(tl,"dot_scaled"))
    shapes=[("FAT n512",640,512,128,384),("FAT n1024",60,1024,256,768)]
    print(f"{'shape':12s} {'B,M,K,N':>17s} {'fp8_1dot':>9s} {'mxfp8_1dot':>11s} {'mx/fp8':>7s}")
    for name,Bb,M,K,N in shapes:
        A=torch.randn(Bb,M,K,device=DEV);B=torch.randn(Bb,K,N,device=DEV)
        A8=(A.clamp(-448,448)).to(torch.float8_e4m3fn);B8=(B.clamp(-448,448)).to(torch.float8_e4m3fn)
        SA=torch.full((Bb,M,K//32),127,dtype=torch.uint8,device=DEV)
        SB=torch.full((Bb,N,K//32),127,dtype=torch.uint8,device=DEV)
        t8=time_fn(lambda:fp8dot(A8,B8))
        tmx=time_fn(lambda:mxfp8dot(A8,B8,SA,SB,M,N,K))
        def f(x): return f"{x:.1f}" if isinstance(x,float) else str(x)[:40]
        ratio=(tmx/t8) if (isinstance(tmx,float) and isinstance(t8,float)) else float('nan')
        print(f"{name:12s} {f'{Bb},{M},{K},{N}':>17s} {f(t8):>9s} {f(tmx):>11s} {ratio:>7.2f}",flush=True)
    print("\nIf mxfp8 ~= plain fp8 -> block scaling is free (no util win), verdict holds.")
    print("If mxfp8 << plain fp8 -> tcgen05 block-scaled MMA tiles better on K-poor; worth a relook.")

if __name__=="__main__":
    main()
