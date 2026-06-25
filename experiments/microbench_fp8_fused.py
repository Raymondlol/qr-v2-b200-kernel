"""Properly FUSED fp8-Ozaki-6dot trailing kernel: split V,Y into 3 e4m3 slices
IN-REGISTER (per-tensor scales precomputed), do all 6 dots (i+j<=2) accumulating
to one fp32 acc, one output write. This is the make-or-break for the fp8 SPEED
thesis: the naive torch version was 7076us (split + 6 launches + 6 intermediates);
fused should be ~400us if the K-poor fp8 MMA utilizes. WIN iff < fused tf32x3 (1267us).

Run: modal run modal_microbench.py --script microbench_fp8_fused.py
"""
import torch, triton, triton.language as tl
torch.set_grad_enabled(False)
torch.backends.cuda.matmul.allow_tf32 = True
DEV = "cuda"

def clear_l2(): torch.empty((32,1024,1024),dtype=torch.int64,device=DEV).fill_(0)
def time_fn(fn, it=30):
    for _ in range(6):
        try: fn()
        except Exception as e: return "ERR:"+type(e).__name__+":"+str(e)[:90]
    torch.cuda.synchronize(); ts=[]
    for _ in range(it):
        clear_l2(); torch.cuda.synchronize()
        s=torch.cuda.Event(enable_timing=True);e=torch.cuda.Event(enable_timing=True)
        s.record(); fn(); e.record(); torch.cuda.synchronize(); ts.append(s.elapsed_time(e))
    ts.sort(); return ts[len(ts)//2]*1000.0

_CFG=[triton.Config({'BM':bm,'BN':bn,'BK':bk},num_warps=nw,num_stages=ns)
      for bm in (64,128) for bn in (64,128) for bk in (64,128) for nw in (4,8) for ns in (3,4)
      if bm*bn*bk<=128*128*128]

@triton.autotune(configs=_CFG, key=['M','N','K'])
@triton.jit
def _x3(A,B,C,M,N,K,sab,sam,sak,sbb,sbk,sbn,scb,scm,scn,
        BM:tl.constexpr,BN:tl.constexpr,BK:tl.constexpr):
    pb=tl.program_id(0);pm=tl.program_id(1);pn=tl.program_id(2)
    rm=pm*BM+tl.arange(0,BM);rn=pn*BN+tl.arange(0,BN);rk=tl.arange(0,BK)
    ap=A+pb*sab+(rm[:,None]*sam+rk[None,:]*sak);bp=B+pb*sbb+(rk[:,None]*sbk+rn[None,:]*sbn)
    acc=tl.zeros((BM,BN),dtype=tl.float32)
    for k0 in range(0,K,BK):
        a=tl.load(ap,mask=(rm[:,None]<M)&(rk[None,:]+k0<K),other=0.0)
        b=tl.load(bp,mask=(rk[:,None]+k0<K)&(rn[None,:]<N),other=0.0)
        acc+=tl.dot(a,b,input_precision="tf32x3");ap+=BK*sak;bp+=BK*sbk
    tl.store(C+pb*scb+(rm[:,None]*scm+rn[None,:]*scn),acc,mask=(rm[:,None]<M)&(rn[None,:]<N))
def x3(A,B):
    Bb,M,K=A.shape;N=B.shape[2];C=torch.empty((Bb,M,N),device=DEV,dtype=torch.float32)
    _x3[lambda m:(Bb,triton.cdiv(M,m['BM']),triton.cdiv(N,m['BN']))](A,B,C,M,N,K,*A.stride(),*B.stride(),*C.stride());return C

# fused fp8 6-dot: split in-register using precomputed per-tensor scales
@triton.autotune(configs=_CFG, key=['M','N','K'])
@triton.jit
def _fp8x6(A,B,C,M,N,K, sA0,sA1,sA2, sB0,sB1,sB2,
          sab,sam,sak,sbb,sbk,sbn,scb,scm,scn,
          BM:tl.constexpr,BN:tl.constexpr,BK:tl.constexpr):
    pb=tl.program_id(0);pm=tl.program_id(1);pn=tl.program_id(2)
    rm=pm*BM+tl.arange(0,BM);rn=pn*BN+tl.arange(0,BN);rk=tl.arange(0,BK)
    ap=A+pb*sab+(rm[:,None]*sam+rk[None,:]*sak);bp=B+pb*sbb+(rk[:,None]*sbk+rn[None,:]*sbn)
    acc=tl.zeros((BM,BN),dtype=tl.float32)
    for k0 in range(0,K,BK):
        a=tl.load(ap,mask=(rm[:,None]<M)&(rk[None,:]+k0<K),other=0.0)
        b=tl.load(bp,mask=(rk[:,None]+k0<K)&(rn[None,:]<N),other=0.0)
        # split a into 3 e4m3 slices (per-tensor scales)
        a0=(a/sA0).to(tl.float8e4nv); ra=a-a0.to(tl.float32)*sA0
        a1=(ra/sA1).to(tl.float8e4nv); ra=ra-a1.to(tl.float32)*sA1
        a2=(ra/sA2).to(tl.float8e4nv)
        b0=(b/sB0).to(tl.float8e4nv); rb=b-b0.to(tl.float32)*sB0
        b1=(rb/sB1).to(tl.float8e4nv); rb=rb-b1.to(tl.float32)*sB1
        b2=(rb/sB2).to(tl.float8e4nv)
        acc+=tl.dot(a0,b0,out_dtype=tl.float32)*(sA0*sB0)
        acc+=tl.dot(a0,b1,out_dtype=tl.float32)*(sA0*sB1)
        acc+=tl.dot(a1,b0,out_dtype=tl.float32)*(sA1*sB0)
        acc+=tl.dot(a0,b2,out_dtype=tl.float32)*(sA0*sB2)
        acc+=tl.dot(a1,b1,out_dtype=tl.float32)*(sA1*sB1)
        acc+=tl.dot(a2,b0,out_dtype=tl.float32)*(sA2*sB0)
        ap+=BK*sak;bp+=BK*sbk
    tl.store(C+pb*scb+(rm[:,None]*scm+rn[None,:]*scn),acc,mask=(rm[:,None]<M)&(rn[None,:]<N))

def scales3(X):
    R=X.float().clone(); ss=[]
    for _ in range(3):
        s=(R.abs().amax()/448.0).clamp_min(1e-30); ss.append(s)
        R=R-(R/s).to(torch.float8_e4m3fn).float()*s
    return [float(s) for s in ss]

def fp8x6(A,B,sA,sB):
    Bb,M,K=A.shape;N=B.shape[2];C=torch.empty((Bb,M,N),device=DEV,dtype=torch.float32)
    _fp8x6[lambda m:(Bb,triton.cdiv(M,m['BM']),triton.cdiv(N,m['BN']))](
        A,B,C,M,N,K,sA[0],sA[1],sA[2],sB[0],sB[1],sB[2],
        *A.stride(),*B.stride(),*C.stride());return C

def main():
    print("torch",torch.__version__,"| dev",torch.cuda.get_device_name(0))
    shapes=[("FAT n512",640,512,128,384),("narrow n512",640,448,64,384),("FAT n1024",60,1024,256,768)]
    print(f"{'shape':14s} {'B,M,K,N':>17s} {'fusedX3':>9s} {'1xTF32':>8s} {'fp8x6_fused':>12s} "
          f"{'+scalecost':>11s} {'fp8/fused':>10s} {'relerr':>9s}")
    for name,Bb,M,K,N in shapes:
        A=torch.randn(Bb,M,K,device=DEV);B=torch.randn(Bb,K,N,device=DEV)
        ref=torch.matmul(A.double(),B.double())
        tf=time_fn(lambda:x3(A,B))
        t1=time_fn(lambda:torch.matmul(A,B))
        sA=scales3(A);sB=scales3(B)
        tk=time_fn(lambda:fp8x6(A,B,sA,sB))               # kernel only (scales precomputed)
        tfull=time_fn(lambda:fp8x6(A,B,scales3(A),scales3(B)))  # + scale cost (deployable)
        try:
            c=fp8x6(A,B,sA,sB);rel=((c.double()-ref).abs().max()/ref.abs().max()).item()
        except Exception as ex: rel=float('nan')
        def f(x): return f"{x:.1f}" if isinstance(x,float) else str(x)[:12]
        ratio=(tk/tf) if (isinstance(tk,float) and isinstance(tf,float)) else float('nan')
        print(f"{name:14s} {f'{Bb},{M},{K},{N}':>17s} {f(tf):>9s} {f(t1):>8s} {f(tk):>12s} "
              f"{f(tfull):>11s} {ratio:>10.2f} {rel:>9.1e}",flush=True)
    print("\nfp8x6_fused = kernel w/ precomputed scales; +scalecost = realistic (3 amax/operand).")
    print("WIN iff fp8x6_fused < fusedX3 AND the +scalecost version stays competitive.")

if __name__=="__main__":
    main()
