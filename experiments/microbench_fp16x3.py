"""fp16x3 vs tf32x3 trailing: same ~22-bit accuracy (SAFE), but fp16 tensor cores
are 2x tf32 on B200, so a tight fused fp16x3 (3 fp16 dots, split in-register) could
be ~2x faster than tf32x3 AND keep the safe margin -- the win 1xTF32 (m1.02, DQ)
couldn't deliver. Docs called fp16x3 '~wash' but may have measured an unoptimized
version; measure a tight AUTOTUNED kernel on the real shapes. Also bf16x3 for ref.

Run: modal run modal_microbench.py --script microbench_fp16x3.py
"""
import torch, triton, triton.language as tl
torch.set_grad_enabled(False)
torch.backends.cuda.matmul.allow_tf32 = True
DEV = "cuda"

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

_CFG=[triton.Config({'BM':bm,'BN':bn,'BK':bk},num_warps=nw,num_stages=ns)
      for bm in (64,128) for bn in (64,128) for bk in (32,64) for nw in (4,8) for ns in (3,4)
      if bm*bn*bk<=128*128*64]

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

@triton.autotune(configs=_CFG, key=['M','N','K'])
@triton.jit
def _f16x3(A,B,C,M,N,K,sab,sam,sak,sbb,sbk,sbn,scb,scm,scn,
          BM:tl.constexpr,BN:tl.constexpr,BK:tl.constexpr):
    pb=tl.program_id(0);pm=tl.program_id(1);pn=tl.program_id(2)
    rm=pm*BM+tl.arange(0,BM);rn=pn*BN+tl.arange(0,BN);rk=tl.arange(0,BK)
    ap=A+pb*sab+(rm[:,None]*sam+rk[None,:]*sak);bp=B+pb*sbb+(rk[:,None]*sbk+rn[None,:]*sbn)
    acc=tl.zeros((BM,BN),dtype=tl.float32)
    for k0 in range(0,K,BK):
        a=tl.load(ap,mask=(rm[:,None]<M)&(rk[None,:]+k0<K),other=0.0)
        b=tl.load(bp,mask=(rk[:,None]+k0<K)&(rn[None,:]<N),other=0.0)
        ah=a.to(tl.float16); al=(a-ah.to(tl.float32)).to(tl.float16)
        bh=b.to(tl.float16); bl=(b-bh.to(tl.float32)).to(tl.float16)
        acc+=tl.dot(ah,bh,out_dtype=tl.float32)
        acc+=tl.dot(ah,bl,out_dtype=tl.float32)
        acc+=tl.dot(al,bh,out_dtype=tl.float32)
        ap+=BK*sak;bp+=BK*sbk
    tl.store(C+pb*scb+(rm[:,None]*scm+rn[None,:]*scn),acc,mask=(rm[:,None]<M)&(rn[None,:]<N))

@triton.autotune(configs=_CFG, key=['M','N','K'])
@triton.jit
def _bf16x3(A,B,C,M,N,K,sab,sam,sak,sbb,sbk,sbn,scb,scm,scn,
          BM:tl.constexpr,BN:tl.constexpr,BK:tl.constexpr):
    pb=tl.program_id(0);pm=tl.program_id(1);pn=tl.program_id(2)
    rm=pm*BM+tl.arange(0,BM);rn=pn*BN+tl.arange(0,BN);rk=tl.arange(0,BK)
    ap=A+pb*sab+(rm[:,None]*sam+rk[None,:]*sak);bp=B+pb*sbb+(rk[:,None]*sbk+rn[None,:]*sbn)
    acc=tl.zeros((BM,BN),dtype=tl.float32)
    for k0 in range(0,K,BK):
        a=tl.load(ap,mask=(rm[:,None]<M)&(rk[None,:]+k0<K),other=0.0)
        b=tl.load(bp,mask=(rk[:,None]+k0<K)&(rn[None,:]<N),other=0.0)
        ah=a.to(tl.bfloat16); al=(a-ah.to(tl.float32)).to(tl.bfloat16)
        bh=b.to(tl.bfloat16); bl=(b-bh.to(tl.float32)).to(tl.bfloat16)
        acc+=tl.dot(ah,bh,out_dtype=tl.float32)
        acc+=tl.dot(ah,bl,out_dtype=tl.float32)
        acc+=tl.dot(al,bh,out_dtype=tl.float32)
        ap+=BK*sak;bp+=BK*sbk
    tl.store(C+pb*scb+(rm[:,None]*scm+rn[None,:]*scn),acc,mask=(rm[:,None]<M)&(rn[None,:]<N))

def runner(kern):
    def f(A,B):
        Bb,M,K=A.shape;N=B.shape[2];C=torch.empty((Bb,M,N),device=DEV,dtype=torch.float32)
        kern[lambda m:(Bb,triton.cdiv(M,m['BM']),triton.cdiv(N,m['BN']))](
            A,B,C,M,N,K,*A.stride(),*B.stride(),*C.stride());return C
    return f
x3=runner(_x3); f16x3=runner(_f16x3); bf16x3=runner(_bf16x3)

def main():
    print("torch",torch.__version__,"| dev",torch.cuda.get_device_name(0))
    shapes=[("FAT n512",640,512,128,384),("narrow n512",640,448,64,384),
            ("FAT n1024",60,1024,256,768),("wide n512 first",640,512,128,384)]
    print(f"{'shape':16s} {'B,M,K,N':>17s} {'tf32x3':>8s} {'fp16x3':>8s} {'bf16x3':>8s} {'1xTF32':>8s} "
          f"{'f16/x3':>7s} {'f16relerr':>10s} {'bf16relerr':>10s}")
    for name,Bb,M,K,N in shapes:
        A=torch.randn(Bb,M,K,device=DEV);B=torch.randn(Bb,K,N,device=DEV)
        ref=torch.matmul(A.double(),B.double())
        t3=time_fn(lambda:x3(A,B)); tf=time_fn(lambda:f16x3(A,B))
        tb=time_fn(lambda:bf16x3(A,B)); t1=time_fn(lambda:torch.matmul(A,B))
        def rel(fn):
            try: return ((fn(A,B).double()-ref).abs().max()/ref.abs().max()).item()
            except: return float('nan')
        r16=rel(f16x3); rbf=rel(bf16x3)
        def f(x): return f"{x:.1f}" if isinstance(x,float) else str(x)[:8]
        ratio=(tf/t3) if (isinstance(tf,float) and isinstance(t3,float)) else float('nan')
        print(f"{name:16s} {f'{Bb},{M},{K},{N}':>17s} {f(t3):>8s} {f(tf):>8s} {f(tb):>8s} {f(t1):>8s} "
              f"{ratio:>7.2f} {r16:>10.1e} {rbf:>10.1e}",flush=True)
    print("\nfp16x3 ~22bit (=tf32x3, SAFE). WIN iff fp16x3 << tf32x3 with relerr ~ tf32x3 (~3e-6).")

if __name__=="__main__":
    main()
