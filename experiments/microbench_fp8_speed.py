"""fp8 trailing SPEED on the real shapes: is fp8-Ozaki-6dot actually faster than
the current fused tf32x3 trailing (and the simpler alternatives)? Uses PLAIN
tl.dot on fp8 inputs (per-tensor-scaled Ozaki slices -> no block-scale machinery
needed). Compares, on the real fat trailing shape and the within-panel shapes:
  fused tf32x3 (current) | 3xcuBLAS tf32x3 | 1xTF32 cuBLAS | fp8 1-dot | fp8 6-dot (full)
The 6-dot path = split V,Y into 3 fp8 slices each (per-tensor scaled) + 6 fp8 dots
(i+j<=2) scaled-accumulated -> this is the deployable fp8-Ozaki-6dot (margin 10.8x).

Run: modal run modal_microbench.py --script microbench_fp8_speed.py
"""
import torch, triton, triton.language as tl
torch.set_grad_enabled(False)
torch.backends.cuda.matmul.allow_tf32 = True
DEV = "cuda"

def clear_l2():
    torch.empty((32, 1024, 1024), dtype=torch.int64, device=DEV).fill_(0)

def time_fn(fn, it=30):
    for _ in range(6):
        try: fn()
        except Exception as e: return "ERR:" + type(e).__name__ + ":" + str(e)[:80]
    torch.cuda.synchronize(); ts = []
    for _ in range(it):
        clear_l2(); torch.cuda.synchronize()
        s = torch.cuda.Event(enable_timing=True); e = torch.cuda.Event(enable_timing=True)
        s.record(); fn(); e.record(); torch.cuda.synchronize(); ts.append(s.elapsed_time(e))
    ts.sort(); return ts[len(ts)//2]*1000.0

# ---- fused tf32x3 (current submission kernel) ----
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
    cp=C+pb*scb+(rm[:,None]*scm+rn[None,:]*scn)
    tl.store(cp,acc,mask=(rm[:,None]<M)&(rn[None,:]<N))
def x3(A,B,BM=64,BN=64,BK=32):
    Bb,M,K=A.shape;N=B.shape[2];C=torch.empty((Bb,M,N),device=DEV,dtype=torch.float32)
    _x3[(Bb,triton.cdiv(M,BM),triton.cdiv(N,BN))](A,B,C,M,N,K,*A.stride(),*B.stride(),*C.stride(),
        BM=BM,BN=BN,BK=BK,num_warps=4,num_stages=3);return C

# ---- plain fp8 batched dot (per-tensor scaled outside) ----
@triton.jit
def _fp8(A,B,C,M,N,K,sab,sam,sak,sbb,sbk,sbn,scb,scm,scn,
         BM:tl.constexpr,BN:tl.constexpr,BK:tl.constexpr):
    pb=tl.program_id(0);pm=tl.program_id(1);pn=tl.program_id(2)
    rm=pm*BM+tl.arange(0,BM);rn=pn*BN+tl.arange(0,BN);rk=tl.arange(0,BK)
    ap=A+pb*sab+(rm[:,None]*sam+rk[None,:]*sak);bp=B+pb*sbb+(rk[:,None]*sbk+rn[None,:]*sbn)
    acc=tl.zeros((BM,BN),dtype=tl.float32)
    for k0 in range(0,K,BK):
        a=tl.load(ap,mask=(rm[:,None]<M)&(rk[None,:]+k0<K))
        b=tl.load(bp,mask=(rk[:,None]+k0<K)&(rn[None,:]<N))
        acc+=tl.dot(a,b,out_dtype=tl.float32);ap+=BK*sak;bp+=BK*sbk
    cp=C+pb*scb+(rm[:,None]*scm+rn[None,:]*scn)
    tl.store(cp,acc,mask=(rm[:,None]<M)&(rn[None,:]<N))
def fp8dot(A8,B8,BM=64,BN=64,BK=64):
    Bb,M,K=A8.shape;N=B8.shape[2];C=torch.empty((Bb,M,N),device=DEV,dtype=torch.float32)
    _fp8[(Bb,triton.cdiv(M,BM),triton.cdiv(N,BN))](A8,B8,C,M,N,K,*A8.stride(),*B8.stride(),*C.stride(),
        BM=BM,BN=BN,BK=BK,num_warps=4,num_stages=3);return C

def to_fp8_slices(X, k):
    """per-tensor-scaled Ozaki split into k fp8 e4m3 slices; returns [(q_fp8, scale_f32)]."""
    R=X.to(torch.float32).clone();out=[]
    for _ in range(k):
        m=R.abs().amax().clamp_min(1e-30);scale=(m/448.0)
        q=(R/scale).to(torch.float8_e4m3fn)
        out.append((q,scale));R=R-q.to(torch.float32)*scale
    return out

def ozaki6(Vs,Ys,M,N,K):
    """6-dot fp8 Ozaki (i+j<=2): C = sum_{i+j<=2} (Vi@Yj)*sVi*sYj."""
    C=None
    for i,(Vi,sVi) in enumerate(Vs):
        for j,(Yj,sYj) in enumerate(Ys):
            if i+j>2: continue
            p=fp8dot(Vi,Yj)*(sVi*sYj)
            C=p if C is None else C+p
    return C

def _mm3(A,B):
    def split(x):
        xi=x.view(torch.int32);hi=(xi & -8192).view(torch.float32);return hi,x-hi
    Ah,Al=split(A);Bh,Bl=split(B)
    return torch.baddbmm(torch.baddbmm(torch.matmul(Ah,Bh),Ah,Bl),Al,Bh)

def main():
    print("torch",torch.__version__,"| dev",torch.cuda.get_device_name(0))
    shapes=[("trail FAT  n512",640,512,128,384),
            ("trail narrow n512",640,448,64,384),
            ("trail FAT  n1024",60,1024,256,768)]
    print(f"{'shape':20s} {'B,M,K,N':>18s} {'fusedX3':>9s} {'3xcuBLAS':>9s} {'1xTF32':>8s} "
          f"{'fp8_1dot':>9s} {'fp8_6dot':>9s} {'6dot/fused':>10s} {'fp8relerr':>10s}")
    for name,Bb,M,K,N in shapes:
        A=torch.randn(Bb,M,K,device=DEV);B=torch.randn(Bb,K,N,device=DEV)
        ref=torch.matmul(A.double(),B.double())
        tf=time_fn(lambda:x3(A,B))
        t3=time_fn(lambda:_mm3(A,B))
        t1=time_fn(lambda:torch.matmul(A,B))
        Vs=to_fp8_slices(A,3);Ys=to_fp8_slices(B,3)
        t8_1=time_fn(lambda:fp8dot(Vs[0][0],Ys[0][0]))
        # full 6-dot incl the split each call (deployable cost)
        def full6():
            vs=to_fp8_slices(A,3);ys=to_fp8_slices(B,3);return ozaki6(vs,ys,M,N,K)
        t8_6=time_fn(full6)
        try:
            c6=full6();rel=((c6.double()-ref).abs().max()/ref.abs().max()).item()
        except Exception as e: rel=float('nan')
        def fmt(x): return f"{x:.1f}" if isinstance(x,float) else str(x)[:9]
        ratio=(t8_6/tf) if (isinstance(t8_6,float) and isinstance(tf,float)) else float('nan')
        print(f"{name:20s} {f'{Bb},{M},{K},{N}':>18s} {fmt(tf):>9s} {fmt(t3):>9s} {fmt(t1):>8s} "
              f"{fmt(t8_1):>9s} {fmt(t8_6):>9s} {ratio:>10.2f} {rel:>10.2e}",flush=True)
    print("\nfp8_6dot is the deployable margin-10.8x trailing. WIN if fp8_6dot < fusedX3 (and < 3xcuBLAS).")

if __name__=="__main__":
    main()
