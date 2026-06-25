"""Gate 3: cp.async 2-stage double-buffered tf32x3 tiled GEMM (overlap global->shared
loads with mma compute) -- the optimized-raw ceiling. Same tile/mma as Gate 2 (which
tied Triton at 33-37 TF/s single-buffered). If double-buffering clears Triton's 38/41
meaningfully, the raw-PTX trailing engine is worth integrating; if it stays ~parity,
the path is empirically bounded by the K-poor shape (cuBLAS itself only 20-31%).
Shapes divide the tile (M%64=N%64=K%16=0) so loads need no predication. No banned subs.
"""
import torch
from torch.utils.cpp_extension import load_inline

cpp = "torch::Tensor x3gemm(torch::Tensor A, torch::Tensor B);"

cuda = r'''
#include <torch/extension.h>
#include <cuda_runtime.h>
#define BM 64
#define BN 64
#define BK 16
__device__ __forceinline__ unsigned hib(float x){ return __float_as_uint(x) & 0xFFFFE000u; }
__device__ __forceinline__ void cpasync16(float* smem, const float* gmem){
  unsigned s = (unsigned)__cvta_generic_to_shared(smem);
  asm volatile("cp.async.cg.shared.global [%0],[%1],16;\n" :: "r"(s), "l"(gmem));
}

__global__ void x3_kernel(const float* __restrict__ A, const float* __restrict__ B,
                          float* __restrict__ C, int M, int N, int K){
  int bz=blockIdx.z, bm0=blockIdx.x*BM, bn0=blockIdx.y*BN;
  const float* Ab=A+(long)bz*M*K; const float* Bb=B+(long)bz*K*N; float* Cb=C+(long)bz*M*N;
  __shared__ float As[2][BM][BK];
  __shared__ float Bs[2][BK][BN];
  int t=threadIdx.x, warp=t>>5, lane=t&31, wrow=warp>>1, wcol=warp&1, gr=lane>>2, tid=lane&3;
  float c[2][4][4];
  #pragma unroll
  for(int i=0;i<2;i++)for(int j=0;j<4;j++)for(int e=0;e<4;e++)c[i][j][e]=0.f;
  int numK = K / BK;

  #define LOAD_TILE(buf, kt) do{ int k0=(kt)*BK; \
    _Pragma("unroll") for(int i=0;i<2;i++){ int ch=t+i*128; int r=ch>>2; int c4=(ch&3)*4; \
      cpasync16(&As[buf][r][c4], &Ab[(long)(bm0+r)*K + (k0+c4)]); } \
    _Pragma("unroll") for(int i=0;i<2;i++){ int ch=t+i*128; int r=ch>>4; int c4=(ch&15)*4; \
      cpasync16(&Bs[buf][r][c4], &Bb[(long)(k0+r)*N + (bn0+c4)]); } \
    asm volatile("cp.async.commit_group;\n"); }while(0)

  LOAD_TILE(0, 0);
  for(int kt=0; kt<numK; kt++){
    int cur = kt & 1;
    if(kt+1 < numK){ LOAD_TILE((kt+1)&1, kt+1); asm volatile("cp.async.wait_group 1;\n"); }
    else { asm volatile("cp.async.wait_group 0;\n"); }
    __syncthreads();
    #pragma unroll
    for(int kk=0; kk<BK; kk+=8){
      #pragma unroll
      for(int mi=0; mi<2; mi++){
        int ar = wrow*32 + mi*16;
        float v0=As[cur][ar+gr][kk+tid], v1=As[cur][ar+gr+8][kk+tid];
        float v2=As[cur][ar+gr][kk+tid+4], v3=As[cur][ar+gr+8][kk+tid+4];
        unsigned ah0=hib(v0),ah1=hib(v1),ah2=hib(v2),ah3=hib(v3);
        unsigned al0=__float_as_uint(v0-__uint_as_float(ah0)),al1=__float_as_uint(v1-__uint_as_float(ah1));
        unsigned al2=__float_as_uint(v2-__uint_as_float(ah2)),al3=__float_as_uint(v3-__uint_as_float(ah3));
        #pragma unroll
        for(int ni=0; ni<4; ni++){
          int bc=wcol*32+ni*8;
          float w0=Bs[cur][kk+tid][bc+gr], w1=Bs[cur][kk+tid+4][bc+gr];
          unsigned bh0=hib(w0),bh1=hib(w1);
          unsigned bl0=__float_as_uint(w0-__uint_as_float(bh0)),bl1=__float_as_uint(w1-__uint_as_float(bh1));
          float* x=c[mi][ni];
          asm volatile("mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 {%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%0,%1,%2,%3};\n"
            :"+f"(x[0]),"+f"(x[1]),"+f"(x[2]),"+f"(x[3]):"r"(ah0),"r"(ah1),"r"(ah2),"r"(ah3),"r"(bh0),"r"(bh1));
          asm volatile("mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 {%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%0,%1,%2,%3};\n"
            :"+f"(x[0]),"+f"(x[1]),"+f"(x[2]),"+f"(x[3]):"r"(ah0),"r"(ah1),"r"(ah2),"r"(ah3),"r"(bl0),"r"(bl1));
          asm volatile("mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 {%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%0,%1,%2,%3};\n"
            :"+f"(x[0]),"+f"(x[1]),"+f"(x[2]),"+f"(x[3]):"r"(al0),"r"(al1),"r"(al2),"r"(al3),"r"(bh0),"r"(bh1));
        }
      }
    }
    __syncthreads();
  }
  #pragma unroll
  for(int mi=0; mi<2; mi++){
    int crow=bm0+wrow*32+mi*16;
    #pragma unroll
    for(int ni=0; ni<4; ni++){
      int ccol=bn0+wcol*32+ni*8; float* x=c[mi][ni];
      int r0=crow+gr, r1=crow+gr+8, c0=ccol+2*tid, c1=ccol+2*tid+1;
      Cb[(long)r0*N+c0]=x[0]; Cb[(long)r0*N+c1]=x[1]; Cb[(long)r1*N+c0]=x[2]; Cb[(long)r1*N+c1]=x[3];
    }
  }
}
torch::Tensor x3gemm(torch::Tensor A, torch::Tensor B){
  int Bb=A.size(0),M=A.size(1),K=A.size(2),N=B.size(2);
  auto C=torch::empty({Bb,M,N},A.options());
  dim3 grid((M+BM-1)/BM,(N+BN-1)/BN,Bb);
  x3_kernel<<<grid,128>>>(A.data_ptr<float>(),B.data_ptr<float>(),C.data_ptr<float>(),M,N,K);
  return C;
}
'''

print("compiling cp.async double-buffered tf32x3 GEMM...")
m = load_inline(name="g3", cpp_sources=cpp, cuda_sources=cuda, functions=["x3gemm"],
                extra_cuda_cflags=["-arch=sm_100", "-O3"])
torch.backends.cuda.matmul.allow_tf32 = True
def clear_l2(): torch.empty((32,1024,1024),dtype=torch.int64,device="cuda").fill_(0)
def time_fn(fn,it=40):
    for _ in range(5): fn()
    torch.cuda.synchronize(); ts=[]
    for _ in range(it):
        clear_l2(); torch.cuda.synchronize()
        s=torch.cuda.Event(enable_timing=True); e=torch.cuda.Event(enable_timing=True)
        s.record(); fn(); e.record(); torch.cuda.synchronize(); ts.append(s.elapsed_time(e))
    ts.sort(); return ts[len(ts)//2]*1000
PEAK=1100.0
for name,Bb,M,K,N in [("n512 fat 640,512,128,384",640,512,128,384),("n1024 fat 60,1024,256,768",60,1024,256,768)]:
    A=torch.randn(Bb,M,K,device="cuda"); B=torch.randn(Bb,K,N,device="cuda")
    ref=torch.matmul(A.double(),B.double()).float(); C=m.x3gemm(A,B)
    rel=(C-ref).abs().max().item()/(ref.abs().max().item()+1e-9)
    flops=2.0*Bb*M*N*K; ceil=PEAK/3
    t=time_fn(lambda:m.x3gemm(A,B)); tf=flops/(t*1e-6)/1e12
    print(f"\n{name}\n  raw tf32x3 DB: {t:8.1f} us  {tf:6.0f} TFLOPs  {100*tf/ceil:4.0f}% ceil  relerr={rel:.1e}  (vs Triton 38/41, Gate2 33/37)")
print("\nGATE3 DONE")
