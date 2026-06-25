"""Gate 2 (decisive): a hand-written tiled tf32x3 batched GEMM via mma.sync.m16n8k8,
correctness-checked vs fp64 and timed vs cuBLAS-1xTF32 + our Triton tf32x3 (38 TF/s).
If this clears Triton meaningfully, the raw-PTX trailing engine is alive; if not, the
path is empirically dead. Tile: BM=64 BN=64 BK=16, 4 warps (2x2), each warp 32x32;
tf32x3 = 3 mma passes (hi*hi + hi*lo + lo*hi) accumulating fp32. No banned substrings.
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

__device__ __forceinline__ unsigned hibits(float x){ return __float_as_uint(x) & 0xFFFFE000u; }

// batched C[b] = A[b] (MxK) @ B[b] (KxN), tf32x3. grid=(M/BM, N/BN, batch), 128 threads.
__global__ void x3_kernel(const float* __restrict__ A, const float* __restrict__ B,
                          float* __restrict__ C, int M, int N, int K){
  int bz = blockIdx.z;
  int bm0 = blockIdx.x * BM;
  int bn0 = blockIdx.y * BN;
  const float* Ab = A + (long)bz * M * K;
  const float* Bb = B + (long)bz * K * N;
  float* Cb = C + (long)bz * M * N;

  __shared__ float As[BM][BK];
  __shared__ float Bs[BK][BN];

  int t = threadIdx.x;            // 0..127
  int warp = t >> 5;             // 0..3
  int lane = t & 31;
  int wrow = warp >> 1;          // 0,1
  int wcol = warp & 1;           // 0,1
  int gr  = lane >> 2;           // 0..7
  int tid = lane & 3;            // 0..3

  // C accumulators: per warp 32x32 = m-subtiles(2) x n-subtiles(4), each 4 floats
  float c[2][4][4];
  #pragma unroll
  for(int i=0;i<2;i++) for(int j=0;j<4;j++) for(int e=0;e<4;e++) c[i][j][e]=0.f;

  for(int k0=0; k0<K; k0+=BK){
    // cooperative load As (64x16) and Bs (16x64)
    #pragma unroll
    for(int i=0;i<8;i++){
      int idx = t + i*128;
      int r = idx >> 4, cc = idx & 15;      // 64x16
      int gm = bm0 + r, gk = k0 + cc;
      As[r][cc] = (gm < M && gk < K) ? Ab[(long)gm*K + gk] : 0.f;
      int idx2 = t + i*128;
      int br = idx2 >> 6, bc = idx2 & 63;   // 16x64
      int gk2 = k0 + br, gn = bn0 + bc;
      Bs[br][bc] = (gk2 < K && gn < N) ? Bb[(long)gk2*N + gn] : 0.f;
    }
    __syncthreads();

    #pragma unroll
    for(int kk=0; kk<BK; kk+=8){
      #pragma unroll
      for(int mi=0; mi<2; mi++){
        int arow = wrow*32 + mi*16;
        float av0 = As[arow + gr  ][kk + tid  ];
        float av1 = As[arow + gr+8][kk + tid  ];
        float av2 = As[arow + gr  ][kk + tid+4];
        float av3 = As[arow + gr+8][kk + tid+4];
        unsigned ah0=hibits(av0),ah1=hibits(av1),ah2=hibits(av2),ah3=hibits(av3);
        unsigned al0=__float_as_uint(av0-__uint_as_float(ah0));
        unsigned al1=__float_as_uint(av1-__uint_as_float(ah1));
        unsigned al2=__float_as_uint(av2-__uint_as_float(ah2));
        unsigned al3=__float_as_uint(av3-__uint_as_float(ah3));
        #pragma unroll
        for(int ni=0; ni<4; ni++){
          int bcol = wcol*32 + ni*8;
          float bv0 = Bs[kk + tid  ][bcol + gr];
          float bv1 = Bs[kk + tid+4][bcol + gr];
          unsigned bh0=hibits(bv0), bh1=hibits(bv1);
          unsigned bl0=__float_as_uint(bv0-__uint_as_float(bh0));
          unsigned bl1=__float_as_uint(bv1-__uint_as_float(bh1));
          float* cc4 = c[mi][ni];
          // pass 1: hi*hi
          asm volatile("mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 {%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%0,%1,%2,%3};\n"
            : "+f"(cc4[0]),"+f"(cc4[1]),"+f"(cc4[2]),"+f"(cc4[3])
            : "r"(ah0),"r"(ah1),"r"(ah2),"r"(ah3),"r"(bh0),"r"(bh1));
          // pass 2: hi*lo
          asm volatile("mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 {%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%0,%1,%2,%3};\n"
            : "+f"(cc4[0]),"+f"(cc4[1]),"+f"(cc4[2]),"+f"(cc4[3])
            : "r"(ah0),"r"(ah1),"r"(ah2),"r"(ah3),"r"(bl0),"r"(bl1));
          // pass 3: lo*hi
          asm volatile("mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 {%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%0,%1,%2,%3};\n"
            : "+f"(cc4[0]),"+f"(cc4[1]),"+f"(cc4[2]),"+f"(cc4[3])
            : "r"(al0),"r"(al1),"r"(al2),"r"(al3),"r"(bh0),"r"(bh1));
        }
      }
    }
    __syncthreads();
  }

  // store
  #pragma unroll
  for(int mi=0; mi<2; mi++){
    int crow = bm0 + wrow*32 + mi*16;
    #pragma unroll
    for(int ni=0; ni<4; ni++){
      int ccol = bn0 + wcol*32 + ni*8;
      float* cc4 = c[mi][ni];
      int r0 = crow + gr, r1 = crow + gr + 8;
      int c0 = ccol + 2*tid, c1 = ccol + 2*tid + 1;
      if(r0<M && c0<N) Cb[(long)r0*N + c0] = cc4[0];
      if(r0<M && c1<N) Cb[(long)r0*N + c1] = cc4[1];
      if(r1<M && c0<N) Cb[(long)r1*N + c0] = cc4[2];
      if(r1<M && c1<N) Cb[(long)r1*N + c1] = cc4[3];
    }
  }
}

torch::Tensor x3gemm(torch::Tensor A, torch::Tensor B){
  int Bb=A.size(0), M=A.size(1), K=A.size(2), N=B.size(2);
  auto C = torch::empty({Bb,M,N}, A.options());
  dim3 grid((M+BM-1)/BM, (N+BN-1)/BN, Bb);
  x3_kernel<<<grid, 128>>>(A.data_ptr<float>(), B.data_ptr<float>(), C.data_ptr<float>(), M,N,K);
  return C;
}
'''

print("compiling tf32x3 tiled GEMM...")
m = load_inline(name="g2", cpp_sources=cpp, cuda_sources=cuda, functions=["x3gemm"],
                extra_cuda_cflags=["-arch=sm_100", "-O3"])
torch.backends.cuda.matmul.allow_tf32 = True


def clear_l2(): torch.empty((32,1024,1024), dtype=torch.int64, device="cuda").fill_(0)
def time_fn(fn, it=40):
    for _ in range(5): fn()
    torch.cuda.synchronize(); ts=[]
    for _ in range(it):
        clear_l2(); torch.cuda.synchronize()
        s=torch.cuda.Event(enable_timing=True); e=torch.cuda.Event(enable_timing=True)
        s.record(); fn(); e.record(); torch.cuda.synchronize(); ts.append(s.elapsed_time(e))
    ts.sort(); return ts[len(ts)//2]*1000

PEAK=1100.0
for name,Bb,M,K,N in [("n512 fat 640,512,128,384",640,512,128,384),
                      ("n1024 fat 60,1024,256,768",60,1024,256,768)]:
    A=torch.randn(Bb,M,K,device="cuda"); B=torch.randn(Bb,K,N,device="cuda")
    ref=torch.matmul(A.double(),B.double()).float()
    C=m.x3gemm(A,B)
    rel=(C-ref).abs().max().item()/(ref.abs().max().item()+1e-9)
    flops=2.0*Bb*M*N*K; ceil=PEAK/3
    t=time_fn(lambda:m.x3gemm(A,B)); tf=flops/(t*1e-6)/1e12
    tcu=time_fn(lambda:torch.matmul(A,B)); tfcu=flops/(tcu*1e-6)/1e12
    print(f"\n{name}")
    print(f"  raw tf32x3 : {t:8.1f} us  {tf:6.0f} TFLOPs  {100*tf/ceil:4.0f}% ceil  relerr={rel:.1e}  (vs Triton 38/41)")
    print(f"  cuBLAS1xtf : {tcu:8.1f} us  {tfcu:6.0f} TFLOPs  {100*tfcu/PEAK:4.0f}% peak (loose)")
print("\nGATE2 DONE")
