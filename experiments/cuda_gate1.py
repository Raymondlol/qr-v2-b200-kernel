"""Gate 1: confirm the load_inline CUDA toolchain compiles + runs on B200 sm_100.
Trivial elementwise kernel + a minimal mma.sync.m16n8k8.tf32 smoke test (does the
Blackwell-compatible tensor-core MMA instruction assemble under nvcc 12.8 sm_100?).
"""
import torch
from torch.utils.cpp_extension import load_inline

print("torch", torch.__version__, "| cuda", torch.version.cuda)
print("dev", torch.cuda.get_device_name(0), "| cc", torch.cuda.get_device_capability(0))

cpp = "torch::Tensor add(torch::Tensor a, torch::Tensor b); torch::Tensor mmasmoke(torch::Tensor a, torch::Tensor b);"

cuda = r'''
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <mma.h>

__global__ void addk(const float* a, const float* b, float* c, int n){
  int i = blockIdx.x*blockDim.x + threadIdx.x;
  if (i < n) c[i] = a[i] + b[i];
}
torch::Tensor add(torch::Tensor a, torch::Tensor b){
  auto c = torch::empty_like(a);
  int n = a.numel();
  addk<<<(n+255)/256, 256>>>(a.data_ptr<float>(), b.data_ptr<float>(), c.data_ptr<float>(), n);
  return c;
}

// One-warp m16n8k8 tf32 MMA: C[16x8] = A[16x8] * B[8x8], accumulate fp32.
// Confirms mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 assembles on sm_100.
__global__ void mma_kernel(const float* A, const float* B, float* C){
  int lane = threadIdx.x; // 0..31
  unsigned a[4], b[2];
  float c[4] = {0.f,0.f,0.f,0.f};
  int gr  = lane >> 2;       // groupID 0..7
  int tid = lane & 3;        // thread-in-group 0..3
  // A (16x8 row-major) m16n8k8 tf32 A-fragment: cols tid and tid+4
  a[0] = __float_as_uint(A[(gr)   *8 + (tid)  ]);
  a[1] = __float_as_uint(A[(gr+8) *8 + (tid)  ]);
  a[2] = __float_as_uint(A[(gr)   *8 + (tid+4)]);
  a[3] = __float_as_uint(A[(gr+8) *8 + (tid+4)]);
  // B (8x8 row-major B[k*8+n]) B-fragment: rows tid and tid+4, col gr
  b[0] = __float_as_uint(B[(tid)  *8 + gr]);
  b[1] = __float_as_uint(B[(tid+4)*8 + gr]);
  asm volatile(
    "mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 "
    "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
    : "+f"(c[0]),"+f"(c[1]),"+f"(c[2]),"+f"(c[3])
    : "r"(a[0]),"r"(a[1]),"r"(a[2]),"r"(a[3]), "r"(b[0]),"r"(b[1]));
  // C (16x8) D-fragment: c0=(gr,2tid) c1=(gr,2tid+1) c2=(gr+8,2tid) c3=(gr+8,2tid+1)
  C[(gr)  *8 + (2*tid)  ] = c[0];
  C[(gr)  *8 + (2*tid+1)] = c[1];
  C[(gr+8)*8 + (2*tid)  ] = c[2];
  C[(gr+8)*8 + (2*tid+1)] = c[3];
}
torch::Tensor mmasmoke(torch::Tensor a, torch::Tensor b){
  auto c = torch::zeros({16,8}, a.options());
  mma_kernel<<<1,32>>>(a.data_ptr<float>(), b.data_ptr<float>(), c.data_ptr<float>());
  return c;
}
'''

print("compiling load_inline (first build may take a minute)...")
m = load_inline(name="g1", cpp_sources=cpp, cuda_sources=cuda,
                functions=["add", "mmasmoke"],
                extra_cuda_cflags=["-arch=sm_100", "-O3"])
a = torch.randn(4096, device="cuda"); b = torch.randn(4096, device="cuda")
c = m.add(a, b)
print("add max err:", (c - (a+b)).abs().max().item(), "->", "OK" if torch.allclose(c, a+b) else "FAIL")

A = torch.randn(16, 8, device="cuda"); B = torch.randn(8, 8, device="cuda")
C = m.mmasmoke(A, B)
ref = A @ B
err = (C - ref).abs().max().item() / (ref.abs().max().item() + 1e-9)
print("mma m16n8k8 tf32 relerr vs fp32:", err, "->", "OK (tf32 ~1e-3)" if err < 5e-3 else "CHECK LAYOUT")
print("GATE1 DONE")
