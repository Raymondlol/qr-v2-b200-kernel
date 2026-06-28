"""Dump PTX/SASS of a cute-DSL kernel (static low-level analysis; ncu/nsys are gVisor-dead on Modal).
Compiles a small representative kernel (cute_apply_gemm1) and the m3c apply, introspects the compiled
object + scans the JIT cache for cubin/ptx, then disassembles with nvdisasm/cuobjdump."""
import os, sys, glob, shutil, subprocess
# best-effort: ask the toolchain to keep intermediates
os.environ.setdefault("CUDA_CACHE_DISABLE", "0")
os.environ.setdefault("CUTLASS_DSL_KEEP_TEMPS", "1")
os.environ.setdefault("CUTE_DSL_KEEP_IR", "1")
import torch
import cutlass
import cutlass.cute as cute
from cutlass.cute.runtime import from_dlpack
P = lambda *a: print(*a, flush=True)

P("tools: nvdisasm=", shutil.which("nvdisasm"), " cuobjdump=", shutil.which("cuobjdump"),
  " ptxas=", shutil.which("ptxas"), " nvcc=", shutil.which("nvcc"))

sys.path.insert(0, "/work")
import cute_apply_gemm1 as G   # the validated GEMM-1 apply (small, fast to compile)

M = 32
V = torch.randn(M, 16, device="cuda", dtype=torch.float32)
C = torch.randn(M, 64, device="cuda", dtype=torch.float32)
W = torch.zeros(128, 256, device="cuda", dtype=torch.float32)
T = lambda x, d: (from_dlpack(x, assumed_align=16).mark_layout_dynamic(leading_dim=1)
                  .mark_compact_shape_dynamic(mode=1, divisibility=d))

snap_before = set()
for d in ["/tmp", "/root/.cache", "/root", "/dev/shm", os.getcwd()]:
    snap_before |= set(glob.glob(f"{d}/**/*.cubin", recursive=True))
    snap_before |= set(glob.glob(f"{d}/**/*.ptx", recursive=True))

P("\ncompiling cute_apply_gemm1.host_function ...")
fn = cute.compile(G.host_function, T(V, 16), T(C, 64), T(W, 256), 32)
P("compiled type:", type(fn))
attrs = [a for a in dir(fn) if not a.startswith("__")]
P("compiled attrs:", attrs)
for a in attrs:
    if any(k in a.lower() for k in ["ptx", "cubin", "sass", "module", "asm", "ir", "kernel", "code"]):
        try:
            v = getattr(fn, a)
            s = v() if callable(v) else v
            P(f"  fn.{a}: {type(s)}  {str(s)[:140]}")
        except Exception as e:
            P(f"  fn.{a}: <err {type(e).__name__}>")

# new artifacts on disk after compile
P("\n--- new cubin/ptx after compile ---")
new = []
for d in ["/tmp", "/root/.cache", "/root", "/dev/shm", os.getcwd()]:
    for ext in ["cubin", "ptx"]:
        for f in glob.glob(f"{d}/**/*.{ext}", recursive=True):
            if f not in snap_before:
                new.append(f); P("  NEW:", f, os.path.getsize(f), "bytes")
if not new:
    P("  (none on disk — cubin likely in-memory)")

# HIGH-VALUE: regs / spills / smem via `ptxas -v` on the PTX; light instr histogram on PTX.
ptxas = shutil.which("ptxas")
for f in new:
    if f.endswith(".ptx"):
        txt = open(f, errors="ignore").read()
        P(f"\n=== {f}: instr counts (static) ===")
        P("  bar.sync=", txt.count("bar.sync"), " mbarrier=", txt.count("mbarrier"),
          " mma/tcgen05=", txt.count("mma.") + txt.count("tcgen05"),
          " ld.shared=", txt.count("ld.shared"), " st.shared=", txt.count("st.shared"),
          " ld.global=", txt.count("ld.global"), " st.global=", txt.count("st.global"),
          " cp.async=", txt.count("cp.async"))
        if ptxas:
            P(f"\n=== ptxas -v -arch=sm_100a {f}  (regs / smem / SPILLS) ===")
            r = subprocess.run([ptxas, "-v", "-arch=sm_100a", f, "-o", "/tmp/_o.cubin"],
                               capture_output=True, text=True)
            for ln in (r.stderr or r.stdout).splitlines():
                if any(k in ln for k in ["register", "smem", "spill", "Used", "bytes", "Function", "stack"]):
                    P("  " + ln.strip())
# also dump regs/smem of an already-compiled cubin if present
for f in new:
    if f.endswith(".cubin") and shutil.which("cuobjdump"):
        P(f"\n=== cuobjdump -res-usage {f} ===")
        r = subprocess.run(["cuobjdump", "-res-usage", f], capture_output=True, text=True)
        P("  " + "\n  ".join((r.stdout or r.stderr).splitlines()[:30]))
print("DONE")
