"""modal_kernel_analysis.py — NCU-SUBSTITUTE low-level kernel analysis on Modal B200.

ncu's hardware perf counters are blocked by Modal's gVisor sandbox (confirmed:
uname=4.19.0-gvisor, /dev/nvidia-caps + /proc/driver/nvidia/capabilities ABSENT,
LibraryNotLoaded across ncu 2025.1.1 AND 2025.3.1). So we recover the NCU-class
signal we actually wanted via mechanisms that survive gVisor:

  (1) STATIC per-kernel resources from the Triton compile cache: n_regs, n_spills,
      shared mem, num_warps — the EXACT register-budget signal the design-A "panel
      must be <=64 regs to co-reside" crux hinges on. + analytic occupancy.
  (2) SASS instruction mix (cuobjdump/nvdisasm on the cubin) — quantifies the
      "panel is latency-bound by the serial reflector reduction" claim (FFMA vs
      SHFL/BAR/LDS reduction/sync instruction counts).
  (3) proton (Triton's instrumentation profiler) — intra-kernel timing if available.
  (4) modern nsys CUPTI *tracing* (not perfmon) — does the kernel timeline + gaps
      survive gVisor? (separate interface from the blocked perf counters.)

Targets submission.py's QR at the dominant n=512 b=640 case (4 of 12 benchmarks).

Run: /Users/raymond/Downloads/SubPY/.modalenv/bin/modal run modal_kernel_analysis.py
"""
import pathlib, modal

HERE = pathlib.Path(__file__).parent
image = (
    modal.Image.from_registry("nvidia/cuda:13.0.1-devel-ubuntu24.04", add_python="3.12")
    .apt_install("cuda-nsight-systems-13-0")     # modern nsys (CUPTI tracing), matched to driver 580
    .pip_install("numpy")
    .pip_install("torch", index_url="https://download.pytorch.org/whl/cu128")
    .add_local_dir(str(HERE / "harness"), remote_path="/work", copy=True)
    .add_local_file(str(HERE / "submission.py"), remote_path="/work/submission.py", copy=True)
)
app = modal.App("qr-kernel-analysis")

# Blackwell B200 (sm_100) per-SM limits for analytic occupancy.
SM = dict(regs=65536, max_threads=2048, max_warps=64, max_blocks=32,
          smem=227 * 1024, reg_alloc_unit=256, warp_size=32)


@app.function(gpu="B200", image=image, timeout=1200)
def analyze():
    import subprocess, os, glob, json, math, shutil, collections

    LOG = []

    def out(s):
        print(s, flush=True)
        LOG.append(str(s))

    def sh(cmd, env=None, timeout=600, cap=12000):
        out(f"\n$ {cmd}")
        try:
            p = subprocess.run(cmd, shell=True, capture_output=True, text=True,
                               cwd="/work", env=env, timeout=timeout)
            o = (p.stdout or "") + (p.stderr or "")
            out(o[:cap])
            return o
        except Exception as e:
            out(f"[exc] {e}")
            return str(e)

    def occupancy(n_regs, n_warps, smem_bytes):
        """Theoretical occupancy of one block on a B200 SM."""
        threads = n_warps * SM["warp_size"]
        # registers are allocated per-warp, rounded up to reg_alloc_unit
        regs_per_warp = math.ceil(n_regs * SM["warp_size"] / SM["reg_alloc_unit"]) * SM["reg_alloc_unit"]
        regs_per_warp = max(regs_per_warp, SM["reg_alloc_unit"])
        warps_by_reg = SM["regs"] // regs_per_warp
        blocks_reg = warps_by_reg // n_warps if n_warps else 0
        blocks_smem = SM["smem"] // smem_bytes if smem_bytes > 0 else SM["max_blocks"]
        blocks_warp = SM["max_warps"] // n_warps if n_warps else 0
        blocks = max(0, min(blocks_reg, blocks_smem, blocks_warp, SM["max_blocks"]))
        active_warps = blocks * n_warps
        return dict(blocks=blocks, active_warps=active_warps,
                    occ=round(active_warps / SM["max_warps"], 3),
                    lim=("regs" if blocks == blocks_reg else
                         "smem" if blocks == blocks_smem else
                         "warps" if blocks == blocks_warp else "blocks"),
                    regs_per_warp=regs_per_warp)

    # ---- environment ----
    cache = "/tmp/tcache"
    env = dict(os.environ, TRITON_CACHE_DIR=cache, QR_N="512", QR_B="640",
               QR_COND="2", QR_WARM="6", QR_ITERS="1")
    cuobjdump = shutil.which("cuobjdump") or "/usr/local/cuda/bin/cuobjdump"

    out("=" * 70 + "\nSECTION 0: env\n" + "=" * 70)
    sh("uname -a")
    sh("nvidia-smi --query-gpu=name,driver_version --format=csv")
    sh(f"{cuobjdump} --version | head -3")
    sh(f"python -c \"import torch,triton;print('torch',torch.__version__,'triton',triton.__version__)\"", env=env)

    # ---- (1) compile the real kernels in-process + introspect CompiledKernel objects ----
    out("\n" + "=" * 70 + "\nSECTION 1: compile real QR kernels (n=512 b=640) + per-kernel resources\n" + "=" * 70)
    introspect = '''
import os, sys, json, importlib.util, gc
sys.path.insert(0, "/work")   # so `import task` / `import reference` resolve
import torch, triton
sys.modules.setdefault("task", __import__("task"))
spec = importlib.util.spec_from_file_location("uut", "submission.py")
mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
import reference
A = reference.generate_input(batch=640, n=512, cond=2, seed=999)
for _ in range(6): mod.custom_kernel(A.clone())
torch.cuda.synchronize()

os.makedirs("/tmp/cubins", exist_ok=True)
rows = []
seen = set()
# Walk every live Triton CompiledKernel (covers panel + all autotuned bmm configs).
for obj in gc.get_objects():
    try:
        cls = type(obj).__name__
        if cls != "CompiledKernel":
            continue
        name = getattr(obj, "name", None)
        nr = getattr(obj, "n_regs", None)
        ns = getattr(obj, "n_spills", None)
        md = getattr(obj, "metadata", None)
        nw = getattr(md, "num_warps", None)
        nstg = getattr(md, "num_stages", None)
        shared = getattr(md, "shared", None)
        key = (name, nw, shared, nr, ns)
        if name is None or key in seen:
            continue
        seen.add(key)
        cubin_path = ""
        try:
            asm = getattr(obj, "asm", {}) or {}
            if "cubin" in asm:
                cubin_path = f"/tmp/cubins/{name}_{nw}w_{nr}r_{len(rows)}.cubin"
                with open(cubin_path, "wb") as f:
                    f.write(asm["cubin"])
        except Exception as e:
            cubin_path = f"[cubin err {e}]"
        rows.append(dict(name=name, num_warps=nw, num_stages=nstg, n_regs=nr,
                         n_spills=ns, shared=shared, cubin=cubin_path))
    except Exception:
        continue
print("===KERN_JSON===")
print(json.dumps(rows))
print("===END_KERN_JSON===")
'''
    with open("/tmp/introspect.py", "w") as f:
        f.write(introspect)
    r = sh("python /tmp/introspect.py", env=env, cap=8000)
    uniq = []
    try:
        blob = r.split("===KERN_JSON===", 1)[1].split("===END_KERN_JSON===", 1)[0].strip()
        uniq = json.loads(blob)
    except Exception as e:
        out(f"[warn] could not parse kernel json: {e}")

    hdr = f"{'kernel':30s} {'warps':>5s} {'stg':>3s} {'regs':>5s} {'spill':>6s} {'smem':>7s} {'occ':>5s} {'limiter':>8s} {'act_w':>6s}"
    out(hdr)
    out("-" * len(hdr))
    panel_cubins = []
    for d in sorted(uniq, key=lambda x: (str(x.get("name")), x.get("num_warps") or 0)):
        name = str(d.get("name", "?"))
        nw = d.get("num_warps") or 0
        nstg = d.get("num_stages") or 0
        nr = d.get("n_regs") or 0
        ns = d.get("n_spills") or 0
        sm = d.get("shared") or 0
        occ = occupancy(nr, nw, sm) if (nr and nw) else dict(occ="?", lim="?", active_warps="?")
        out(f"{name[:30]:30s} {nw:5d} {nstg:3d} {nr:5d} {ns:6d} {sm:7d} {str(occ['occ']):>5s} {str(occ['lim']):>8s} {str(occ['active_warps']):>6s}")
        if "panel" in name and isinstance(d.get("cubin"), str) and d["cubin"].endswith(".cubin"):
            panel_cubins.append(d["cubin"])

    # ---- (2) SASS instruction mix for the panel kernel(s) ----
    out("\n" + "=" * 70 + "\nSECTION 2: SASS instruction mix — _panel_kernel (latency-bound check)\n" + "=" * 70)
    cubins = sorted(set(panel_cubins))
    if not cubins:
        cubins = sorted(glob.glob("/tmp/cubins/*panel*.cubin")) or sorted(glob.glob("/tmp/cubins/*.cubin"))
        out(f"[note] no panel cubin from introspection; scanning {len(cubins)} dumped cubins")
    for cb in cubins[:4]:
        sass = sh(f"{cuobjdump} -sass {cb} 2>/dev/null | head -4000", cap=2000)
        # opcode histogram
        hist = collections.Counter()
        total = 0
        for line in sass.splitlines():
            line = line.strip()
            # SASS lines look like:  /*0a00*/  FFMA R4, R5, R6, R7 ;
            if "/*" in line and "*/" in line:
                tail = line.split("*/", 1)[1].strip()
                op = tail.split()[0].rstrip("@!").split(".")[0] if tail else ""
                if op and op.isupper() and len(op) >= 2:
                    hist[op] += 1
                    total += 1
        if total:
            out(f"\n[{os.path.basename(cb)}] {total} SASS instrs — top 25 opcodes:")
            # group into families
            fam = collections.Counter()
            for op, c in hist.items():
                if op in ("FFMA", "FADD", "FMUL", "FSETP", "MUFU"): fam["FP-math"] += c
                elif op.startswith("LDS") or op.startswith("STS"): fam["shared-mem"] += c
                elif op.startswith("LDG") or op.startswith("STG") or op.startswith("LD") or op.startswith("ST"): fam["global-mem"] += c
                elif op in ("BAR", "DEPBAR", "MEMBAR"): fam["barrier"] += c
                elif op in ("SHFL", "REDUX", "VOTE"): fam["warp-reduce"] += c
                elif op.startswith("BRA") or op.startswith("BSSY") or op.startswith("BSYNC") or op in ("EXIT", "JMP", "CALL", "RET"): fam["control-flow"] += c
                else: fam["other"] += c
            for op, c in hist.most_common(25):
                out(f"    {op:12s} {c:6d}  {100*c/total:5.1f}%")
            out("  --- families ---")
            for f, c in fam.most_common():
                out(f"    {f:12s} {c:6d}  {100*c/total:5.1f}%")

    # ---- (3) kineto per-kernel timeline (the gVisor-SAFE timeline; nsys is dead — see note) ----
    # NOTE: nsys is gVisor-blocked too — `ConvertGpuTicksToSyncNs InternalErrorException`
    # (GPU-tick<->wall-clock sync fails under gVisor), so no .nsys-rep is ever produced.
    # torch.profiler/kineto uses in-process CUPTI activity records (no global clock sync)
    # and DOES work — it's the only kernel-level timeline we can get on Modal.
    out("\n" + "=" * 70 + "\nSECTION 3: kineto per-kernel timeline (gVisor-safe) — durations + launch gaps\n" + "=" * 70)
    out("[note] nsys is ALSO gVisor-blocked (GPU clock-sync InternalError, no report). kineto works.")
    kineto = '''
import sys, importlib.util, statistics
sys.path.insert(0, "/work")
import torch
sys.modules.setdefault("task", __import__("task"))
spec = importlib.util.spec_from_file_location("uut", "submission.py")
mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
import reference
from torch.profiler import profile, ProfilerActivity
A = reference.generate_input(batch=640, n=512, cond=2, seed=999)
for _ in range(6): mod.custom_kernel(A.clone())
torch.cuda.synchronize()
REPS=10
with profile(activities=[ProfilerActivity.CUDA]) as prof:
    for _ in range(REPS): mod.custom_kernel(A.clone())
    torch.cuda.synchronize()

def cat(n):
    n=n.lower()
    if "panel" in n: return "PANEL"
    if "_x3" in n or "bmm_x3" in n: return "tf32x3-GEMM(trailing)"
    if any(s in n for s in ("trsm","triangular","potrf","getrf","ormqr","geqrf","larf","syrk","trmm")): return "cuSOLVER/solve"
    if any(s in n for s in ("gemm","cutlass","tensorop","cublas","sm100","sm90","ampere")): return "cuBLAS-GEMM"
    if any(s in n for s in ("elementwise","tril","triu","copy","fill","cast","reduce","norm","sub","add","mul","div","cat","arange","memcpy","reciprocal","sign","sqrt","where","index","zero")): return "glue"
    return "other"

evs=[e for e in prof.key_averages() if (getattr(e,"self_device_time_total",0) or 0)>0]
import collections
bycat=collections.Counter()
perker=[]
for e in evs:
    t=(e.self_device_time_total or 0)/REPS
    bycat[cat(e.key)]+=t
    perker.append((e.key, t, e.count/REPS))
tot=sum(bycat.values()) or 1
print("===KINETO===")
print(f"total GPU us/iter: {tot:.1f}")
print("by category:")
for c,v in sorted(bycat.items(), key=lambda x:-x[1]):
    print(f"  {c:24s} {v:8.1f} us  {100*v/tot:5.1f}%")
print("top 18 kernels by self GPU time (us/iter, launches/iter):")
for k,t,n in sorted(perker, key=lambda x:-x[1])[:18]:
    print(f"  {t:8.2f} us  x{n:6.1f}  {k[:80]}")
print("===END_KINETO===")
'''
    with open("/work/kineto_run.py", "w") as f:
        f.write(kineto)
    r = sh("python /work/kineto_run.py", env=env, cap=8000)

    out("\n" + "=" * 70 + "\nDONE\n" + "=" * 70)
    return "\n".join(LOG)


@app.local_entrypoint()
def main():
    print(analyze.remote())
