"""modal_ncu_lite_deep_probe.py — deeper NCU-lite probe for Modal B200.

Modal/gVisor blocks real ncu/nsys, so this script recovers lower-level signal with:
  - Triton CompiledKernel resources: regs, spills, smem, analytic occupancy.
  - cuobjdump SASS opcode families for production and ablation kernels.
  - kineto per-kernel timeline for the real n=512 b=640 QR workload.
  - panel ablations that isolate load/store, reduction/norm, rank-1 update, and full panel.

Run:
  modal run modal_ncu_lite_deep_probe.py
"""
import pathlib
import modal

HERE = pathlib.Path(__file__).parent
# Pinned 2026-09-11. The original runs did NOT pin torch, which means the lab numbers in
# results/ cannot be reproduced bit-for-bit -- a real gap in a project whose central claim
# is that codegen-sensitive results do not transfer. 2.12.0 is the version modal_cute_lab.py
# was pinned to during the same sessions. If it fails to resolve, set TORCH_SPEC = "torch"
# and record what actually got installed (every lab run now prints it).
TORCH_SPEC = "torch==2.12.0"

image = (
    modal.Image.from_registry("nvidia/cuda:13.0.1-devel-ubuntu24.04", add_python="3.12")
    .pip_install("numpy")
    .pip_install(TORCH_SPEC, index_url="https://download.pytorch.org/whl/cu128")
    .add_local_dir(str(HERE / "harness"), remote_path="/work", copy=True)
    .add_local_file(str(HERE / "submission.py"), remote_path="/work/submission.py", copy=True)
)
app = modal.App("qr-ncu-lite-deep-probe")

SM = dict(regs=65536, max_warps=64, max_blocks=32, smem=227 * 1024,
          reg_alloc_unit=256, warp_size=32)


@app.function(gpu="B200", image=image, timeout=1500)
def run_probe():
    import collections
    import json
    import math
    import os
    import re
    import shutil
    import subprocess
    import textwrap

    log = []

    def out(s=""):
        print(s, flush=True)
        log.append(str(s))

    def sh(cmd, env=None, timeout=900, cap=20000):
        out(f"\n$ {cmd}")
        p = subprocess.run(cmd, shell=True, cwd="/work", env=env, timeout=timeout,
                           capture_output=True, text=True)
        txt = (p.stdout or "") + (p.stderr or "")
        out(txt[:cap])
        return txt

    def occupancy(n_regs, n_warps, smem_bytes):
        if not n_regs or not n_warps:
            return {"occ": "?", "lim": "?", "active_warps": "?", "blocks": "?"}
        regs_per_warp = math.ceil(n_regs * SM["warp_size"] / SM["reg_alloc_unit"]) * SM["reg_alloc_unit"]
        blocks_reg = (SM["regs"] // regs_per_warp) // n_warps
        blocks_smem = SM["smem"] // smem_bytes if smem_bytes > 0 else SM["max_blocks"]
        blocks_warp = SM["max_warps"] // n_warps
        blocks = max(0, min(blocks_reg, blocks_smem, blocks_warp, SM["max_blocks"]))
        active_warps = blocks * n_warps
        lim = "regs" if blocks == blocks_reg else "smem" if blocks == blocks_smem else "warps"
        return {"blocks": blocks, "active_warps": active_warps,
                "occ": round(active_warps / SM["max_warps"], 3), "lim": lim}

    def sass_summary(cubin):
        cuobjdump = shutil.which("cuobjdump") or "/usr/local/cuda/bin/cuobjdump"
        p = subprocess.run(f"{cuobjdump} -sass {cubin} 2>/dev/null", shell=True,
                           capture_output=True, text=True, timeout=120)
        hist = collections.Counter()
        total = 0
        for line in p.stdout.splitlines():
            if "/*" not in line or "*/" not in line:
                continue
            tail = line.split("*/", 1)[1].strip()
            if not tail:
                continue
            token = tail.split()[0]
            # Drop predication and suffixes: @!P0 FFMA.SAT -> FFMA
            if token.startswith("@"):
                parts = tail.split()
                token = parts[1] if len(parts) > 1 else token
            op = token.split(".")[0].rstrip(";")
            if op.isupper() and len(op) >= 2:
                hist[op] += 1
                total += 1
        fam = collections.Counter()
        for op, c in hist.items():
            if op in ("FFMA", "FADD", "FMUL", "FSEL", "FSETP", "MUFU", "RRO"):
                fam["fp_math_or_select"] += c
            elif op.startswith(("IMAD", "IADD", "ISETP", "LOP", "LEA", "P2R", "S2R", "CS2R", "SEL", "VIADD")):
                fam["int_addr_pred"] += c
            elif op in ("SHFL", "REDUX", "VOTE"):
                fam["warp_reduce"] += c
            elif op.startswith(("LDS", "STS")):
                fam["shared_mem"] += c
            elif op.startswith(("LDG", "STG", "LD", "ST")):
                fam["global_or_const_mem"] += c
            elif op.startswith(("LDL", "STL")):
                fam["local_spill_mem"] += c
            elif op in ("BAR", "DEPBAR", "MEMBAR"):
                fam["barrier"] += c
            elif op.startswith(("BRA", "BSSY", "BSYNC")) or op in ("EXIT", "JMP", "RET", "CALL"):
                fam["control_flow"] += c
            else:
                fam["other"] += c
        return total, hist, fam

    out("=" * 78)
    out("SECTION 0: environment")
    out("=" * 78)
    env = dict(os.environ, TRITON_CACHE_DIR="/tmp/tcache")
    sh("uname -a")
    sh("nvidia-smi --query-gpu=name,driver_version --format=csv")
    sh("python - <<'PY'\nimport torch,triton\nprint('torch', torch.__version__)\nprint('triton', triton.__version__)\nprint('device', torch.cuda.get_device_name(0))\nPY", env=env)

    probe_py = r'''
import gc, json, os, sys, importlib.util, collections
import torch, triton, triton.language as tl
from torch.profiler import profile, ProfilerActivity

sys.path.insert(0, "/work")
import reference
sys.modules.setdefault("task", __import__("task"))
spec = importlib.util.spec_from_file_location("uut", "submission.py")
sub = importlib.util.module_from_spec(spec); spec.loader.exec_module(sub)

torch.backends.cuda.matmul.allow_tf32 = True
try: torch.set_float32_matmul_precision("high")
except Exception: pass

def clear_l2():
    torch.empty((32, 1024, 1024), dtype=torch.int64, device="cuda").fill_(0)

def median_us(fn, iters=28):
    for _ in range(6):
        fn()
    torch.cuda.synchronize()
    vals = []
    for _ in range(iters):
        clear_l2(); torch.cuda.synchronize()
        s = torch.cuda.Event(enable_timing=True); e = torch.cuda.Event(enable_timing=True)
        s.record(); fn(); e.record(); torch.cuda.synchronize()
        vals.append(s.elapsed_time(e) * 1000.0)
    vals.sort()
    return vals[len(vals) // 2]

@triton.jit
def _panel_full_probe(Hptr, tauptr, N, K, M, BWID, BN: tl.constexpr, BCOLS: tl.constexpr):
    bid = tl.program_id(0)
    Hb = Hptr + bid * N * N
    taub = tauptr + bid * N
    rar = tl.arange(0, BN); car = tl.arange(0, BCOLS)
    ptr = Hb + (K + rar)[:, None] * N + (K + car)[None, :]
    tmask = (rar[:, None] < M) & (car[None, :] < BWID)
    tile = tl.load(ptr, mask=tmask, other=0.0)
    for j in range(BCOLS):
        colj = tl.sum(tl.where(car[None, :] == j, tile, 0.0), axis=1)
        alpha = tl.sum(tl.where(rar == j, colj, 0.0))
        xnorm2 = tl.sum(tl.where(rar > j, colj * colj, 0.0))
        normfull = tl.sqrt(alpha * alpha + xnorm2)
        sgn = tl.where(alpha >= 0.0, 1.0, -1.0)
        beta = -sgn * normfull
        need = xnorm2 > 0.0
        scale = tl.where(need, 1.0 / (alpha - beta), 0.0)
        tau_j = tl.where(need, (beta - alpha) / beta, 0.0)
        v = tl.where(rar > j, colj * scale, 0.0)
        v = tl.where(rar == j, 1.0, v)
        w = tl.sum(v[:, None] * tile, axis=0)
        upd = tile - tau_j * (v[:, None] * w[None, :])
        tile = tl.where(car[None, :] > j, upd, tile)
        diagval = tl.where(need, beta, alpha)
        newcol = tl.where(rar > j, colj * scale, colj)
        newcol = tl.where(rar == j, diagval, newcol)
        tile = tl.where(car[None, :] == j, newcol[:, None], tile)
        tl.store(taub + K + j, tau_j, mask=(j < BWID))
    tl.store(ptr, tile, mask=tmask)

@triton.jit
def _panel_mem_probe(Hptr, tauptr, N, K, M, BWID, BN: tl.constexpr, BCOLS: tl.constexpr):
    bid = tl.program_id(0)
    Hb = Hptr + bid * N * N
    rar = tl.arange(0, BN); car = tl.arange(0, BCOLS)
    ptr = Hb + (K + rar)[:, None] * N + (K + car)[None, :]
    tmask = (rar[:, None] < M) & (car[None, :] < BWID)
    tile = tl.load(ptr, mask=tmask, other=0.0)
    tl.store(ptr, tile, mask=tmask)

@triton.jit
def _panel_reduce_probe(Hptr, tauptr, N, K, M, BWID, BN: tl.constexpr, BCOLS: tl.constexpr):
    bid = tl.program_id(0)
    Hb = Hptr + bid * N * N
    taub = tauptr + bid * N
    rar = tl.arange(0, BN); car = tl.arange(0, BCOLS)
    ptr = Hb + (K + rar)[:, None] * N + (K + car)[None, :]
    tmask = (rar[:, None] < M) & (car[None, :] < BWID)
    tile = tl.load(ptr, mask=tmask, other=0.0)
    for j in range(BCOLS):
        colj = tl.sum(tl.where(car[None, :] == j, tile, 0.0), axis=1)
        alpha = tl.sum(tl.where(rar == j, colj, 0.0))
        xnorm2 = tl.sum(tl.where(rar > j, colj * colj, 0.0))
        normfull = tl.sqrt(alpha * alpha + xnorm2)
        sgn = tl.where(alpha >= 0.0, 1.0, -1.0)
        beta = -sgn * normfull
        need = xnorm2 > 0.0
        tau_j = tl.where(need, (beta - alpha) / beta, 0.0)
        tl.store(taub + K + j, tau_j, mask=(j < BWID))

@triton.jit
def _panel_update_probe(Hptr, tauptr, N, K, M, BWID, BN: tl.constexpr, BCOLS: tl.constexpr):
    bid = tl.program_id(0)
    Hb = Hptr + bid * N * N
    rar = tl.arange(0, BN); car = tl.arange(0, BCOLS)
    ptr = Hb + (K + rar)[:, None] * N + (K + car)[None, :]
    tmask = (rar[:, None] < M) & (car[None, :] < BWID)
    tile = tl.load(ptr, mask=tmask, other=0.0)
    for j in range(BCOLS):
        colj = tl.sum(tl.where(car[None, :] == j, tile, 0.0), axis=1)
        v = tl.where(rar >= j, colj, 0.0)
        w = tl.sum(v[:, None] * tile, axis=0)
        upd = tile - (v[:, None] * w[None, :]) * 0.001
        tile = tl.where(car[None, :] > j, upd, tile)
    tl.store(ptr, tile, mask=tmask)

@triton.jit
def _panel_mask_probe(Hptr, tauptr, N, K, M, BWID, BN: tl.constexpr, BCOLS: tl.constexpr):
    bid = tl.program_id(0)
    Hb = Hptr + bid * N * N
    rar = tl.arange(0, BN); car = tl.arange(0, BCOLS)
    ptr = Hb + (K + rar)[:, None] * N + (K + car)[None, :]
    tmask = (rar[:, None] < M) & (car[None, :] < BWID)
    tile = tl.load(ptr, mask=tmask, other=0.0)
    for j in range(BCOLS):
        # Predication/addressing pressure with minimal arithmetic.
        a = tl.where(car[None, :] > j, tile, 0.0)
        b = tl.where(rar[:, None] > j, a, tile)
        tile = tl.where((car[None, :] == j) | (rar[:, None] == j), b, a)
    tl.store(ptr, tile, mask=tmask)

def launch_panel(kern, H, tau, nw, m=512, bwid=64):
    BN = triton.next_power_of_2(m)
    BCOLS = triton.next_power_of_2(bwid)
    kern[(H.shape[0],)](H, tau, H.shape[1], 0, m, bwid, BN=BN, BCOLS=BCOLS, num_warps=nw)

def kernel_resources():
    os.makedirs("/tmp/cubins", exist_ok=True)
    rows = []
    seen = set()
    for obj in gc.get_objects():
        try:
            if type(obj).__name__ != "CompiledKernel":
                continue
            name = getattr(obj, "name", None)
            md = getattr(obj, "metadata", None)
            row = dict(name=name,
                       num_warps=getattr(md, "num_warps", None),
                       num_stages=getattr(md, "num_stages", None),
                       n_regs=getattr(obj, "n_regs", None),
                       n_spills=getattr(obj, "n_spills", None),
                       shared=getattr(md, "shared", None))
            key = tuple(row.items())
            if name is None or key in seen:
                continue
            seen.add(key)
            asm = getattr(obj, "asm", {}) or {}
            if "cubin" in asm:
                path = f"/tmp/cubins/{len(rows):02d}_{name}_{row['num_warps']}w_{row['n_regs']}r.cubin"
                with open(path, "wb") as f:
                    f.write(asm["cubin"])
                row["cubin"] = path
            rows.append(row)
        except Exception:
            pass
    return rows

def kineto_real_qr():
    A = reference.generate_input(batch=640, n=512, cond=2, seed=999)
    for _ in range(6):
        sub.custom_kernel(A.clone())
    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        for _ in range(8):
            sub.custom_kernel(A.clone())
        torch.cuda.synchronize()
    def cat(n):
        s = n.lower()
        if "panel" in s: return "panel"
        if "_x3" in s or "bmm_x3" in s: return "tf32x3"
        if any(x in s for x in ("trsm", "triangular", "geqrf", "ormqr", "potrf", "getrf")): return "solve_qr"
        if any(x in s for x in ("gemm", "cublas", "cutlass", "tensorop")): return "cublas"
        if any(x in s for x in ("tril", "triu", "copy", "memcpy", "elementwise", "fill", "where", "sub", "add", "mul", "div", "zero", "arange", "reciprocal")): return "glue"
        return "other"
    bycat = collections.Counter()
    rows = []
    for e in prof.key_averages():
        t = (getattr(e, "self_device_time_total", 0) or 0) / 8.0
        if t <= 0:
            continue
        bycat[cat(e.key)] += t
        rows.append({"key": e.key, "us": t, "launches": e.count / 8.0, "cat": cat(e.key)})
    return {"bycat": dict(bycat), "top": sorted(rows, key=lambda r: -r["us"])[:28]}

def run_panel_ablation():
    B, n, m, b = 640, 512, 512, 64
    variants = [
        ("mem_load_store", _panel_mem_probe),
        ("mask_predication", _panel_mask_probe),
        ("reduce_norm_tau", _panel_reduce_probe),
        ("rank1_update_no_norm", _panel_update_probe),
        ("full_householder", _panel_full_probe),
    ]
    timings = []
    for name, kern in variants:
        H = torch.randn(B, n, n, device="cuda")
        tau = torch.zeros(B, n, device="cuda")
        us = median_us(lambda kern=kern, H=H, tau=tau: launch_panel(kern, H, tau, 8, m, b), iters=24)
        timings.append({"name": name, "us": us, "per_col_us": us / b})
    # Full panel warp sweep, same kernel body.
    sweep = []
    for nw in (4, 8, 16, 32):
        H = torch.randn(B, n, n, device="cuda")
        tau = torch.zeros(B, n, device="cuda")
        try:
            us = median_us(lambda nw=nw, H=H, tau=tau: launch_panel(_panel_full_probe, H, tau, nw, m, b), iters=18)
            sweep.append({"num_warps": nw, "us": us, "per_col_us": us / b})
        except Exception as e:
            sweep.append({"num_warps": nw, "error": repr(e)})
    return {"timings": timings, "warp_sweep": sweep}

timeline = kineto_real_qr()
ablation = run_panel_ablation()
resources = kernel_resources()
print("===DEEP_JSON===")
print(json.dumps({"timeline": timeline, "ablation": ablation, "resources": resources}))
print("===END_DEEP_JSON===")
'''

    with open("/work/deep_probe_inner.py", "w") as f:
        f.write(probe_py)

    out("\n" + "=" * 78)
    out("SECTION 1: run QR timeline + panel ablation + collect compiled kernels")
    out("=" * 78)
    raw = sh("python /work/deep_probe_inner.py", env=env, timeout=1200, cap=30000)
    m = re.search(r"===DEEP_JSON===\n(.*)\n===END_DEEP_JSON===", raw, re.S)
    if not m:
        out("[error] failed to parse DEEP_JSON")
        return "\n".join(log)
    data = json.loads(m.group(1))

    out("\n" + "=" * 78)
    out("SECTION 2: kineto timeline for real submission.py, n=512 b=640")
    out("=" * 78)
    bycat = data["timeline"]["bycat"]
    total = sum(bycat.values()) or 1.0
    for k, v in sorted(bycat.items(), key=lambda kv: -kv[1]):
        out(f"{k:12s} {v:9.1f} us/iter  {100*v/total:5.1f}%")
    out("\nTop kernels:")
    for r in data["timeline"]["top"][:18]:
        out(f"{r['us']:9.1f} us  x{r['launches']:5.1f}  {r['cat']:10s}  {r['key'][:95]}")

    out("\n" + "=" * 78)
    out("SECTION 3: panel ablation timings, B=640 n=512 M=512 ib=64 nw=8")
    out("=" * 78)
    full = next((r["us"] for r in data["ablation"]["timings"] if r["name"] == "full_householder"), None)
    for r in data["ablation"]["timings"]:
        pct = 100 * r["us"] / full if full else 0
        out(f"{r['name']:22s} {r['us']:9.1f} us  {r['per_col_us']:7.2f} us/col  {pct:5.1f}% of full")
    out("\nFull panel warp sweep:")
    for r in data["ablation"]["warp_sweep"]:
        if "error" in r:
            out(f"nw={r['num_warps']:2d}: {r['error']}")
        else:
            out(f"nw={r['num_warps']:2d}: {r['us']:9.1f} us  {r['per_col_us']:7.2f} us/col")

    out("\n" + "=" * 78)
    out("SECTION 4: compiled resources + occupancy")
    out("=" * 78)
    rows = sorted(data["resources"], key=lambda r: (str(r.get("name")), r.get("num_warps") or 0, r.get("n_regs") or 0))
    hdr = f"{'kernel':28s} {'warps':>5s} {'regs':>5s} {'spill':>5s} {'smem':>7s} {'occ':>5s} {'lim':>6s} {'cubin'}"
    out(hdr)
    out("-" * len(hdr))
    for r in rows:
        occ = occupancy(r.get("n_regs") or 0, r.get("num_warps") or 0, r.get("shared") or 0)
        out(f"{str(r.get('name'))[:28]:28s} {r.get('num_warps') or 0:5d} {r.get('n_regs') or 0:5d} "
            f"{r.get('n_spills') or 0:5d} {r.get('shared') or 0:7d} {str(occ['occ']):>5s} "
            f"{str(occ['lim']):>6s} {r.get('cubin', '')}")

    out("\n" + "=" * 78)
    out("SECTION 5: SASS family summaries for selected kernels")
    out("=" * 78)
    interesting = []
    names = ("_panel_kernel", "_panel_full_probe", "_panel_reduce_probe",
             "_panel_update_probe", "_panel_mask_probe", "_bmm_x3_kernel", "_bmm_x3_sub_kernel")
    for nm in names:
        cands = [r for r in rows if r.get("name") == nm and r.get("cubin")]
        if cands:
            # pick the highest-register variant, usually the binding full-size compile.
            interesting.append(max(cands, key=lambda r: r.get("n_regs") or 0))
    seen = set()
    for r in interesting:
        cubin = r.get("cubin")
        if not cubin or cubin in seen:
            continue
        seen.add(cubin)
        total_i, hist, fam = sass_summary(cubin)
        out(f"\n{r.get('name')}  warps={r.get('num_warps')} regs={r.get('n_regs')} spills={r.get('n_spills')}  "
            f"static_sass={total_i}")
        for k, v in fam.most_common():
            out(f"  {k:20s} {v:6d}  {100*v/max(total_i,1):5.1f}%")
        out("  top opcodes: " + ", ".join(f"{op}:{cnt}" for op, cnt in hist.most_common(12)))

    out("\n" + "=" * 78)
    out("INTERPRETATION HINTS")
    out("=" * 78)
    out("If mem_load_store is tiny, HBM traffic is not the panel wall.")
    out("If reduce_norm_tau + rank1_update_no_norm roughly explain full, the gap is within the serial reflector loop.")
    out("High regs/spills + low occupancy mean the kernel cannot co-reside with a tcgen05 worker.")
    out("Large int_addr_pred/SASS share means branchless triangular masking/predication is a first-order cost.")
    return "\n".join(log)


@app.local_entrypoint()
def main():
    txt = run_probe.remote()
    print(txt)
    out_path = HERE / "results" / "ncu_lite_deep_probe.txt"
    out_path.parent.mkdir(exist_ok=True)
    out_path.write_text(txt)
    print(f"\n[probe] saved to {out_path}")
