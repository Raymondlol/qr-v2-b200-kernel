"""CuTe-DSL lab on a FAITHFUL replica of the qr_v2 eval image.

Why this exists: the old `modal_cutedsl_test.py` used debian_slim+torch-cu128, which pulled
cuda-python 13.3.1 against torch 2.11 (cuda-bindings<13) -> native ABI mismatch -> `LLVM ERROR:
unsupported operation` SIGABRT. The REAL eval image (from gpu-mode/kernelbot main
src/runners/modal_runner.py) is nvidia/cuda:12.9.1-devel + python 3.13 + torch==2.12.0 +
nvidia-cutlass-dsl==4.5.2 + cuda-python[all]==13.0 + cuda-core[cu13]. Replicating it (a) fixes the
crash and (b) makes this Modal loop a faithful proxy for the board -> we can iterate cute-dsl here
instead of burning gpumode submissions.

Run:
  /Users/raymond/Downloads/SubPY/.modalenv/bin/modal run modal_cute_lab.py            # info + smoke
  /Users/raymond/Downloads/SubPY/.modalenv/bin/modal run modal_cute_lab.py::info      # introspection only
"""
import pathlib, modal
HERE = pathlib.Path(__file__).parent

# ── FAITHFUL eval-image replica, TRIMMED to what cute-dsl needs ──────────────────────────────────
# The eval (kernelbot main) base = nvidia/cuda:12.9.1-devel + python 3.13 + torch==2.12.0 +
# nvidia-cutlass-dsl==4.5.2 + cuda-python[all]==13.0 + cuda-core[cu13] (+ tinygrad/helion/cupynumeric/
# pytest/yaml which are EVAL-HARNESS deps, irrelevant to whether OUR cute kernel compiles+runs+is
# correct+fast). We pin the versions that matter (torch/cutlass-dsl/cuda-python) EXACTLY and drop the
# rest — the dropped cupynumeric pulled scikit-learn/zarr/h5py and tripped a pypi-mirror HASH MISMATCH.
# pip split into layers => smaller per-layer hash sets => robust + cache-friendly.
EVAL = (
    modal.Image.from_registry("nvidia/cuda:12.9.1-devel-ubuntu24.04", add_python="3.13")
    .apt_install("git", "curl", "gcc-13", "g++-13", "clang-18")
    .pip_install("numpy~=2.3")
    .pip_install("torch==2.12.0")
    .pip_install("nvidia-cutlass-dsl==4.5.2", "cuda-core[cu13]", "cuda-python==13.0")
    .run_commands("git clone --depth 1 --branch v4.5.1 https://github.com/NVIDIA/cutlass.git /opt/cutlass")
    .env({"CUTLASS_PATH": "/opt/cutlass",
          "CPLUS_INCLUDE_PATH": "/opt/cutlass/include:/opt/cutlass/tools/util/include"})
)

# Dev-loop image: EVAL + the experiments/ dir mounted at /work (copy=False → updates each run, no
# rebuild). Lets us iterate our own cute candidate files: edit experiments/cute_*.py → modal run.
LAB = EVAL.add_local_dir(str(HERE / "experiments"), remote_path="/work", copy=False)

app = modal.App("cute-lab")


@app.function(gpu="B200", image=LAB, timeout=1200)
def run_candidate(script: str):
    """Exec a cute candidate file from experiments/ (mounted at /work) on B200. PRINTS stdout+stderr
    (so it shows regardless of how the function is invoked) and also returns it."""
    import subprocess, sys
    p = subprocess.run([sys.executable, script], cwd="/work", capture_output=True, text=True)
    out = p.stdout + "\n--- STDERR (tail) ---\n" + p.stderr[-6000:]
    print(out, flush=True)
    return out


@app.function(gpu="B200", image=EVAL, timeout=600)
def info():
    """Introspection ONLY (no GPU codegen -> cannot SIGABRT from a kernel bug). Prints the AUTHORITATIVE
    4.5.2 API surface: versions, package tree, the warp-spec/persistent/pipeline/tcgen05/TMA classes and
    their signatures. This is ground-truth for the port (more reliable than web docs)."""
    import sys, os, shutil, inspect, pkgutil, importlib, traceback
    P = lambda *a: print(*a, file=sys.stdout, flush=True)

    P("===== ENV =====")
    P("nvcc:", shutil.which("nvcc"))
    P("python:", sys.version.split()[0])
    P("CUTLASS_PATH:", os.environ.get("CUTLASS_PATH"), "exists:", os.path.isdir("/opt/cutlass"))
    for m in ["torch", "cutlass", "cuda", "cuda.core", "cuda.bindings", "triton", "tinygrad", "helion"]:
        try:
            mod = importlib.import_module(m)
            P(f"  {m}: v{getattr(mod,'__version__','?')}")
        except Exception as e:
            P(f"  {m}: FAIL {type(e).__name__}: {str(e)[:120]}")

    import cutlass
    P("\ncutlass.__file__:", getattr(cutlass, "__file__", "?"))
    pkg_root = os.path.dirname(cutlass.__file__)
    P("pkg_root:", pkg_root)

    # Walk the cutlass package tree (submodule names only).
    P("\n===== cutlass submodule tree (depth-limited) =====")
    try:
        for m in pkgutil.walk_packages(cutlass.__path__, prefix="cutlass."):
            depth = m.name.count(".")
            if depth <= 3:
                P(("  " * depth) + m.name + ("  [pkg]" if m.ispkg else ""))
    except Exception:
        P(traceback.format_exc()[-1500:])

    # Hunt for the key class names across the package by grepping source files.
    P("\n===== KEY API symbols (grep package source) =====")
    targets = [
        "PipelineTmaUmma", "PipelineUmmaAsync", "PipelineTmaAsync", "PipelineAsync",
        "MbarrierArray", "PipelineState", "CooperativeGroup",
        "StaticPersistentTileScheduler", "PersistentTileScheduler", "TileScheduler",
        "warp_specialize", "warpgroup_reg_alloc", "warpgroup_reg_dealloc", "setmaxnreg",
        "tcgen05", "TmemAllocator", "TensorMemoryLayout", "make_tiled_mma", "tiled_mma",
        "TensorDescriptor", "NVMMASharedLayout", "tma", "elect_one", "NamedBarrier", "two_ctas",
    ]
    hits = {t: [] for t in targets}
    for dirpath, _, files in os.walk(pkg_root):
        for fn in files:
            if not fn.endswith(".py"):
                continue
            fp = os.path.join(dirpath, fn)
            try:
                txt = open(fp, "r", errors="ignore").read()
            except Exception:
                continue
            rel = os.path.relpath(fp, pkg_root)
            for t in targets:
                if t in txt:
                    # record class/def definition lines specifically
                    for ln in txt.splitlines():
                        s = ln.strip()
                        if (s.startswith("class " + t) or s.startswith("def " + t)
                                or s.startswith("class " + t + "(") or ("def " + t + "(") in s):
                            hits[t].append(f"cutlass/{rel}: {s[:140]}")
    for t in targets:
        if hits[t]:
            P(f"\n## {t}")
            for h in hits[t][:6]:
                P("   " + h)
        else:
            P(f"\n## {t}: (no class/def; substring may still appear)")

    # Try to print __init__ signatures for the heavy pipeline/scheduler classes.
    P("\n===== signatures (best-effort) =====")
    probes = [
        ("cutlass.pipeline", ["PipelineTmaUmma", "PipelineUmmaAsync", "PipelineTmaAsync",
                              "PipelineState", "CooperativeGroup", "MbarrierArray"]),
        ("cutlass.utils", ["StaticPersistentTileScheduler", "PersistentTileScheduler"]),
        ("cutlass.cute.nvgpu", []),
    ]
    for modname, names in probes:
        try:
            mod = importlib.import_module(modname)
            P(f"\n[{modname}] dir -> {[n for n in dir(mod) if not n.startswith('_')][:40]}")
            for nm in names:
                obj = getattr(mod, nm, None)
                if obj is None:
                    continue
                try:
                    sig = inspect.signature(obj.__init__ if isinstance(obj, type) else obj)
                    P(f"   {nm}{sig}")
                except Exception as e:
                    P(f"   {nm}: <no sig: {type(e).__name__}>")
        except Exception as e:
            P(f"[{modname}] import FAIL: {type(e).__name__}: {str(e)[:120]}")

    # Find example .py kernels shipped anywhere on the image.
    P("\n===== example kernels on image =====")
    found = 0
    for base in ["/opt/cutlass", os.path.dirname(pkg_root)]:
        for dirpath, _, files in os.walk(base):
            if "example" in dirpath.lower() and "python" in dirpath.lower():
                for fn in files:
                    if fn.endswith(".py"):
                        P("  " + os.path.join(dirpath, fn))
                        found += 1
                        if found > 60:
                            break
            if found > 60:
                break
        if found > 60:
            break
    P(f"(examples found: {found})")
    return "info() done"


@app.function(gpu="B200", image=EVAL, timeout=600)
def kernel_smoke():
    """M0 K1 gate: confirm the cute-dsl JIT→launch→correct loop works on our Modal B200 eval-replica
    image with NO nvcc-for-the-kernel (driver PTX JIT, the deployable path). Two checks:
      (1) run the SHIPPED elementwise_add example from /opt/cutlass (zero transcription risk — if it
          ref-checks PASS, the whole loop is validated end-to-end), and
      (2) author our OWN minimal @cute.kernel using the canonical TV-layout/tiled-copy/fragment idiom
          (proves we can WRITE kernels, not just run theirs). The earlier SIGSEGV was naive scalar
          tensor[i] indexing on a from_dlpack tensor + missing .mark_layout_dynamic()."""
    import sys, traceback, importlib.util
    P = lambda *a: print(*a, file=sys.stdout, flush=True)
    import torch, cutlass
    import cutlass.cute as cute
    from cutlass.cute.runtime import from_dlpack
    P("smoke: imports OK, cutlass v", getattr(cutlass, "__version__", "?"),
      "torch", torch.__version__)

    # ---- (1) the shipped example (the decisive zero-risk green check) --------------------------
    ex = "/opt/cutlass/examples/python/CuTeDSL/cute/ampere/kernel/elementwise/elementwise_add.py"
    P(f"\nsmoke(1): running SHIPPED {ex.split('/')[-1]} with ref-check ...")
    try:
        spec = importlib.util.spec_from_file_location("_ew", ex)
        mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
        mod.run_elementwise_add(1024, 512, dtype=cutlass.Float32,
                                is_a_dynamic_layout=True, is_b_dynamic_layout=True,
                                is_result_dynamic_layout=True, skip_ref_check=False, benchmark=False)
        P("smoke(1): SHIPPED elementwise_add ref-check PASSED  ✅")
    except Exception:
        P("smoke(1): SHIPPED example FAIL\n" + traceback.format_exc()[-2500:])

    # ---- (2) our own minimal kernel (canonical TV-layout tiled copy; add 1.0) ------------------
    P("\nsmoke(2): authoring our own add-one kernel (TV-layout/tiled-copy/fragment) ...")

    @cute.kernel
    def add_one_k(gX: cute.Tensor, gY: cute.Tensor, cC: cute.Tensor, shape: cute.Shape,
                  thr_layout: cute.Layout, val_layout: cute.Layout):
        tidx, _, _ = cute.arch.thread_idx()
        bidx, _, _ = cute.arch.block_idx()
        blk = ((None, None), bidx)
        blkX = gX[blk]; blkY = gY[blk]; blkCrd = cC[blk]
        ld = cute.make_copy_atom(cute.nvgpu.CopyUniversalOp(), gX.element_type)
        st = cute.make_copy_atom(cute.nvgpu.CopyUniversalOp(), gY.element_type)
        tcX = cute.make_tiled_copy_tv(ld, thr_layout, val_layout)
        tcY = cute.make_tiled_copy_tv(st, thr_layout, val_layout)
        thrX = tcX.get_slice(tidx).partition_S(blkX)
        thrY = tcY.get_slice(tidx).partition_S(blkY)
        thrCrd = tcY.get_slice(tidx).partition_S(blkCrd)
        frgX = cute.make_fragment_like(thrX)
        frgY = cute.make_fragment_like(thrY)
        frgP = cute.make_fragment(thrCrd.shape, cutlass.Boolean)
        for i in range(0, cute.size(frgP), 1):
            frgP[i] = cute.elem_less(thrCrd[i], shape)
        cute.copy(ld, thrX, frgX, pred=frgP)
        frgY.store(frgX.load() + 1.0)
        cute.copy(st, frgY, thrY, pred=frgP)

    @cute.jit
    def add_one_host(mX, mY):
        thr_layout = cute.make_ordered_layout((4, 32), order=(1, 0))
        val_layout = cute.make_ordered_layout((4, 4), order=(1, 0))
        tiler_mn, tv_layout = cute.make_layout_tv(thr_layout, val_layout)
        gX = cute.zipped_divide(mX, tiler_mn)
        gY = cute.zipped_divide(mY, tiler_mn)
        idC = cute.make_identity_tensor(mY.shape)
        cC = cute.zipped_divide(idC, tiler=tiler_mn)
        add_one_k(gX, gY, cC, mY.shape, thr_layout, val_layout).launch(
            grid=[cute.size(gY, mode=[1]), 1, 1],
            block=[cute.size(tv_layout, mode=[0]), 1, 1])

    M, N = 1024, 512
    x = torch.randn(M, N, device="cuda", dtype=torch.float32)
    y = torch.zeros_like(x)
    xt = from_dlpack(x).mark_layout_dynamic()
    yt = from_dlpack(y).mark_layout_dynamic()
    try:
        P("smoke(2): compiling ...")
        fn = cute.compile(add_one_host, xt, yt)
        P("smoke(2): compiled OK, launching ...")
        fn(xt, yt)
        torch.cuda.synchronize()
        err = (y - (x + 1.0)).abs().max().item()
        ok = err < 1e-5
        P(f"smoke(2): RESULT correct={ok} maxerr={err:.2e}  {'✅' if ok else '❌'}")
        return f"smoke OK own_correct={ok}"
    except Exception:
        P("smoke(2): EXCEPTION\n" + traceback.format_exc()[-2500:])
        return "smoke FAIL (see log)"


@app.function(gpu="B200", image=EVAL, timeout=900)
def run_gemm_example(m: int = 512, n: int = 512, k: int = 256):
    """M2/M3 de-risk: run the SHIPPED Blackwell tcgen05 GEMM (fp16_gemm_0.py) from /opt/cutlass on our
    B200 image. If it PASSES, the ENTIRE heavy path — TMA producer ring + tcgen05 UMMA + TMEM
    accumulator + PipelineTmaUmma/PipelineUmmaAsync + epilogue — is confirmed working on our setup,
    end-to-end, with no nvcc-for-the-kernel. This is the template we mutate for the tf32x3 QR apply."""
    import sys, traceback, importlib.util
    P = lambda *a: print(*a, file=sys.stdout, flush=True)
    path = "/opt/cutlass/examples/python/CuTeDSL/cute/blackwell/tutorial/tutorial_gemm/fp16_gemm_0.py"
    P(f"running shipped Blackwell tcgen05 GEMM  mnk=({m},{n},{k})  {path}")
    try:
        spec = importlib.util.spec_from_file_location("_g0", path)
        mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
        mod.run_dense_gemm((m, n, k), tolerance=1e-1)
        P("GEMM example PASS ✅  (tcgen05 + TMA + TMEM + pipeline all work on our image)")
        return "gemm OK"
    except Exception:
        P("GEMM example FAIL\n" + traceback.format_exc()[-3500:])
        return "gemm FAIL"


@app.local_entrypoint()
def main():
    print(info.remote())
    print("\n\n########## KERNEL SMOKE ##########")
    print(kernel_smoke.remote())
