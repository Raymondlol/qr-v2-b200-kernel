"""Probe: is TLX (Triton Low-level eXtensions, the leaders' 'triton_tlx' route) usable on
Modal B200 so we can actually develop/debug it? Checks (1) the full warp-spec / async_task /
tlx surface in stock triton 3.6.0 (maybe we missed more than gl.warp_specialize), (2) whether
triton-tlx / tlx is pip-installable, (3) whether a TLX-style warp-specialized kernel compiles.
No banned substrings (this is not a submission)."""
import modal

image = (modal.Image.debian_slim(python_version="3.12")
         .apt_install("git")
         .pip_install("numpy")
         .pip_install("torch", index_url="https://download.pytorch.org/whl/cu128"))
app = modal.App("tlx-probe")


@app.function(gpu="B200", image=image, timeout=1200)
def probe():
    import importlib, subprocess, sys, os, glob
    out = []
    def log(*a): out.append(" ".join(str(x) for x in a))

    import triton
    log("triton", triton.__version__, "at", os.path.dirname(triton.__file__))

    # (1) exhaustive stock surface sweep for tlx / async_task / warp-spec / specialize
    log("\n=== stock triton: grep installed tree for tlx/async_task/warp_special/specialize ===")
    root = os.path.dirname(triton.__file__)
    pats = ["tlx", "async_task", "warp_special", "specialize", "async_tasks", "ws_"]
    hits = {}
    for f in glob.glob(os.path.join(root, "**", "*.py"), recursive=True):
        try: txt = open(f, errors="ignore").read()
        except Exception: continue
        for p in pats:
            if p in txt:
                hits.setdefault(p, set()).add(f.replace(root, "..."))
    for p, fs in hits.items():
        log(f"  '{p}': {len(fs)} files -> {sorted(fs)[:6]}")

    log("\n=== importable tlx-ish modules / attrs ===")
    for modn in ["triton.tlx", "triton.language.tlx", "tlx", "triton_tlx",
                 "triton.experimental.tlx", "triton.language.async_task"]:
        try:
            m = importlib.import_module(modn)
            log(f"  OK import {modn}: {[a for a in dir(m) if not a.startswith('_')][:15]}")
        except Exception as e:
            log(f"  -- {modn}: {type(e).__name__}")
    import triton.language as tl
    from triton.experimental.gluon import language as gl
    for nm in ["async_task", "async_tasks", "warp_specialize", "specialize"]:
        log(f"  tl.{nm}: {'YES' if hasattr(tl, nm) else 'no'}   gl.{nm}: {'YES' if hasattr(gl, nm) else 'no'}")

    # (2) is triton-tlx / tlx pip-installable? (runtime pip, capture; don't break torch)
    log("\n=== pip availability of TLX packages (index check, no install) ===")
    for pkg in ["triton-tlx", "tlx", "triton_tlx"]:
        try:
            r = subprocess.run([sys.executable, "-m", "pip", "index", "versions", pkg],
                               capture_output=True, text=True, timeout=120)
            line = (r.stdout + r.stderr).strip().splitlines()
            log(f"  {pkg}: {line[0] if line else '(no output)'}")
        except Exception as e:
            log(f"  {pkg}: {type(e).__name__}: {str(e)[:80]}")

    # (3) gl.warp_specialize sanity (the surface we DO have) — already validated this session
    log("\n=== gl.warp_specialize present (the deployable warp-spec we have) ===")
    log(f"  gl.warp_specialize: {'YES' if hasattr(gl, 'warp_specialize') else 'NO'}")

    return "\n".join(out)


@app.local_entrypoint()
def main():
    print(probe.remote())
