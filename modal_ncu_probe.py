"""modal_ncu_probe.py — diagnose whether Nsight Compute (ncu) can profile on Modal B200.

The classic blocker on managed cloud GPUs is ERR_NVGPUCTRPERM: the container lacks
permission to read GPU performance counters (host driver has
NVreg_RestrictProfilingToAdminUsers=1 AND the container lacks CAP_SYS_ADMIN).

This probe gathers ALL diagnostics in ONE B200 container so we know exactly what
the situation is before trying fixes:
  - is ncu present / installable?
  - are we root? what capabilities does the container have?
  - what does the kernel profiling-permission knob say?
  - does a minimal `ncu` run actually work, and if not, what's the exact error?

Run:  modal run modal_ncu_probe.py
"""
import modal

# CUDA devel image ships the full toolkit incl. nsight-compute (ncu).
image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.8.1-devel-ubuntu24.04", add_python="3.12"
    )
    .pip_install("numpy")
    .pip_install("torch", index_url="https://download.pytorch.org/whl/cu128")
)
app = modal.App("qr-ncu-probe")


@app.function(gpu="B200", image=image, timeout=900)
def probe():
    import subprocess, os, shutil, textwrap

    LOG = []  # accumulate everything so the RETURN value carries it (robust to log-stream drops)

    def out(s):
        print(s, flush=True)
        LOG.append(str(s))

    def sh(cmd, **kw):
        out(f"\n$ {cmd}")
        try:
            p = subprocess.run(cmd, shell=True, capture_output=True, text=True,
                               timeout=300, **kw)
            o = (p.stdout or "") + (p.stderr or "")
            out(o[:6000])
            return o
        except Exception as e:
            out(f"[exc] {e}")
            return str(e)

    out("=" * 70)
    out("SECTION 1: identity & capabilities")
    out("=" * 70)
    sh("id")
    sh("whoami")
    sh("cat /proc/self/status | grep -iE 'Cap|Uid|Gid'")
    # decode the effective capability set
    sh("capsh --decode=$(grep CapEff /proc/self/status | awk '{print $2}') 2>/dev/null || echo 'capsh not present'")

    out("=" * 70)
    out("SECTION 2: GPU profiling-permission knobs (host driver state)")
    out("=" * 70)
    sh("cat /proc/driver/nvidia/params 2>/dev/null | grep -i prof || echo 'no nvidia params file'")
    sh("nvidia-smi")
    sh("nvidia-smi --query-gpu=name,driver_version --format=csv")

    out("=" * 70)
    out("SECTION 3: ncu availability")
    out("=" * 70)
    sh("which ncu || echo 'ncu not on PATH'")
    sh("ls -la /usr/local/cuda/bin/ncu* 2>/dev/null || echo 'no ncu in /usr/local/cuda/bin'")
    sh("find / -name 'ncu' -type f 2>/dev/null | head")
    ncu = shutil.which("ncu") or "/usr/local/cuda/bin/ncu"
    sh(f"{ncu} --version 2>&1 | head -5 || echo 'ncu version failed'")

    out("=" * 70)
    out("SECTION 4: minimal CUDA workload (tiny torch matmul)")
    out("=" * 70)
    work = textwrap.dedent("""
        import torch
        a = torch.randn(512, 512, device='cuda')
        b = torch.randn(512, 512, device='cuda')
        for _ in range(3):
            c = a @ b
        torch.cuda.synchronize()
        print('matmul done', c.sum().item())
    """)
    with open("/tmp/work.py", "w") as f:
        f.write(work)
    sh("python /tmp/work.py")

    out("=" * 70)
    out("SECTION 5: ncu attempts (escalating)")
    out("=" * 70)
    # 5a: just launch stats (does NOT need HW perf counters) — should work even w/o perms
    sh(f"{ncu} --set launchstats --launch-count 1 --target-processes all "
       f"python /tmp/work.py 2>&1 | head -60")
    # 5b: full default set (NEEDS HW perf counters — the ERR_NVGPUCTRPERM gate)
    sh(f"{ncu} --launch-count 1 --target-processes all "
       f"python /tmp/work.py 2>&1 | head -80")
    # 5c: explicitly request a counter-based metric set
    sh(f"{ncu} --set basic --launch-count 1 --target-processes all "
       f"python /tmp/work.py 2>&1 | head -80")

    out("=" * 70)
    out("DONE")
    out("=" * 70)
    return "\n".join(LOG)


@app.local_entrypoint()
def main():
    print(probe.remote())
