"""modal_ncu_probe2b.py — CLEAN CUDA-13 ncu/nsys probe (no ubuntu nsight-systems junk).

probe2 apt-installed ubuntu's ancient nsight-systems (2022.4) which dragged in
libnvidia-compute-535 (may shadow the host driver's libcuda). This version uses ONLY
the CUDA-13 devel image's bundled ncu + the NVIDIA-repo nsys (cuda-nsight-systems-13-0),
both matched to driver 580. Answers: does a driver-matched ncu fix LibraryNotLoaded?
"""
import modal

image = (
    modal.Image.from_registry("nvidia/cuda:13.0.1-devel-ubuntu24.04", add_python="3.12")
    .apt_install("cuda-nsight-systems-13-0")  # NVIDIA-repo nsys matched to CUDA 13 (repo already in image)
    .pip_install("numpy")
    .pip_install("torch", index_url="https://download.pytorch.org/whl/cu128")
)
app = modal.App("qr-ncu-probe2b")


@app.function(gpu="B200", image=image, timeout=900)
def probe():
    import subprocess, textwrap, glob

    LOG = []

    def out(s):
        print(s, flush=True)
        LOG.append(str(s))

    def sh(cmd, timeout=300):
        out(f"\n$ {cmd}")
        try:
            p = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=timeout)
            o = (p.stdout or "") + (p.stderr or "")
            out(o[:8000])
            return o
        except Exception as e:
            out(f"[exc] {e}")
            return str(e)

    # locate the image-bundled tools (avoid PATH ambiguity)
    ncu_cands = glob.glob("/opt/nvidia/nsight-compute/*/ncu") + ["/usr/local/cuda/bin/ncu"]
    nsys_cands = glob.glob("/opt/nvidia/nsight-systems/*/target-linux-x64/nsys") + \
                 glob.glob("/opt/nvidia/nsight-systems-cli/*/target-linux-x64/nsys") + \
                 ["/usr/local/cuda/bin/nsys"]
    ncu = next((c for c in ncu_cands if __import__("os").path.exists(c)), "ncu")
    nsys = next((c for c in nsys_cands if __import__("os").path.exists(c)), "nsys")

    out("=" * 70 + "\nSECTION A: versions + sandbox + device nodes\n" + "=" * 70)
    sh("uname -a")
    sh("ls -la /dev/nvidia* 2>/dev/null; echo '---caps---'; ls -la /dev/nvidia-caps* 2>/dev/null")
    sh("ls /proc/driver/nvidia/capabilities/ 2>/dev/null || echo 'no capabilities dir'")
    sh("find /opt/nvidia -maxdepth 3 -name ncu -o -name nsys 2>/dev/null")
    sh(f"{ncu} --version")
    sh(f"{nsys} --version")

    out("\n" + "=" * 70 + "\nSECTION B: toy workload (verify torch/cuda not shadowed)\n" + "=" * 70)
    work = textwrap.dedent("""
        import torch
        print('torch', torch.__version__, torch.version.cuda, torch.cuda.get_device_name(0))
        a = torch.randn(1024,1024,device='cuda'); b = torch.randn(1024,1024,device='cuda')
        torch.cuda.profiler.start()
        for _ in range(3): c = a@b
        torch.cuda.synchronize(); torch.cuda.profiler.stop()
        print('matmul', float(c.sum()))
    """)
    with open("/tmp/work.py", "w") as f:
        f.write(work)
    sh("python /tmp/work.py")

    out("\n" + "=" * 70 + "\nSECTION C: ncu (CUDA-13 matched) — the A-vs-B test\n" + "=" * 70)
    sh(f"{ncu} --set basic --launch-count 1 --target-processes all python /tmp/work.py 2>&1 | head -100")

    out("\n" + "=" * 70 + "\nSECTION D: nsys timeline (fallback)\n" + "=" * 70)
    sh(f"{nsys} profile -t cuda --force-overwrite=true -o /tmp/toy "
       f"--capture-range=cudaProfilerApi --capture-range-end=stop python /tmp/work.py 2>&1 | head -40")
    sh(f"{nsys} stats --report cuda_gpu_kern_sum --format table /tmp/toy.nsys-rep 2>&1 | head -40")

    out("\n" + "=" * 70 + "\nDONE\n" + "=" * 70)
    return "\n".join(LOG)


@app.local_entrypoint()
def main():
    print(probe.remote())
