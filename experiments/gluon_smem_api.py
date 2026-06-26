"""Stage A.1 recon: exact Gluon shared_memory_descriptor API for streaming a [BN,BCOLS]
panel tile in ROW-BLOCKS (load a block to regs, update, store back). Dump signatures + source
of .load/.store/.slice/.index + the shared-memory layouts, and a TINY end-to-end smem round-trip
(global->smem->block-load->modify->block-store->global) to confirm the idioms compile+run.
No banned substrings."""
import inspect, importlib, torch, triton
from triton.experimental import gluon
from triton.experimental.gluon import language as gl
from triton.tools.triton_to_gluon_translater.translator_helpers import default_blocked_layout
print("triton", triton.__version__)

core = importlib.import_module("triton.experimental.gluon.language._core")
smd = getattr(core, "shared_memory_descriptor", None)
print("\n=== shared_memory_descriptor methods ===")
if smd:
    for nm in dir(smd):
        if not nm.startswith("_") and callable(getattr(smd, nm, None)):
            try: print(f"  smd.{nm}{inspect.signature(getattr(smd,nm))}")
            except Exception: print(f"  smd.{nm} (sig N/A)")
    for nm in ["slice", "index", "load", "store"]:
        o = getattr(smd, nm, None)
        if o:
            try:
                print(f"\n--- smd.{nm} source ---\n{inspect.getsource(o)[:1400]}")
            except Exception as e:
                print(f"  smd.{nm} src N/A: {e}")

print("\n=== shared layouts ===")
for nm in ["SwizzledSharedLayout", "NVMMASharedLayout"]:
    o = getattr(gl, nm, None)
    if o:
        try: print(f"  gl.{nm}{inspect.signature(o)}")
        except Exception: print(f"  gl.{nm} present")


# --- tiny end-to-end: global -> smem -> per-row-block load/modify/store -> global ---
@gluon.jit
def roundtrip(X, Y, BN: gl.constexpr, BC: gl.constexpr, RBLK: gl.constexpr, MODE: gl.constexpr):
    blk: gl.constexpr = default_blocked_layout([BN, BC], gl.num_warps())
    rl: gl.constexpr = gl.SliceLayout(1, blk)
    cl: gl.constexpr = gl.SliceLayout(0, blk)
    r = gl.arange(0, BN, layout=rl)[:, None]
    c = gl.arange(0, BC, layout=cl)[None, :]
    t = gl.load(X + r * BC + c)
    smem = gl.allocate_shared_memory(gl.float32, [BN, BC], gl.SwizzledSharedLayout(1, 1, 1, [1, 0]))
    smem.store(t)
    if MODE == 0:
        # whole-tile load back, add 1
        u = smem.load(blk) + 1.0
        gl.store(Y + r * BC + c, u)
    else:
        # per-row-block: slice rows [rb:rb+RBLK], load block to regs, *2, store back to smem.
        # slice() needs a COMPILE-TIME start -> static (unrolled) loop so rb is a constexpr int.
        blkr: gl.constexpr = default_blocked_layout([RBLK, BC], gl.num_warps())
        for i in gl.static_range(0, BN // RBLK):
            sub = smem.slice(i * RBLK, RBLK)      # slice first dim -> [RBLK, BC] descriptor
            tile_rb = sub.load(blkr)
            sub.store(tile_rb * 2.0)
        u = smem.load(blk)
        gl.store(Y + r * BC + c, u)


def run(MODE, BN=128, BC=64, RBLK=32):
    X = torch.randn(BN, BC, device="cuda")
    Y = torch.empty(BN, BC, device="cuda")
    roundtrip[(1,)](X, Y, BN=BN, BC=BC, RBLK=RBLK, MODE=MODE, num_warps=4)
    return X, Y


def main():
    print("\n=== end-to-end smem roundtrip ===")
    try:
        X, Y = run(0)
        torch.cuda.synchronize()
        err = (Y - (X + 1.0)).abs().max().item()
        print(f"  MODE0 whole-tile smem roundtrip: maxerr={err:.2e} {'OK' if err < 1e-5 else 'BAD'}")
    except Exception as e:
        import traceback; print("  MODE0 FAIL"); traceback.print_exc()
    try:
        X, Y = run(1)
        torch.cuda.synchronize()
        err = (Y - (X * 2.0)).abs().max().item()
        print(f"  MODE1 row-block slice roundtrip: maxerr={err:.2e} {'OK' if err < 1e-5 else 'BAD'}")
    except Exception as e:
        import traceback; print("  MODE1 (slice) FAIL"); traceback.print_exc()
    print("\nSMEM API RECON DONE")


if __name__ == "__main__":
    main()
