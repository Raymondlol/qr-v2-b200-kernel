"""Deployability recon: what LOW-LEVEL control is available in the STOCK triton 3.6.0
we'd actually deploy with (= the modal_app env, ~= the competition eval env)? TLX is a
Meta fork (not in stock triton); the in-tree equivalent is Gluon (triton's own
warp-spec / tcgen05 layer). If Gluon is here, it's a DEPLOYABLE near-podium route; if
not, low-level warp-spec is as undeployable for us as raw-PTX was (nvcc). No banned subs.
"""
import importlib, torch, triton
print("torch", torch.__version__, "| triton", triton.__version__)
print("dev", torch.cuda.get_device_name(0), "cc", torch.cuda.get_device_capability(0))

def probe(modname):
    try:
        m = importlib.import_module(modname)
        attrs = [a for a in dir(m) if not a.startswith('__')]
        print(f"  OK  {modname:42s} -> {attrs[:18]}")
        return m
    except Exception as e:
        print(f"  --  {modname:42s} : {type(e).__name__}: {str(e)[:60]}")
        return None

print("\n=== GLUON (in-tree triton low-level layer) ===")
for mod in ['triton.experimental.gluon', 'triton.experimental', 'triton.gluon',
            'triton.language.extra', 'triton.experimental.gluon.language',
            'triton.experimental.gluon.language.nvidia',
            'triton.experimental.gluon.language.nvidia.blackwell',
            'triton.experimental.gluon.language.nvidia.hopper']:
    probe(mod)

print("\n=== TLX (Meta fork — expect absent in stock) ===")
for mod in ['triton_tlx', 'tlx', 'triton.tlx', 'triton.tools.experimental_descriptor']:
    probe(mod)

print("\n=== triton.language low-level primitives present? ===")
import triton.language as tl
for f in ['async_task', 'warp_specialize', 'make_tensor_descriptor', 'dot_scaled',
          'inline_asm_elementwise', 'associative_scan']:
    print(f"  tl.{f:26s} {'YES' if hasattr(tl, f) else 'no'}")

# If gluon exists, probe its Blackwell tcgen05 / warp-spec API surface
print("\n=== gluon Blackwell tcgen05 / warp-spec API (if present) ===")
g = None
for mod in ['triton.experimental.gluon.language.nvidia.blackwell',
            'triton.experimental.gluon']:
    try:
        g = importlib.import_module(mod)
        ks = [a for a in dir(g) if any(s in a.lower() for s in
              ('mma','tmem','tcgen','tensor_memory','mbarrier','warp','async','tma','alloc'))]
        print(f"  {mod}: {ks}")
    except Exception as e:
        print(f"  {mod}: {type(e).__name__}")

print("\nDONE")
