"""Find the cute-DSL 4.5.2 API for in-register bitcast + bitwise-and (for the tf32x3 limb split).
`cute.bitcast` does NOT exist. Hunt the real names. Plain-Python introspection (no kernel)."""
import cutlass
import cutlass.cute as cute

KW = ['cast', 'bit', 'reinterpret', 'tf32', 'tfloat', 'recast', 'and_', 'view']


def hits(mod):
    return sorted([a for a in dir(mod) if any(k in a.lower() for k in KW)])


print("=== module attribute hunt ===")
print("cutlass.cute      :", hits(cute))
print("cutlass           :", hits(cutlass))
try:
    import cutlass.cute.arch as arch
    print("cutlass.cute.arch :", hits(arch))
except Exception as e:
    print("arch import:", e)
try:
    import cutlass.cute.typing as typ
    print("cutlass.cute.typing:", hits(typ))
except Exception as e:
    print("typing import:", e)

print("\n=== numeric type surfaces ===")
print("has cutlass.TFloat32:", hasattr(cutlass, 'TFloat32'))
for tn in ['Float32', 'Int32', 'TFloat32']:
    t = getattr(cutlass, tn, None)
    if t is not None:
        meth = [a for a in dir(t) if not a.startswith('__')]
        print(f"  cutlass.{tn}:", meth[:50])

print("\n=== Numeric base / scalar ops ===")
for modname in ['cutlass.cute.core', 'cutlass._mlir', 'cutlass.base_dsl']:
    try:
        import importlib
        m = importlib.import_module(modname)
        print(f"  {modname}:", hits(m))
    except Exception as e:
        print(f"  {modname}: {type(e).__name__}")
print("DONE")
