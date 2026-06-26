import os, triton
f = os.path.join(os.path.dirname(triton.__file__), "tools/triton_to_gluon_translater/translator_helpers.py")
lines = open(f).read().splitlines()
print(f"TOTAL {len(lines)} lines\n")
print("\n".join(f"{i:4d}: {l}" for i, l in enumerate(lines[:165])))
