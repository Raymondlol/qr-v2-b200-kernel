import sys, importlib.util, torch
import reference
sys.path.insert(0,'.')
spec = importlib.util.spec_from_file_location("cand", "../cand_D_hybrid.py")
m = importlib.util.module_from_spec(spec); import task; sys.modules['task']=task
spec.loader.exec_module(m)
# Force custom path (3xTF32) regardless of shape, to validate accuracy on CPU.
cases = [
  {"batch":4,"n":512,"cond":2,"seed":32523},
  {"batch":4,"n":512,"cond":4,"seed":32524,"case":"dense"},
  {"batch":4,"n":512,"cond":0,"seed":32525,"case":"rankdef"},
  {"batch":4,"n":512,"cond":0,"seed":32526,"case":"clustered"},
  {"batch":4,"n":512,"cond":2,"seed":32530,"case":"mixed"},
  {"batch":4,"n":256,"cond":2,"seed":4332,"case":"mixed"},
  {"batch":4,"n":1024,"cond":2,"seed":4332,"case":"mixed"},
]
nfail=0
for tc in cases:
    data = reference.generate_input(**tc)
    out = m._factor_custom(data.clone())
    good,msg = reference.check_implementation(data,out)
    tag = "PASS" if good else "FAIL"
    if not good: nfail+=1
    print(f"[{tag}] n={tc['n']} {tc.get('case','dense')}: {msg[:110]}")
print("FAILS:",nfail)
