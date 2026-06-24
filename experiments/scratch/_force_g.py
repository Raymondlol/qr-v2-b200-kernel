import sys, importlib.util, torch
import reference
sys.path.insert(0,'.')
spec = importlib.util.spec_from_file_location("cand", "../cand_G_largeN.py")
m = importlib.util.module_from_spec(spec); import task; sys.modules['task']=task
spec.loader.exec_module(m)
cases = [
  {"batch":2,"n":256,"cond":2,"seed":224468,"case":"mixed"},
  {"batch":2,"n":512,"cond":0,"seed":224467,"case":"rankdef"},
  {"batch":2,"n":512,"cond":2,"seed":99,"case":"mixed"},
  {"batch":2,"n":256,"cond":0,"seed":4330,"case":"nearrank"},
]
nf=0
for tc in cases:
    data = reference.generate_input(**tc)
    out = m._factor_geqrf_blocked(data.clone())
    good,msg = reference.check_implementation(data,out)
    print(("PASS" if good else "FAIL"), f"n={tc['n']} {tc.get('case')}", msg[:90])
    nf += (not good)
print("FAILS",nf)
