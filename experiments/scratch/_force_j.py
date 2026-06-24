import sys, importlib.util, torch
import reference
sys.path.insert(0,'.')
spec = importlib.util.spec_from_file_location("cand", "../cand_J_2level.py")
m = importlib.util.module_from_spec(spec); import task; sys.modules['task']=task
spec.loader.exec_module(m)
cases = [
  {"batch":3,"n":352,"cond":1,"seed":1200},
  {"batch":3,"n":512,"cond":2,"seed":32523},
  {"batch":3,"n":512,"cond":4,"seed":32524,"case":"dense"},
  {"batch":3,"n":512,"cond":0,"seed":32525,"case":"rankdef"},
  {"batch":3,"n":512,"cond":0,"seed":32526,"case":"clustered"},
  {"batch":3,"n":512,"cond":2,"seed":32530,"case":"mixed"},
  {"batch":2,"n":640,"cond":2,"seed":7,"case":"mixed"},   # 640 -> NB=128*5
  {"batch":2,"n":300,"cond":2,"seed":9,"case":"mixed"},   # NB=128, last super-panel partial
  {"batch":3,"n":176,"cond":1,"seed":3321},               # small path
]
nf=0
for tc in cases:
    data = reference.generate_input(**tc)
    out = m._factor_custom(data.clone())
    good,msg = reference.check_implementation(data,out)
    print(("PASS" if good else "FAIL"), f"n={tc['n']:4d} {tc.get('case','dense'):12s}", msg[:78])
    nf += (not good)
print("FAILS",nf)
