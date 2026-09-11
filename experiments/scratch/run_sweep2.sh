#!/bin/zsh
PY=python
SIM="$(dirname $0)/fp8_ozaki_sim.py"
OUT="$(dirname $0)/sweep_out2.txt"
: > "$OUT"
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
B=640
run() { $PY -u "$SIM" --batch $B "$@" >> "$OUT" 2>&1; }

# Asymmetric fp8: fewer slices on V (unit reflectors), more on C/Y (dynamic range)?
run --mode ozaki --fmt e4m3 --kV 2 --kCY 3 --T 3 &              # 5 dots, full T
run --mode ozaki --fmt e4m3 --kV 3 --kCY 2 --T 3 &             # 5 dots
run --mode ozaki --fmt e4m3 --kV 2 --kCY 4 --T 3 &            # 7 dots
wait
# Reseed robustness of the fp8 3-slice boundary (competition reseeds!)
run --mode ozaki --fmt e4m3 --kV 3 --kCY 3 --T 2 --seed 99001 &
run --mode ozaki --fmt e4m3 --kV 3 --kCY 3 --T 2 --seed 12345 &
# Reseed fp4 5-slice per_vec boundary
run --mode ozaki --fmt e2m1 --kV 5 --kCY 5 --T 4 --scaleV per_vec --scaleCY per_vec --seed 99001 &
wait
# fp8 3-slice trailing-only vs all-three at the reseed (does gram/W need slices?)
run --mode ozaki --fmt e4m3 --kV 3 --kCY 3 --T 2 --gram 0 --W 0 --seed 99001 &
# fp4 6-slice per_tensor reseed
run --mode ozaki --fmt e2m1 --kV 6 --kCY 6 --T 5 --seed 99001 &
wait
echo "SWEEP2 DONE" >> "$OUT"
