#!/bin/zsh
# Parallel sweep driver for fp8_ozaki_sim.py on batch=640 (worst-of-640 gate).
# Each job thread-limited; bounded parallelism to avoid thrash on 12c/18GB.
PY=python
SIM="$(dirname $0)/fp8_ozaki_sim.py"
OUT="$(dirname $0)/sweep_out.txt"
: > "$OUT"
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
B=${1:-640}

run() { $PY -u "$SIM" --batch $B "$@" >> "$OUT" 2>&1; }

# Calibration anchors
run --mode calib &
run --mode tf32x3 &
run --mode tf32 &
wait

# e4m3 (fp8) per_tensor ladder
run --mode ozaki --fmt e4m3 --kV 2 --kCY 2 --T 1 &
run --mode ozaki --fmt e4m3 --kV 3 --kCY 3 --T 2 &
run --mode ozaki --fmt e4m3 --kV 4 --kCY 4 --T 3 &
wait
# e4m3 per_vec block scale + asymmetric (more on C/Y, the dynamic-range operand)
run --mode ozaki --fmt e4m3 --kV 2 --kCY 2 --T 1 --scaleV per_vec --scaleCY per_vec &
run --mode ozaki --fmt e4m3 --kV 3 --kCY 3 --T 2 --scaleV per_vec --scaleCY per_vec &
run --mode ozaki --fmt e4m3 --kV 2 --kCY 3 --T 2 &
wait
# trailing-only (Gram & W left fp32) to isolate where slices are needed
run --mode ozaki --fmt e4m3 --kV 2 --kCY 2 --T 1 --gram 0 --W 0 &
run --mode ozaki --fmt e4m3 --kV 3 --kCY 3 --T 2 --gram 0 --W 0 &
wait

# e2m1 (fp4) per_tensor ladder
run --mode ozaki --fmt e2m1 --kV 4 --kCY 4 --T 3 &
run --mode ozaki --fmt e2m1 --kV 5 --kCY 5 --T 4 &
run --mode ozaki --fmt e2m1 --kV 6 --kCY 6 --T 5 &
wait
run --mode ozaki --fmt e2m1 --kV 3 --kCY 3 --T 2 &
run --mode ozaki --fmt e2m1 --kV 4 --kCY 4 --T 3 --scaleV per_vec --scaleCY per_vec &
run --mode ozaki --fmt e2m1 --kV 5 --kCY 5 --T 4 --scaleV per_vec --scaleCY per_vec &
wait

# e5m2 reference (2 mantissa bits)
run --mode ozaki --fmt e5m2 --kV 3 --kCY 3 --T 2 &
run --mode ozaki --fmt e5m2 --kV 4 --kCY 4 --T 3 &
wait

echo "SWEEP DONE" >> "$OUT"
