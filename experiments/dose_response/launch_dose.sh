#!/bin/bash
# Launch the dose-response run across the 4 GPUs, splitting the 48 dates round-robin.
#   launch_dose.sh <model> <patch_var>
set -eo pipefail
cd ~/weather-interpretability
source ~/venv/bin/activate

MODEL=${1:-pangu}
PATCH=${2:-Z1000}
NG=4

# same 48 inits as the 19-variable patching experiment: days 4/11/18/25, hours cycling 00/06/12/18
DATES=$(python3 -c "
tri=[(2024,m,d) for m in range(1,13) for d in (4,11,18,25)]
hrs=[0,6,12,18]*(len(tri)//4)
print(','.join(f'{y}{m:02d}{d:02d}_{h:02d}' for (y,m,d),h in zip(tri,hrs)))")

IFS=',' read -ra A <<< "$DATES"
echo "${#A[@]} дат, $NG воркеров, поле $PATCH, модель $MODEL"

for g in $(seq 0 $((NG-1))); do
  sub=$(python3 -c "
import sys
a='''$DATES'''.split(',')
print(','.join(a[$g::$NG]))")
  log=/tmp/dose_${MODEL}_${PATCH}_w$g.log
  CUDA_VISIBLE_DEVICES=$g nohup python3 -u experiments/dose_response/run_dose_response.py \
      "$MODEL" "w$g" "$sub" "$PATCH" > "$log" 2>&1 &
  echo "  GPU $g: $(echo $sub | tr ',' '\n' | wc -l) дат -> $log"
done
wait
echo "=== все воркеры завершились ==="
