#!/usr/bin/env bash
# Run the four downstream subset evaluations and the TerraTorch loading checks one after another.
# Every evaluator resumes from its per-sample file, so each step is simply repeated a few times: a repeat after
# the sample is finished exits quickly. Run detached, e.g.
#   setsid nohup bash scripts/Surya/run_downstream_queue.sh > /home/user/work/logs/queue.log 2>&1 &
set -u
cd "$(dirname "$0")/../.."
PY=/home/user/envs/surya/bin/python
TT=/home/user/envs/terratorch/bin/python
D=scripts/Surya/downstream
LOGS=/home/user/work/logs
FT=/home/user/work/surya_ft
mkdir -p "$LOGS"

stamp() { date -u +%H:%M:%S; }

# wait for an evaluation started by hand to finish (only one model fits in memory)
while pgrep -f "_subset_eval.py --split" > /dev/null; do sleep 60; done

step() {  # step <name> <repeats> <command...>
  local name=$1 reps=$2; shift 2
  for i in $(seq 1 "$reps"); do
    echo "$(stamp) start $name (run $i)"
    "$@" >> "$LOGS/$name.log" 2>&1
    echo "$(stamp) end $name (run $i) exit $?"
  done
}

tt_check() {  # tt_check <task> <registry name> <checkpoint>
  if [ -f "$3" ]; then
    step "terratorch_$1" 1 $TT scripts/Surya/terratorch_check.py --name "$2" --ckpt "$3" \
      --out "results/Surya/terratorch/$1.json"
  else
    echo "$(stamp) skip terratorch_$1: no checkpoint at $3"
  fi
}

step flare_test 2 $PY $D/flare_forecasting_subset_eval.py --split test --n 40 --seed 0
step flare_validation 2 $PY $D/flare_forecasting_subset_eval.py --split validation --n 20 --seed 0
step flare_summary 1 $PY $D/flare_forecasting_subset_eval.py --summarize-only
tt_check flare heliofm_backbone_surya_solar_flare_forecasting $FT/flare_forecasting/solar_flare_weights.pth

step ar_test 2 $PY $D/ar_segmentation_subset_eval.py --split test --n 30 --seed 0
step ar_validation 2 $PY $D/ar_segmentation_subset_eval.py --split validation --n 10 --seed 0
step ar_summary 1 $PY $D/ar_segmentation_subset_eval.py --summarize-only
tt_check ar_segmentation heliofm_backbone_surya_ar_segmentation $FT/ar_segmentation/ar_segmentation_weights.pth

step wind_validation 2 $PY $D/solar_wind_subset_eval.py --split validation --n 10 --seed 0
step wind_test 3 $PY $D/solar_wind_subset_eval.py --split test --n 30 --seed 0 --include-reserves
step wind_summary 1 $PY $D/solar_wind_subset_eval.py --summarize-only
tt_check solar_wind heliofm_backbone_surya_solar_wind_forecasting $FT/solar_wind/solar_wind_weights.pth

step euv_test 2 $PY $D/euv_spectra_subset_eval.py --split test --n 30 --seed 0
step euv_validation 2 $PY $D/euv_spectra_subset_eval.py --split validation --n 20 --seed 0
step euv_summary 1 $PY $D/euv_spectra_subset_eval.py --summarize-only
tt_check euv_spectra heliofm_backbone_surya_euv_spectra_prediction $FT/euv_spectra/euv_spectra_weights.pth

echo "$(stamp) QUEUE DONE"
