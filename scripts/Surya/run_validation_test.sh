#!/usr/bin/env bash
# Score the Surya-1.0 validation test group by group. Each precision mode runs in its own process so that
# memory is released between them; a group's files are kept for the first mode and deleted after the second.
set -euo pipefail
PY=${PY:-/home/user/envs/surya/bin/python}
UP=${UP:-/home/user/upstream/surya/Surya}
WD=${WD:-/home/user/work/surya_run}
OUT=${OUT:-results/Surya/validation_test}
HERE=$(cd "$(dirname "$0")" && pwd)
for g in ${GROUPS_TO_RUN:-3 4 0 1 2}; do
  "$PY" "$HERE/run_validation_test_grouped.py" --upstream "$UP" --workdir "$WD" --out "$OUT" --modes cpu_bf16 --groups "$g" --keep-files
  "$PY" "$HERE/run_validation_test_grouped.py" --upstream "$UP" --workdir "$WD" --out "$OUT" --modes cuda_autocast_noop --groups "$g"
done
