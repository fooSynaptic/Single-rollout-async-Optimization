#!/usr/bin/env bash
# Run MATH experiment settings serially:
# SAO, GRPO, GRPO+DIS (G=8), GRPO+DIS (G=1), running-mean.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
STOP_ON_FAIL="${STOP_ON_FAIL:-1}"
START_MONITOR="${START_MONITOR:-1}"
LOG_DIR="${LOG_DIR:-$ROOT/logs}"
mkdir -p "$LOG_DIR"

EXPERIMENTS=(
  phase2_hard_sao
  phase2_hard_grpo
  phase2_hard_grpo_dis
  phase2_hard_grpo_dis_g1
  phase2_hard_running_mean
)

summary="$LOG_DIR/math_hard_compare_summary_$(date +%Y%m%d_%H%M%S).txt"
echo "MATH-hard comparison started $(date -u +%Y-%m-%dT%H:%M:%SZ)" | tee "$summary"

if [[ "$START_MONITOR" == "1" ]]; then
  LOG_DIR="$LOG_DIR" "$ROOT/scripts/start_monitor.sh"
fi

for exp in "${EXPERIMENTS[@]}"; do
  config="$ROOT/configs/$exp.yaml"
  echo "===== $exp =====" | tee -a "$summary"
  if "$ROOT/scripts/launch_experiment.sh" "$config"; then
    echo "OK $exp $(date -u +%Y-%m-%dT%H:%M:%SZ)" | tee -a "$summary"
  else
    rc=$?
    echo "FAIL $exp rc=$rc $(date -u +%Y-%m-%dT%H:%M:%SZ)" | tee -a "$summary"
    if [[ "$STOP_ON_FAIL" == "1" ]]; then
      exit "$rc"
    fi
  fi
done

echo "MATH-hard comparison finished $(date -u +%Y-%m-%dT%H:%M:%SZ)" | tee -a "$summary"
echo "summary=$summary"
