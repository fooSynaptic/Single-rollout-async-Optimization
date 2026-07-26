#!/usr/bin/env bash
# Launch one experiment config on an existing Ray cluster, or create a single-node cluster.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONFIG="${1:-${CONFIG:-}}"
AREAL_ROOT="${AREAL_ROOT:-$ROOT/vendor/AReaL}"
if [[ -z "${PYTHON:-}" ]]; then
  if [[ -x "$AREAL_ROOT/.venv/bin/python" ]]; then
    PYTHON="$AREAL_ROOT/.venv/bin/python"
  else
    PYTHON="$(command -v python3 || true)"
  fi
fi
if [[ -z "${RAY:-}" ]]; then
  if [[ -x "$AREAL_ROOT/.venv/bin/ray" ]]; then
    RAY="$AREAL_ROOT/.venv/bin/ray"
  else
    RAY="$(command -v ray || true)"
  fi
fi
START_RAY="${START_RAY:-1}"
STOP_RAY_AFTER="${STOP_RAY_AFTER:-0}"
NUM_GPUS="${NUM_GPUS:-$(nvidia-smi -L 2>/dev/null | wc -l | tr -d ' ')}"
LOG_DIR="${LOG_DIR:-$ROOT/logs}"

if [[ -z "$CONFIG" ]]; then
  echo "usage: $0 configs/<experiment>.yaml" >&2
  exit 2
fi
if [[ "$CONFIG" != /* ]]; then
  CONFIG="$ROOT/$CONFIG"
fi
if [[ ! -e "$CONFIG" ]]; then
  echo "missing: $CONFIG" >&2
  exit 1
fi
if [[ -z "$PYTHON" || ! -x "$PYTHON" ]]; then
  echo "missing python. Set PYTHON=... or run: INSTALL=1 bash scripts/bootstrap_areal.sh" >&2
  exit 1
fi
if [[ "$START_RAY" == "1" ]] && { [[ -z "${RAY:-}" ]] || [[ ! -x "$RAY" ]]; }; then
  echo "missing ray. Set RAY=... or run: INSTALL=1 bash scripts/bootstrap_areal.sh" >&2
  exit 1
fi
: "${SAO_WS:=$ROOT}"
: "${MODEL_ROOT:?set MODEL_ROOT to the directory containing the model}"
: "${REWARD_ROOT:=$SAO_WS/reward}"
: "${ADMIN_API_KEY:=local-sao}"
export SAO_WS MODEL_ROOT REWARD_ROOT ADMIN_API_KEY
export PYTHONPATH="$AREAL_ROOT:$ROOT${PYTHONPATH:+:$PYTHONPATH}"
export SAO_CRITIC_K="${SAO_CRITIC_K:-2}"

mkdir -p "$LOG_DIR" "$SAO_WS/experiments" "$SAO_WS/tmp" "$REWARD_ROOT"
exp="$(basename "$CONFIG" .yaml)"
log="$LOG_DIR/${exp}_$(date +%Y%m%d_%H%M%S).log"
started_ray=0
reward_run_root=""

if [[ "$START_RAY" == "1" ]]; then
  if ! "$RAY" status >/dev/null 2>&1; then
    "$RAY" start \
      --head \
      --num-gpus="$NUM_GPUS" \
      --disable-usage-stats \
      --temp-dir="${RAY_TMPDIR:-$SAO_WS/tmp/ray}"
    started_ray=1
  fi
fi

cleanup() {
  if [[ -n "$reward_run_root" ]]; then
    "$PYTHON" "$ROOT/scripts/reward_workers_ctl.py" stop \
      --root "$reward_run_root" --grace "${REWARD_STOP_GRACE:-60}" || true
  fi
  if [[ "$started_ray" == "1" && "$STOP_RAY_AFTER" == "1" ]]; then
    "$RAY" stop --grace-period "${RAY_STOP_GRACE:-60}" || true
  fi
}
trap cleanup EXIT INT TERM

echo "experiment=$exp config=$CONFIG log=$log" | tee "$log"
readarray -t reward_meta < <("$PYTHON" - "$CONFIG" <<'PY'
import json, sys, yaml
d = yaml.safe_load(open(sys.argv[1]))
r = d.get("reward") or {}
v = r.get("fs_shard") or {}
print(r.get("backend", "local"))
print(d.get("experiment_name", "sao"))
print(d.get("trial_name", "trial"))
print(json.dumps(v.get("workers") or []))
PY
)
if [[ "${reward_meta[0]}" == "fs_shard" ]]; then
  start_out="$("$PYTHON" "$ROOT/scripts/reward_workers_ctl.py" start \
    --root "$REWARD_ROOT" \
    --experiment "${reward_meta[1]}" \
    --trial "${reward_meta[2]}" \
    --workers-json "${reward_meta[3]}" \
    --ws "$ROOT" \
    --venv "$AREAL_ROOT/.venv" \
    --repo "$AREAL_ROOT")"
  echo "$start_out" | tee -a "$log"
  reward_run_root="$(printf '%s\n' "$start_out" | "$PYTHON" -c '
import json, sys
root = ""
for line in sys.stdin:
    try:
        value = json.loads(line)
    except Exception:
        continue
    root = value.get("root", root)
print(root)
')"
  if [[ -z "$reward_run_root" ]]; then
    echo "reward workers started without reporting a run root" >&2
    exit 1
  fi
  export SAO_REWARD_FS_ROOT="$reward_run_root"
  export SAO_REWARD_RUN_ID="$(basename "$reward_run_root")"
fi

"$PYTHON" "$ROOT/scripts/train.py" --config "$CONFIG" 2>&1 | tee -a "$log"
