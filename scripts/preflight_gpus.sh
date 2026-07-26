#!/usr/bin/env bash
# Preflight: reject nodes/GPUs with zombie VRAM before training.
# Usage:
#   bash preflight_gpus.sh head worker-0 worker-1
#   MAX_USED_MIB=500 bash preflight_gpus.sh head
# Exit 0 = all listed hosts clean enough; 1 = dirty or unreachable.
set -euo pipefail

MAX_USED_MIB=${MAX_USED_MIB:-500}
HOSTS=("$@")
if [ "${#HOSTS[@]}" -eq 0 ]; then
  echo "usage: $0 <host> [host...]" >&2
  exit 2
fi

dirty=0
for host in "${HOSTS[@]}"; do
  echo "=== preflight $host (max_used=${MAX_USED_MIB}MiB) ==="
  if ! out=$(ssh -o BatchMode=yes -o ConnectTimeout=15 "$host" bash -s <<'EOS'
set -euo pipefail
echo "-- memory.used --"
nvidia-smi --query-gpu=index,memory.used,memory.free --format=csv,noheader
echo "-- compute-apps --"
nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_memory --format=csv,noheader || true
echo "-- train/ray leftovers --"
ps -eo pid,cmd | grep -E '[p]hase2_gsm8k|[r]aylet|[v]llm' | head -20 || echo "(none)"
EOS
  ); then
    echo "FAIL: cannot ssh $host" >&2
    dirty=1
    continue
  fi
  echo "$out"

  # Any [Not Found] with significant VRAM => zombie (tiny leftover < MAX is OK)
  while IFS= read -r line; do
    if echo "$line" | grep -q '\[Not Found\]'; then
      mem=$(echo "$line" | awk -F',' '{print $NF}' | tr -dc '0-9')
      if [ -n "$mem" ] && [ "$mem" -gt "$MAX_USED_MIB" ]; then
        echo "FAIL: $host zombie compute-app ${mem}MiB > ${MAX_USED_MIB}: $line" >&2
        dirty=1
      else
        echo "WARN: $host tiny/orphan app ignored (<=${MAX_USED_MIB}MiB): $line" >&2
      fi
    fi
  done < <(echo "$out" | sed -n '/-- compute-apps --/,/-- train\/ray leftovers --/{//!p;}')

  # Per-GPU used memory
  while IFS=',' read -r idx used free; do
    idx=$(echo "$idx" | tr -d ' ')
    used_num=$(echo "$used" | tr -dc '0-9')
    if [ -n "$used_num" ] && [ "$used_num" -gt "$MAX_USED_MIB" ]; then
      echo "FAIL: $host GPU$idx memory.used=${used_num}MiB > ${MAX_USED_MIB}" >&2
      dirty=1
    fi
  done < <(echo "$out" | sed -n '/-- memory.used --/,/-- compute-apps --/{//!p;}')
done

if [ "$dirty" -ne 0 ]; then
  echo "PREFLIGHT FAIL: do not launch training on dirty GPUs. Bypass with CVD or L5 reboot." >&2
  exit 1
fi
echo "PREFLIGHT OK: ${HOSTS[*]}"
exit 0
