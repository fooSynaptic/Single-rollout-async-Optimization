#!/usr/bin/env bash
# CLUSTER-ONLY smoke: needs SSH aliases worker-0 / worker-1 (or edit hosts),
# shared reward root, and a patched AReaL + runtime python. Not part of the default
# third-party quick start — see README Q1 and docs/features/shared-fs-reward.md.
# Start shared FS reward workers on train nodes and verify heartbeats (no train).
# Usage: SAO_WS=... bash scripts/smoke_fs_reward_cluster_workers.sh
set -euo pipefail
WS=${SAO_WS:?set SAO_WS}
VENV=$WS/tmp/runtime-venv
export PATH="$VENV/bin:/usr/local/bin:$PATH"
export PYTHONPATH="$VENV/lib/python3.12/site-packages:$WS/repo/AReaL${PYTHONPATH:+:$PYTHONPATH}"

OUT=$("$VENV/bin/python" "$WS/scripts/reward_workers_ctl.py" start \
  --root ${REWARD_ROOT} \
  --experiment smoke-shared-fs \
  --trial cluster-workers \
  --ws "$WS" \
  --venv "$VENV" \
  --repo "$WS/repo/AReaL")
echo "$OUT"
ROOT=$(echo "$OUT" | python3 -c "
import sys,json
root=''
for line in sys.stdin:
  line=line.strip()
  if line.startswith('{') and 'root' in line:
    try:
      d=json.loads(line)
      if d.get('root'): root=d['root']
    except Exception:
      pass
print(root)
")
echo "ROOT=$ROOT"
"$VENV/bin/python" "$WS/scripts/reward_workers_ctl.py" status --root "$ROOT"
# quick submit from head using client
"$VENV/bin/python" - <<PY
import asyncio, json
from areal.reward.fs_client import FSRewardClient
client = FSRewardClient(
  "areal.reward.simple_answer.simple_answer_reward_fn",
  root="$ROOT",
  shard_count=2,
  timeout_seconds=30,
  on_error="raise",
  worker_hosts=["worker-0","worker-1"],
)
async def main():
  scores = await asyncio.gather(*[
    client("q", f"The answer is {i}", [], [], answer=str(i)) for i in range(32)
  ])
  print(json.dumps({"scores": scores, "ok": all(s==1.0 for s in scores), "stats": client.stats}))
asyncio.run(main())
PY
"$VENV/bin/python" "$WS/scripts/reward_workers_ctl.py" stop --root "$ROOT" --grace 30
"$VENV/bin/python" "$WS/scripts/reward_workers_ctl.py" cleanup --root "$ROOT"
echo cluster_workers_smoke_ok
