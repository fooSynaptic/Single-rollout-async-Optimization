#!/usr/bin/env bash
# CLUSTER-ONLY smoke: shared FS reward respawn wrapper — kill worker, expect auto-restart.
# Requires SSH to cluster roles and a patched AReaL checkout. Not in default quick start.
# Run with SAO_WS set; see docs/features/shared-fs-reward.md.
set -euo pipefail

WS="${SAO_WS:?set SAO_WS}"
# Prefer local checkout for ctl script when present
LOCAL_WS="$(cd "$(dirname "$0")/.." && pwd)"
CTL_PY="${LOCAL_WS}/scripts/reward_workers_ctl.py"
VENV="${WS}/tmp/runtime-venv"
PY="${VENV}/bin/python"
KILL_AFTER_SEC="${KILL_AFTER_SEC:-8}"
WAIT_RESTART_SEC="${WAIT_RESTART_SEC:-45}"
HOST="${KILL_HOST:-worker-0}"
RUN_ID="smoke_respawn_$(date -u +%Y%m%d_%H%M%S)"

echo "=== sync code to shared FS (fs_worker + reward_workers_ctl) ==="
ssh head "mkdir -p '${WS}/scripts' '${WS}/repo/AReaL/areal/reward' '${WS}/repo/AReaL/tests'"
rsync -az "${LOCAL_WS}/scripts/reward_workers_ctl.py" "head:${WS}/scripts/reward_workers_ctl.py"
rsync -az "${LOCAL_WS}/repo/AReaL/areal/reward/fs_worker.py" \
  "head:${WS}/repo/AReaL/areal/reward/fs_worker.py"
rsync -az "${LOCAL_WS}/repo/AReaL/tests/test_fs_worker_drain.py" \
  "head:${WS}/repo/AReaL/tests/test_fs_worker_drain.py" 2>/dev/null || true

echo "=== unit test P0 drain (on head, no pytest) ==="
ssh head "cd '${WS}/repo/AReaL' && PYTHONPATH='${WS}/tmp/runtime-venv/lib/python3.12/site-packages:${WS}/repo/AReaL' \
  '${PY}' - <<'PY'
from pathlib import Path
import tempfile
from concurrent.futures.process import BrokenProcessPool
from unittest.mock import MagicMock
from areal.reward.fs_worker import FSRewardWorker

class _BrokenFut:
    def __init__(self, err): self._err = err
    def done(self): return True
    def result(self): raise self._err

td = Path(tempfile.mkdtemp(dir='${TMP_ROOT}'))
w = FSRewardWorker(root=td, shard=0, host='worker-0', concurrency=4)
(td / 'outbox').mkdir(parents=True)
(td / 'inbox' / '0').mkdir(parents=True)
a = td / 'inbox' / '0' / 'a.json'; a.write_text('{}')
b = td / 'inbox' / '0' / 'b.json'; b.write_text('{}')
fut1 = _BrokenFut(BrokenProcessPool('pool dead'))
fut2 = _BrokenFut(BrokenProcessPool('pool dead'))
w._inflight[fut1] = {'task_id': 'a', 'inbox_path': a, 'created_at': 1.0, 'submit_at_wall': 1.1}
w._inflight[fut2] = {'task_id': 'b', 'inbox_path': b, 'created_at': 1.0, 'submit_at_wall': 1.1}
w._recreate_pool = MagicMock(wraps=w._recreate_pool)
w._drain_done()
assert w._recreate_pool.called, 'recreate not called'
assert w._inflight == {}, 'inflight not cleared'
assert (td / 'outbox' / 'a.json').exists(), 'no outbox for first task'
print('PASS P0 drain on head')
PY
"

echo "=== start workers with respawn wrap ==="
START_OUT=$(ssh head "cd '${WS}' && '${PY}' scripts/reward_workers_ctl.py start \
  --root ${REWARD_ROOT} \
  --experiment smoke-respawn --trial wrap --run-id '${RUN_ID}' \
  --ready-timeout 90 \
  --workers-json '[{\"host\":\"worker-0\",\"shard\":0,\"concurrency\":8},{\"host\":\"worker-1\",\"shard\":1,\"concurrency\":8}]'")
echo "$START_OUT"
ROOT=$(echo "$START_OUT" | python3 -c "
import sys,json
root=None
for line in sys.stdin.read().splitlines():
  line=line.strip()
  if line.startswith('{') and 'root' in line:
    try:
      d=json.loads(line)
      if d.get('root'): root=d['root']
    except Exception:
      pass
print(root or '')
")
if [[ -z "$ROOT" ]]; then
  echo "FAIL: could not parse ROOT from start output"
  exit 1
fi
echo "ROOT=$ROOT"

echo "=== status before kill ==="
ssh head "${PY} ${WS}/scripts/reward_workers_ctl.py status --root '${ROOT}'" | tail -40

OLD_PID=$(ssh "$HOST" "cat '${ROOT}/ctl/${HOST}.pid'")
WRAP_PID=$(ssh "$HOST" "cat '${ROOT}/ctl/${HOST}.wrap.pid'")
echo "OLD_PID=$OLD_PID WRAP_PID=$WRAP_PID on $HOST"

echo "=== schedule kill worker in ${KILL_AFTER_SEC}s (wrap must stay) ==="
ssh "$HOST" "nohup bash -c 'sleep ${KILL_AFTER_SEC}; kill -TERM ${OLD_PID}; echo killed ${OLD_PID} at \$(date -u)' \
  >>'${ROOT}/ctl/smoke_kill.log' 2>&1 &"

echo "=== wait for respawn (up to ${WAIT_RESTART_SEC}s) ==="
OK=0
for i in $(seq 1 "$WAIT_RESTART_SEC"); do
  sleep 1
  NEW_PID=$(ssh "$HOST" "cat '${ROOT}/ctl/${HOST}.pid' 2>/dev/null || true")
  HB=$(ssh head "python3 -c \"
import json,time
from pathlib import Path
p=Path('${ROOT}/status/${HOST}.json')
if not p.exists():
  print('missing'); raise SystemExit
st=json.loads(p.read_text())
age=time.time()-float(st.get('heartbeat_at',0))
print(f\\\"pid={st.get('pid')} age={age:.1f} ready={st.get('ready')}\\\")
\"")
  echo "[$i] pidfile=$NEW_PID status=$HB"
  if [[ -n "$NEW_PID" && "$NEW_PID" != "$OLD_PID" ]]; then
    # fresh heartbeat?
    AGE=$(echo "$HB" | sed -n 's/.*age=\([0-9.]*\).*/\1/p')
    if python3 -c "import sys; sys.exit(0 if float('${AGE:-999}') < 15 else 1)"; then
      OK=1
      break
    fi
  fi
done

echo "=== wrap log tail ==="
ssh "$HOST" "tail -20 '${ROOT}/ctl/${HOST}.wrap.log' || true"

echo "=== stop + cleanup ==="
ssh head "${PY} ${WS}/scripts/reward_workers_ctl.py stop --root '${ROOT}' --grace 20"
ssh head "${PY} ${WS}/scripts/reward_workers_ctl.py cleanup --root '${ROOT}'"

if [[ "$OK" -eq 1 ]]; then
  echo "PASS: worker ${OLD_PID} -> ${NEW_PID} auto-restarted on ${HOST}"
  exit 0
fi
echo "FAIL: worker did not restart with fresh heartbeat within ${WAIT_RESTART_SEC}s"
exit 1
