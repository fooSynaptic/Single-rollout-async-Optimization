#!/usr/bin/env bash
# Local / shared-dir CPU smoke for shared FS reward (no Ray / no GPU).
# Spawns two worker *processes* (not threads) so ProcessPool is safe.
# Usage:
#   N=1280 bash scripts/smoke_fs_reward_cpu.sh
set -euo pipefail

if [ -z "${WS:-}" ]; then
  if [ -d ${SAO_WS} ]; then
    WS=${SAO_WS}
  else
    WS=${SAO_WS}
  fi
fi
REPO=${REPO:-$WS/repo/AReaL}
N=${N:-1280}
if [ -d ${TMP_ROOT} ]; then
  ROOT=${ROOT:-${REWARD_ROOT}-smoke-$(date +%Y%m%d_%H%M%S)}
  mkdir -p "$ROOT"
else
  ROOT=${ROOT:-$(mktemp -d /tmp/sao-reward-smoke.XXXXXX)}
fi
export PYTHONPATH="$REPO${PYTHONPATH:+:$PYTHONPATH}"
export AREAL_MATH_VERIFY_SOFT_TIMEOUT=${AREAL_MATH_VERIFY_SOFT_TIMEOUT:-1}
PY=${PY:-python3}

echo "[smoke] root=$ROOT N=$N"

# Manifest
"$PY" - <<PY
from pathlib import Path
from areal.reward.fs_protocol import write_manifest
root = Path("$ROOT")
write_manifest(root, {
  "shard_count": 2,
  "run_id": "smoke",
  "reward_fn": "areal.reward.simple_answer.simple_answer_reward_fn",
})
print("manifest_ok")
PY

# Start two worker processes
"$PY" -m areal.reward.fs_worker --root "$ROOT" --shard 0 --host worker-0 --concurrency 8 \
  >"$ROOT/w0.log" 2>&1 &
PID0=$!
"$PY" -m areal.reward.fs_worker --root "$ROOT" --shard 1 --host worker-1 --concurrency 8 \
  >"$ROOT/w1.log" 2>&1 &
PID1=$!
cleanup() {
  kill -TERM "$PID0" "$PID1" 2>/dev/null || true
  wait "$PID0" "$PID1" 2>/dev/null || true
}
trap cleanup EXIT

# Wait ready
"$PY" - <<PY
import json, time
from pathlib import Path
root = Path("$ROOT")
deadline = time.time() + 30
while time.time() < deadline:
    ok = True
    for host in ("worker-0", "worker-1"):
        p = root / "status" / f"{host}.json"
        if not p.exists():
            ok = False
            break
        if not json.loads(p.read_text()).get("ready"):
            ok = False
            break
    if ok:
        print("workers_ready")
        raise SystemExit(0)
    time.sleep(0.1)
print("workers not ready")
raise SystemExit(1)
PY

# Client vs local
"$PY" - <<PY
import asyncio, json, time
from pathlib import Path
from areal.reward.fs_client import FSRewardClient
from areal.api.reward_api import build_async_reward_executor
from areal.api.cli_args import RewardConfig
from areal.reward.simple_answer import simple_answer_reward_fn

root = Path("$ROOT")
N = int("$N")
client = FSRewardClient(
    "areal.reward.simple_answer.simple_answer_reward_fn",
    root=root,
    shard_count=2,
    timeout_seconds=60,
    poll_interval_ms=20,
    on_error="raise",
    worker_hosts=["worker-0", "worker-1"],
)

async def run():
    local = build_async_reward_executor(
        simple_answer_reward_fn, RewardConfig(backend="local", max_retries=0, max_workers=4)
    )
    corpus = []
    for i in range(N):
        ans = str(i % 50)
        comp = f"The answer is {ans}" if i % 3 else f"wrong {i}"
        corpus.append((comp, ans, 1.0 if ans in comp else 0.0))

    t0 = time.perf_counter()
    local_scores = [await local("q", c, [], [], answer=a) for c, a, _ in corpus]
    t_local = time.perf_counter() - t0

    sem = asyncio.Semaphore(64)
    async def one(c, a):
        async with sem:
            return await client("q", c, [], [], answer=a)

    t0 = time.perf_counter()
    fs_scores = await asyncio.gather(*[one(c, a) for c, a, _ in corpus])
    t_fs = time.perf_counter() - t0

    expect = [e for _, _, e in corpus]
    mism = sum(1 for a, b, e in zip(local_scores, fs_scores, expect) if a != b or b != e)
    print(json.dumps({
        "n": N,
        "mismatch": mism,
        "local_sec": round(t_local, 3),
        "fs_sec": round(t_fs, 3),
        "fs_qps": round(N / max(t_fs, 1e-6), 1),
        "client_stats": client.stats,
        "ok": mism == 0,
    }, indent=2))
    return mism == 0

ok = asyncio.run(run())
raise SystemExit(0 if ok else 1)
PY
