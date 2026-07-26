#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${PYTHON:-python3}"
PORT="${PORT:-8790}"
HOST="${HOST:-127.0.0.1}"
LOG_DIR="${LOG_DIR:-$ROOT/logs}"
RUN_DIR="${RUN_DIR:-$ROOT/.run}"
pidfile="$RUN_DIR/monitor.pid"
outfile="$RUN_DIR/monitor.log"
mkdir -p "$RUN_DIR" "$LOG_DIR"

if [[ -f "$pidfile" ]]; then
  pid="$(cat "$pidfile")"
  if kill -0 "$pid" 2>/dev/null; then
    echo "monitor already running: pid=$pid http://$HOST:$PORT/"
    exit 0
  fi
fi

nohup env PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}" \
  "$PYTHON" "$ROOT/scripts/monitor.py" \
  --host "$HOST" --port "$PORT" --log-dir "$LOG_DIR" \
  >"$outfile" 2>&1 &
echo $! >"$pidfile"
sleep 1

pid="$(cat "$pidfile")"
if ! kill -0 "$pid" 2>/dev/null; then
  echo "monitor failed to start; see $outfile" >&2
  exit 1
fi
echo "monitor started: pid=$pid http://$HOST:$PORT/"
