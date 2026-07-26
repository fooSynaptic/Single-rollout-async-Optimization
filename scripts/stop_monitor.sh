#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
pidfile="${RUN_DIR:-$ROOT/.run}/monitor.pid"
if [[ ! -f "$pidfile" ]]; then
  echo "monitor is not running"
  exit 0
fi
pid="$(cat "$pidfile")"
if kill -0 "$pid" 2>/dev/null; then
  kill -TERM "$pid"
  echo "monitor stopped: pid=$pid"
fi
rm -f "$pidfile"
