#!/usr/bin/env python3
"""Start / status / stop / cleanup / watch file-sharded reward workers.

P1 (issue 20260724): workers are launched under a respawn wrapper so coordinator
exits (e.g. KeyError, OOM kill) auto-restart until ctl/<host>.stop exists.
Optional `watch` also restarts a host when heartbeat is stale / pid dead.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ALLOWED_HOSTS = set(
    os.environ.get("SAO_REWARD_WORKER_HOSTS", "worker-0 worker-1").split()
)
DEFAULT_WS = os.environ.get("SAO_WS", str(ROOT))
DEFAULT_ROOT_PREFIX = os.environ.get("REWARD_ROOT", str(ROOT / "reward"))
WORKER_MODULE = "areal.reward.fs_worker"


def _ssh(host: str, remote: str, timeout: int = 120) -> tuple[int, str]:
    cmd = [
        "ssh",
        "-o",
        "BatchMode=yes",
        "-o",
        "ConnectTimeout=20",
        host,
        remote,
    ]
    try:
        out = subprocess.check_output(
            cmd, text=True, timeout=timeout, stderr=subprocess.STDOUT
        )
        return 0, out
    except subprocess.CalledProcessError as e:
        return e.returncode, e.output or ""
    except Exception as e:
        return 1, repr(e)


def _validate_workers(workers: list[dict]) -> None:
    for w in workers:
        host = w["host"]
        if host not in ALLOWED_HOSTS:
            raise SystemExit(
                f"refuse non-whitelist reward host {host!r}; "
                f"allowed={sorted(ALLOWED_HOSTS)}"
            )
        if host == "head":
            raise SystemExit("forbid reward consumer on head")


def _default_workers() -> list[dict]:
    return [
        {"host": host, "shard": shard, "concurrency": 48}
        for shard, host in enumerate(sorted(ALLOWED_HOSTS))
    ]


def _ctl_dir(root: Path) -> Path:
    d = root / "ctl"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _pidfile(root: Path, host: str) -> Path:
    return _ctl_dir(root) / f"{host}.pid"


def _wrap_pidfile(root: Path, host: str) -> Path:
    return _ctl_dir(root) / f"{host}.wrap.pid"


def _stop_sentinel(root: Path, host: str) -> Path:
    return _ctl_dir(root) / f"{host}.stop"


def _watch_pidfile(root: Path) -> Path:
    return _ctl_dir(root) / "watch.pid"


def _watch_stop_sentinel(root: Path) -> Path:
    return _ctl_dir(root) / "watch.stop"


def _load_workers(root: Path) -> list[dict]:
    meta_path = root / "ctl" / "launch_meta.json"
    workers = _default_workers()
    if meta_path.exists():
        try:
            workers = json.loads(meta_path.read_text(encoding="utf-8")).get(
                "workers"
            ) or workers
        except Exception:
            pass
    return workers


def _write_manifest(
    root: Path,
    *,
    experiment_name: str,
    trial_name: str,
    run_id: str,
    reward_fn: str,
    shard_count: int,
    workers: list[dict],
) -> None:
    from areal.reward.fs_protocol import ensure_run_dirs, write_manifest

    ensure_run_dirs(root, shard_count)
    write_manifest(
        root,
        {
            "experiment_name": experiment_name,
            "trial_name": trial_name,
            "run_id": run_id,
            "reward_fn": reward_fn,
            "shard_count": shard_count,
            "workers": workers,
        },
    )


def _paths(args: argparse.Namespace) -> tuple[str, str, str, str]:
    ws = getattr(args, "ws", None) or DEFAULT_WS
    venv = getattr(args, "venv", None) or f"{ws}/tmp/runtime-venv"
    repo = getattr(args, "repo", None) or f"{ws}/repo/AReaL"
    site = f"{venv}/lib/python3.12/site-packages"
    return ws, venv, repo, site


def _start_one_host(
    *,
    root: Path,
    host: str,
    shard: int,
    concurrency: int,
    venv: str,
    repo: str,
    site: str,
    ws: str,
    respawn: bool = True,
) -> tuple[int, str]:
    log_path = root / "ctl" / f"{host}.log"
    wrap_log = root / "ctl" / f"{host}.wrap.log"
    wrap_sh = root / "ctl" / f"{host}.wrap.sh"
    pid_path = _pidfile(root, host)
    wrap_pid_path = _wrap_pidfile(root, host)
    stop_path = _stop_sentinel(root, host)

    # Write wrap script on shared shared FS (avoid nested SSH quoting hell).
    _ctl_dir(root)
    if respawn:
        wrap_sh.write_text(
            f"""#!/usr/bin/env bash
set +e
STOP="{stop_path}"
PIDF="{pid_path}"
LOG="{log_path}"
WLOG="{wrap_log}"
PY="{venv}/bin/python"
MOD="{WORKER_MODULE}"
ROOT="{root}"
SHARD="{shard}"
HOST="{host}"
CONC="{concurrency}"
export PATH="/usr/bin:/bin:{venv}/bin:/usr/local/bin"
export PYTHONPATH="{site}:{repo}"
export HOME="{ws}/tmp/home"
export TMPDIR="{ws}/tmp"
export AREAL_MATH_VERIFY_SOFT_TIMEOUT=1
while [ ! -f "$STOP" ]; do
  echo "[wrap $(date -u +%Y-%m-%dT%H:%M:%SZ)] starting worker" >> "$WLOG"
  "$PY" -m "$MOD" --root "$ROOT" --shard "$SHARD" --host "$HOST" --concurrency "$CONC" \\
    >>"$LOG" 2>&1 &
  wpid=$!
  echo "$wpid" > "$PIDF"
  wait "$wpid"
  rc=$?
  echo "[wrap $(date -u +%Y-%m-%dT%H:%M:%SZ)] worker pid=$wpid exited rc=$rc" >> "$WLOG"
  if [ -f "$STOP" ]; then break; fi
  sleep 2
done
echo "[wrap $(date -u +%Y-%m-%dT%H:%M:%SZ)] stop sentinel seen; exit" >> "$WLOG"
""",
            encoding="utf-8",
        )
        # ensure executable bit via ssh chmod (nfs may ignore local mode)
        remote = f"""
set -euo pipefail
export PATH='/usr/bin:/bin:{venv}/bin:/usr/local/bin'
mkdir -p '{root}/ctl' '{root}/inbox/{shard}' '{root}/outbox' '{root}/status'
rm -f '{stop_path}'
for pf in '{wrap_pid_path}' '{pid_path}'; do
  if [ -f "$pf" ]; then
    old=$(cat "$pf" || true)
    if [ -n "${{old:-}}" ] && [ -d /proc/$old ]; then
      kill -TERM "$old" 2>/dev/null || true
    fi
  fi
done
sleep 1
ps -eo pid=,args= | while read -r pid args; do
  case "$args" in
    *{WORKER_MODULE}*--host*{host}*) kill -TERM "$pid" 2>/dev/null || true;;
  esac
done || true
sleep 1
chmod +x '{wrap_sh}'
nohup bash '{wrap_sh}' >>'{wrap_log}' 2>&1 &
echo $! > '{wrap_pid_path}'
# wait briefly for worker pidfile
for i in 1 2 3 4 5 6 7 8 9 10; do
  if [ -s '{pid_path}' ]; then break; fi
  sleep 0.5
done
echo started host={host} wrap_pid=$(cat '{wrap_pid_path}') worker_pid=$(cat '{pid_path}' 2>/dev/null || echo none) respawn=1
"""
    else:
        remote = f"""
set -euo pipefail
export PATH='/usr/bin:/bin:{venv}/bin:/usr/local/bin'
export PYTHONPATH='{site}:{repo}'
export HOME='{ws}/tmp/home'
export TMPDIR='{ws}/tmp'
export AREAL_MATH_VERIFY_SOFT_TIMEOUT=1
mkdir -p '{root}/ctl' '{root}/inbox/{shard}' '{root}/outbox' '{root}/status'
rm -f '{stop_path}'
if [ -f '{pid_path}' ]; then
  old=$(cat '{pid_path}' || true)
  if [ -n "${{old:-}}" ] && [ -d /proc/$old ]; then
    kill -TERM "$old" 2>/dev/null || true
    sleep 1
  fi
fi
nohup '{venv}/bin/python' -m {WORKER_MODULE} \\
  --root '{root}' --shard {shard} --host '{host}' --concurrency {concurrency} \\
  >>'{log_path}' 2>&1 &
echo $! > '{pid_path}'
echo started host={host} pid=$(cat '{pid_path}') respawn=0
"""
    return _ssh(host, remote, timeout=60)


def cmd_start(args: argparse.Namespace) -> int:
    workers = args.workers_json
    if workers:
        workers = json.loads(workers)
    else:
        workers = _default_workers()
    _validate_workers(workers)

    run_id = args.run_id or time.strftime("%Y%m%d_%H%M%S") + "_" + uuid.uuid4().hex[:8]
    root = Path(args.root)
    if args.experiment and args.trial:
        if root.name != run_id:
            root = root / args.experiment / args.trial / run_id

    shard_count = max(w["shard"] for w in workers) + 1
    if args.shard_count:
        shard_count = args.shard_count

    ws, venv, repo, site = _paths(args)
    respawn = not getattr(args, "no_respawn", False)

    _write_manifest(
        root,
        experiment_name=args.experiment or "",
        trial_name=args.trial or "",
        run_id=run_id,
        reward_fn=args.reward_fn,
        shard_count=shard_count,
        workers=workers,
    )

    meta = {
        "root": str(root),
        "run_id": run_id,
        "workers": workers,
        "shard_count": shard_count,
        "reward_fn": args.reward_fn,
        "respawn": respawn,
    }
    (root / "ctl" / "launch_meta.json").write_text(
        json.dumps(meta, indent=2), encoding="utf-8"
    )
    print(json.dumps({"ok": True, "phase": "manifest", **meta}), flush=True)

    for w in workers:
        host = w["host"]
        shard = int(w["shard"])
        concurrency = int(w.get("concurrency", 48))
        rc, out = _start_one_host(
            root=root,
            host=host,
            shard=shard,
            concurrency=concurrency,
            venv=venv,
            repo=repo,
            site=site,
            ws=ws,
            respawn=respawn,
        )
        print(out, end="" if out.endswith("\n") else out + "\n")
        if rc != 0:
            print(f"FAIL start {host} rc={rc}", file=sys.stderr)
            cmd_stop(argparse.Namespace(root=str(root), grace=30))
            return rc or 1

    # Wait for heartbeats
    deadline = time.time() + args.ready_timeout
    while time.time() < deadline:
        if _all_ready(root, workers, stale_s=args.heartbeat_stale):
            print(json.dumps({"ok": True, "phase": "ready", "root": str(root)}))
            pointer = Path(ws) / "tmp" / "sao-reward-current.json"
            pointer.parent.mkdir(parents=True, exist_ok=True)
            pointer.write_text(json.dumps(meta, indent=2), encoding="utf-8")
            return 0
        time.sleep(0.5)

    print("FAIL: workers not ready in time", file=sys.stderr)
    cmd_stop(argparse.Namespace(root=str(root), grace=30))
    return 1


def _all_ready(root: Path, workers: list[dict], stale_s: float = 15.0) -> bool:
    now = time.time()
    for w in workers:
        path = root / "status" / f"{w['host']}.json"
        if not path.exists():
            return False
        try:
            st = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return False
        if not st.get("ready"):
            return False
        if now - float(st.get("heartbeat_at", 0)) > stale_s:
            return False
    return True


def _host_health(
    root: Path, host: str, *, stale_s: float
) -> dict:
    """Return health dict: ok / reason / pid / heartbeat_age."""
    now = time.time()
    sp = root / "status" / f"{host}.json"
    pf = _pidfile(root, host)
    pid = pf.read_text().strip() if pf.exists() else ""
    out: dict = {"host": host, "pidfile": pid or None, "ok": False}
    if not sp.exists():
        out["reason"] = "status_missing"
        return out
    try:
        st = json.loads(sp.read_text(encoding="utf-8"))
    except Exception as e:
        out["reason"] = f"status_bad:{e!r}"
        return out
    age = now - float(st.get("heartbeat_at", 0))
    out["heartbeat_age_s"] = round(age, 2)
    out["status_pid"] = st.get("pid")
    out["ready"] = bool(st.get("ready"))
    if age > stale_s:
        out["reason"] = "heartbeat_stale"
        return out
    if not st.get("ready"):
        out["reason"] = "not_ready"
        return out
    out["ok"] = True
    out["reason"] = "ok"
    return out


def cmd_status(args: argparse.Namespace) -> int:
    root = Path(args.root)
    meta_path = root / "ctl" / "launch_meta.json"
    workers = _load_workers(root)
    if meta_path.exists():
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        print(json.dumps({"meta": meta}, indent=2))
    statuses = {}
    for w in workers:
        host = w["host"]
        statuses[host] = _host_health(root, host, stale_s=getattr(args, "heartbeat_stale", 15.0))
        pf = _pidfile(root, host)
        wpf = _wrap_pidfile(root, host)
        if pf.exists():
            statuses[host]["pidfile"] = pf.read_text().strip()
        if wpf.exists():
            statuses[host]["wrap_pidfile"] = wpf.read_text().strip()
        if (root / "status" / f"{host}.json").exists():
            try:
                statuses[host]["raw"] = json.loads(
                    (root / "status" / f"{host}.json").read_text(encoding="utf-8")
                )
            except Exception:
                pass
    print(json.dumps({"status": statuses}, indent=2))
    return 0


def cmd_stop(args: argparse.Namespace) -> int:
    root = Path(args.root)
    workers = _load_workers(root)
    grace = getattr(args, "grace", 60)

    # Stop local/remote watch first
    wsp = _watch_stop_sentinel(root)
    wsp.write_text(str(time.time()), encoding="utf-8")
    wpf = _watch_pidfile(root)
    if wpf.exists():
        try:
            wpid = int(wpf.read_text().strip())
            os.kill(wpid, 15)
        except Exception:
            pass

    for w in workers:
        host = w["host"]
        if host not in ALLOWED_HOSTS:
            continue
        pid_path = _pidfile(root, host)
        wrap_pid_path = _wrap_pidfile(root, host)
        stop_path = _stop_sentinel(root, host)
        remote = f"""
set +e
export PATH=/usr/bin:/bin:/usr/local/bin
# tell wrap loop to exit
mkdir -p '{root}/ctl'
echo stop > '{stop_path}'
# TERM wrap then worker
for pf in '{wrap_pid_path}' '{pid_path}'; do
  if [ -f "$pf" ]; then
    pid=$(cat "$pf")
    if [ -n "$pid" ] && [ -d /proc/$pid ]; then
      kill -TERM "$pid" 2>/dev/null
      echo TERM $pid on {host} via $pf
      for i in $(seq 1 {grace}); do
        if [ ! -d /proc/$pid ]; then echo exited $pid; break; fi
        sleep 1
      done
      if [ -d /proc/$pid ]; then
        echo WARN still alive after {grace}s — not using kill -9
      fi
    else
      echo pidfile stale $pf on {host}
    fi
  fi
done
# belt: exact module cmdline match only
ps -eo pid=,args= | while read -r pid args; do
  case "$args" in
    *{WORKER_MODULE}*--host*{host}*) kill -TERM "$pid" 2>/dev/null; echo TERM_match $pid;;
    *{host}.wrap.sh*) kill -TERM "$pid" 2>/dev/null; echo TERM_wrap_sh $pid;;
  esac
done || true
true
"""
        rc, out = _ssh(host, remote, timeout=grace + 30)
        print(out, end="" if out.endswith("\n") else out + "\n")
    return 0


def cmd_restart_host(args: argparse.Namespace) -> int:
    """Restart a single host under the same root (used by watch / smoke)."""
    root = Path(args.root)
    host = args.host
    if host not in ALLOWED_HOSTS:
        raise SystemExit(f"refuse host {host}")
    workers = _load_workers(root)
    w = next((x for x in workers if x["host"] == host), None)
    if not w:
        raise SystemExit(f"host {host} not in launch_meta workers")
    ws, venv, repo, site = _paths(args)
    # Clear stop so wrap can run again
    sp = _stop_sentinel(root, host)
    if sp.exists():
        sp.unlink()
    rc, out = _start_one_host(
        root=root,
        host=host,
        shard=int(w["shard"]),
        concurrency=int(w.get("concurrency", 48)),
        venv=venv,
        repo=repo,
        site=site,
        ws=ws,
        respawn=True,
    )
    print(out, end="" if out.endswith("\n") else out + "\n")
    return rc


def cmd_watch(args: argparse.Namespace) -> int:
    """Poll status/*.json; restart hosts with stale heartbeat or dead wrap."""
    root = Path(args.root)
    workers = _load_workers(root)
    stale_s = float(args.heartbeat_stale)
    interval = float(args.interval)
    stop_path = _watch_stop_sentinel(root)
    if stop_path.exists():
        stop_path.unlink()
    _watch_pidfile(root).write_text(str(os.getpid()), encoding="utf-8")
    log_path = root / "ctl" / "watch.log"

    def wlog(msg: str) -> None:
        line = f"[{time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}] {msg}"
        print(line, flush=True)
        try:
            with log_path.open("a", encoding="utf-8") as f:
                f.write(line + "\n")
        except OSError:
            pass

    wlog(
        f"watch start root={root} stale={stale_s}s interval={interval}s "
        f"hosts={[w['host'] for w in workers]}"
    )
    # cooldown per host to avoid restart storms
    last_restart: dict[str, float] = {}
    cooldown = float(args.restart_cooldown)

    try:
        while not stop_path.exists():
            for w in workers:
                host = w["host"]
                h = _host_health(root, host, stale_s=stale_s)
                if h.get("ok"):
                    continue
                now = time.time()
                if now - last_restart.get(host, 0) < cooldown:
                    continue
                wlog(
                    f"UNHEALTHY {host} reason={h.get('reason')} "
                    f"age={h.get('heartbeat_age_s')} → restart"
                )
                ns = argparse.Namespace(
                    root=str(root),
                    host=host,
                    ws=getattr(args, "ws", None),
                    venv=getattr(args, "venv", None),
                    repo=getattr(args, "repo", None),
                )
                rc = cmd_restart_host(ns)
                last_restart[host] = now
                wlog(f"restart {host} rc={rc}")
            time.sleep(interval)
    finally:
        try:
            _watch_pidfile(root).unlink(missing_ok=True)  # type: ignore[arg-type]
        except TypeError:
            # py3.9
            p = _watch_pidfile(root)
            if p.exists():
                p.unlink()
        wlog("watch stopped")
    return 0


def cmd_cleanup(args: argparse.Namespace) -> int:
    root = Path(args.root)
    if not str(root).startswith("${DATA_ROOT}/"):
        raise SystemExit(f"refuse cleanup outside allowed reward root: {root}")
    if root.exists():
        shutil.rmtree(root)
        print(f"removed {root}")
    else:
        print(f"missing {root}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ws", default=DEFAULT_WS)
    parser.add_argument("--venv", default=None)
    parser.add_argument("--repo", default=None)
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_start = sub.add_parser("start")
    p_start.add_argument("--root", default=DEFAULT_ROOT_PREFIX)
    p_start.add_argument("--experiment", default="")
    p_start.add_argument("--trial", default="")
    p_start.add_argument("--run-id", default=None)
    p_start.add_argument(
        "--reward-fn", default="areal.reward.gsm8k.gsm8k_reward_fn"
    )
    p_start.add_argument("--workers-json", default=None)
    p_start.add_argument("--shard-count", type=int, default=None)
    p_start.add_argument("--ready-timeout", type=float, default=60.0)
    p_start.add_argument("--heartbeat-stale", type=float, default=15.0)
    p_start.add_argument(
        "--no-respawn",
        action="store_true",
        help="disable outer wrap loop (legacy one-shot nohup)",
    )
    p_start.add_argument("--ws", default=None)
    p_start.add_argument("--venv", default=None)
    p_start.add_argument("--repo", default=None)

    p_status = sub.add_parser("status")
    p_status.add_argument("--root", required=True)
    p_status.add_argument("--heartbeat-stale", type=float, default=15.0)

    p_stop = sub.add_parser("stop")
    p_stop.add_argument("--root", required=True)
    p_stop.add_argument("--grace", type=int, default=60)

    p_restart = sub.add_parser("restart-host")
    p_restart.add_argument("--root", required=True)
    p_restart.add_argument("--host", required=True, choices=sorted(ALLOWED_HOSTS))
    p_restart.add_argument("--ws", default=None)
    p_restart.add_argument("--venv", default=None)
    p_restart.add_argument("--repo", default=None)

    p_watch = sub.add_parser(
        "watch",
        help="poll heartbeats; restart stale hosts (run on a node with shared FS + ssh)",
    )
    p_watch.add_argument("--root", required=True)
    p_watch.add_argument("--heartbeat-stale", type=float, default=60.0)
    p_watch.add_argument("--interval", type=float, default=10.0)
    p_watch.add_argument("--restart-cooldown", type=float, default=30.0)
    p_watch.add_argument("--ws", default=None)
    p_watch.add_argument("--venv", default=None)
    p_watch.add_argument("--repo", default=None)

    p_cleanup = sub.add_parser("cleanup")
    p_cleanup.add_argument("--root", required=True)

    args = parser.parse_args(argv)
    if args.cmd == "start":
        return cmd_start(args)
    if args.cmd == "status":
        return cmd_status(args)
    if args.cmd == "stop":
        return cmd_stop(args)
    if args.cmd == "restart-host":
        return cmd_restart_host(args)
    if args.cmd == "watch":
        return cmd_watch(args)
    if args.cmd == "cleanup":
        return cmd_cleanup(args)
    return 2


if __name__ == "__main__":
    # Can run locally or via SSH to reward worker hosts.
    # Ensure AReaL import path when writing manifest locally.
    ws = os.environ.get("SAO_WS", DEFAULT_WS)
    repo = f"{ws}/repo/AReaL"
    if Path(repo).exists() and repo not in sys.path:
        sys.path.insert(0, repo)
    # Local Mac checkout
    local_repo = Path(__file__).resolve().parents[1] / "repo" / "AReaL"
    if local_repo.exists() and str(local_repo) not in sys.path:
        sys.path.insert(0, str(local_repo))
    raise SystemExit(main())
