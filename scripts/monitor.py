#!/usr/bin/env python3
"""Small dependency-free dashboard for SAO/AReaL training logs."""

from __future__ import annotations

import argparse
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from sao.metrics import collect_metrics

HTML = r"""<!doctype html>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>SAO training monitor</title>
<style>
body{font:14px system-ui;margin:24px;background:#0b1020;color:#dce6ff}
h1{font-size:22px}.cards{display:flex;gap:10px;flex-wrap:wrap}
.card{background:#151d34;border:1px solid #293657;border-radius:9px;padding:12px;min-width:170px}
.bad{color:#ff7b7b}.ok{color:#75e6a4}canvas{background:#fff;border-radius:9px;width:100%;height:360px;margin-top:16px}
code{color:#a9c6ff}
</style>
<h1>SAO training monitor</h1>
<div id="meta"></div><div class="cards" id="cards"></div>
<canvas id="chart" width="1200" height="420"></canvas>
<script>
const colors={sao:"#1f77b4",grpo:"#d62728",grpo_dis:"#2ca02c",grpo_dis_g1:"#ff7f0e",running_mean:"#9467bd"};
function last(a,k){const x=a.series[k]||[];return x.length?x[x.length-1].value:null}
function draw(data,key="acc"){
 const c=document.querySelector("canvas"),x=c.getContext("2d"),W=c.width,H=c.height,p=50;
 x.clearRect(0,0,W,H);x.strokeStyle="#ccd3df";x.strokeRect(p,20,W-p-20,H-p-20);
 x.fillStyle="#172033";x.font="14px system-ui";x.fillText(key==="acc"?"online accuracy":"reward",p,15);
 let maxStep=1,maxY=key==="acc"?1:0;
 Object.values(data.experiments).forEach(a=>(a.series[key]||[]).forEach(q=>{maxStep=Math.max(maxStep,q.step);maxY=Math.max(maxY,q.value)}));
 Object.entries(data.experiments).forEach(([name,a])=>{
   const pts=a.series[key]||[];if(!pts.length)return;x.beginPath();x.strokeStyle=colors[name]||"#333";x.lineWidth=2;
   pts.forEach((q,i)=>{const px=p+q.step/maxStep*(W-p-20),py=20+(1-q.value/maxY)*(H-p-20);i?x.lineTo(px,py):x.moveTo(px,py)});x.stroke();
 });
 let y=42;Object.keys(data.experiments).forEach(name=>{x.fillStyle=colors[name]||"#333";x.fillRect(W-210,y-10,14,3);x.fillStyle="#172033";x.fillText(name,W-188,y-5);y+=20});
}
async function refresh(){
 const d=await (await fetch("/api/metrics")).json();
 document.getElementById("meta").innerHTML=`logs: <code>${d.log_dir}</code> · auto refresh 15s`;
 document.getElementById("cards").innerHTML=Object.entries(d.experiments).map(([name,a])=>{
  const acc=last(a,"acc"),reward=last(a,"reward"),bad=a.health.oom||a.health.nan_or_inf;
  return `<div class="card"><b>${name}</b><br>step ${a.last_step}/${a.total_steps}<br>acc ${acc==null?"—":(100*acc).toFixed(1)+"%"}<br>reward ${reward==null?"—":reward.toFixed(3)}<br><span class="${bad?"bad":"ok"}">${bad?"OOM / NaN detected":a.health.completed?"completed":"running / stopped"}</span></div>`
 }).join("");draw(d);
}
refresh();setInterval(refresh,15000);
</script>"""


def handler(log_dir: Path):
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:  # noqa: N802
            if self.path == "/api/metrics":
                body = json.dumps(collect_metrics(log_dir)).encode()
                content_type = "application/json"
            elif self.path in {"/", "/index.html"}:
                body = HTML.encode()
                content_type = "text/html; charset=utf-8"
            else:
                self.send_error(404)
                return
            self.send_response(200)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, format: str, *args: object) -> None:
            return

    return Handler


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--log-dir", type=Path, default=Path("logs"))
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8790)
    args = parser.parse_args()
    server = ThreadingHTTPServer((args.host, args.port), handler(args.log_dir.resolve()))
    print(f"SAO monitor: http://{args.host}:{args.port}/ (logs={args.log_dir.resolve()})")
    server.serve_forever()


if __name__ == "__main__":
    main()
