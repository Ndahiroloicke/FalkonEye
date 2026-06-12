"""Simple local web dashboard for FaceLocking live stats."""

from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Optional

from . import config
from .dashboard_state import snapshot

DASHBOARD_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>FaceLocking Dashboard</title>
  <style>
    :root {
      --bg: #0f1419;
      --card: #1a2332;
      --border: #2d3a4f;
      --text: #e7ecf3;
      --muted: #8b9cb3;
      --accent: #3b82f6;
      --ok: #22c55e;
      --warn: #f59e0b;
      --bad: #ef4444;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      font-family: "Segoe UI", system-ui, sans-serif;
      background: var(--bg);
      color: var(--text);
      min-height: 100vh;
    }
    header {
      padding: 1.25rem 1.5rem;
      border-bottom: 1px solid var(--border);
      display: flex;
      justify-content: space-between;
      align-items: center;
      flex-wrap: wrap;
      gap: 0.75rem;
    }
    h1 { margin: 0; font-size: 1.35rem; font-weight: 600; }
    .badge {
      padding: 0.25rem 0.65rem;
      border-radius: 999px;
      font-size: 0.8rem;
      font-weight: 600;
      text-transform: uppercase;
    }
    .badge.live { background: rgba(34,197,94,0.2); color: var(--ok); }
    .badge.stale { background: rgba(245,158,11,0.2); color: var(--warn); }
    .badge.off { background: rgba(239,68,68,0.2); color: var(--bad); }
    main { padding: 1.25rem 1.5rem 2rem; max-width: 1100px; margin: 0 auto; }
    .grid {
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
      gap: 1rem;
      margin-bottom: 1rem;
    }
    .card {
      background: var(--card);
      border: 1px solid var(--border);
      border-radius: 12px;
      padding: 1rem 1.1rem;
    }
    .card h2 {
      margin: 0 0 0.75rem;
      font-size: 0.78rem;
      text-transform: uppercase;
      letter-spacing: 0.06em;
      color: var(--muted);
      font-weight: 600;
    }
    .big {
      font-size: 2rem;
      font-weight: 700;
      line-height: 1.1;
    }
    .sub { color: var(--muted); font-size: 0.88rem; margin-top: 0.35rem; }
    .row { display: flex; justify-content: space-between; padding: 0.35rem 0; font-size: 0.92rem; }
    .row span:last-child { font-weight: 600; }
    .state-IDLE { color: var(--muted); }
    .state-TRACKING { color: #38bdf8; }
    .state-LOCKED { color: var(--ok); }
    .state-SEARCHING { color: var(--warn); }
    .gauge-wrap { margin-top: 0.5rem; }
    .gauge {
      height: 14px;
      background: #111827;
      border-radius: 999px;
      overflow: hidden;
      border: 1px solid var(--border);
      position: relative;
    }
    .gauge-fill {
      height: 100%;
      background: linear-gradient(90deg, #2563eb, #38bdf8);
      width: 50%;
      transition: width 0.25s ease;
    }
    .gauge-target {
      position: absolute;
      top: -2px;
      width: 3px;
      height: 18px;
      background: var(--warn);
      border-radius: 2px;
      transform: translateX(-50%);
    }
    .gauge-labels {
      display: flex;
      justify-content: space-between;
      font-size: 0.75rem;
      color: var(--muted);
      margin-top: 0.35rem;
    }
    .face-bar {
      height: 28px;
      background: #111827;
      border-radius: 8px;
      border: 1px solid var(--border);
      position: relative;
      margin: 0.75rem 0 0.5rem;
    }
    .face-center {
      position: absolute;
      top: 0; bottom: 0;
      left: 50%;
      width: 2px;
      background: rgba(255,255,255,0.35);
    }
    .face-dot {
      position: absolute;
      top: 50%;
      width: 14px; height: 14px;
      background: var(--ok);
      border-radius: 50%;
      transform: translate(-50%, -50%);
      box-shadow: 0 0 10px rgba(34,197,94,0.6);
      transition: left 0.2s ease;
    }
    canvas {
      width: 100%;
      height: 140px;
      display: block;
      margin-top: 0.5rem;
      background: #111827;
      border-radius: 8px;
      border: 1px solid var(--border);
    }
  </style>
</head>
<body>
  <header>
    <h1>FaceLocking Live Dashboard</h1>
    <span id="liveBadge" class="badge off">Connecting</span>
  </header>
  <main>
    <div class="grid">
      <div class="card">
        <h2>System</h2>
        <div id="systemState" class="big state-IDLE">IDLE</div>
        <div class="sub">Locked: <strong id="lockedSpeaker">—</strong></div>
        <div class="sub">Uptime: <strong id="uptime">0s</strong></div>
      </div>
      <div class="card">
        <h2>Servo angle</h2>
        <div class="big"><span id="servoAngle">90</span>°</div>
        <div class="sub">Target: <strong id="servoTarget">90</strong>° ·
          <strong id="servoMoving">idle</strong></div>
        <div class="gauge-wrap">
          <div class="gauge">
            <div id="gaugeFill" class="gauge-fill"></div>
            <div id="gaugeTarget" class="gauge-target"></div>
          </div>
          <div class="gauge-labels"><span>0°</span><span>90°</span><span>180°</span></div>
        </div>
      </div>
      <div class="card">
        <h2>Camera / face</h2>
        <div class="big"><span id="faceCount">0</span> <span style="font-size:1rem;font-weight:500;color:var(--muted)">faces</span></div>
        <div class="sub">Confidence: <strong id="confidence">0.00</strong></div>
        <div class="face-bar">
          <div class="face-center"></div>
          <div id="faceDot" class="face-dot" style="left:50%"></div>
        </div>
        <div class="sub">Face X: <strong id="faceX">—</strong> · Error: <strong id="faceError">—</strong></div>
      </div>
      <div class="card">
        <h2>Performance</h2>
        <div class="row"><span>Track FPS</span><span id="trackFps">0.0</span></div>
        <div class="row"><span>Recog FPS</span><span id="recogFps">0.0</span></div>
        <div class="row"><span>Frame #</span><span id="frameNum">0</span></div>
        <div class="row"><span>Motor cmd</span><span id="motorCmd">STOP</span></div>
      </div>
      <div class="card">
        <h2>MQTT</h2>
        <div class="big" id="mqttStatus" style="font-size:1.4rem">—</div>
        <div class="sub" id="mqttBroker">—</div>
        <div class="row" style="margin-top:0.75rem"><span>Match threshold</span><span id="threshold">0.45</span></div>
        <div class="row"><span>Lost for</span><span id="lostFor">0.0s</span></div>
      </div>
    </div>
    <div class="card">
      <h2>Servo angle history (last ~2 min)</h2>
      <canvas id="chart" width="900" height="140"></canvas>
    </div>
  </main>
  <script>
    const $ = (id) => document.getElementById(id);
    const canvas = $("chart");
    const ctx = canvas.getContext("2d");

    function pct(v, min, max) {
      return Math.max(0, Math.min(100, ((v - min) / (max - min)) * 100));
    }

    function drawChart(history) {
      const w = canvas.width, h = canvas.height;
      ctx.clearRect(0, 0, w, h);
      ctx.strokeStyle = "#2d3a4f";
      ctx.lineWidth = 1;
      for (let i = 0; i <= 4; i++) {
        const y = (h / 4) * i;
        ctx.beginPath(); ctx.moveTo(0, y); ctx.lineTo(w, y); ctx.stroke();
      }
      if (!history || history.length < 2) return;
      ctx.strokeStyle = "#38bdf8";
      ctx.lineWidth = 2;
      ctx.beginPath();
      history.forEach((v, i) => {
        const x = (i / (history.length - 1)) * w;
        const y = h - (v / 180) * (h - 8) - 4;
        if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
      });
      ctx.stroke();
    }

    async function refresh() {
      try {
        const res = await fetch("/api/status");
        const d = await res.json();
        const badge = $("liveBadge");
        if (!d.running) {
          badge.textContent = "Stopped";
          badge.className = "badge off";
        } else if (d.stale) {
          badge.textContent = "Stale";
          badge.className = "badge stale";
        } else {
          badge.textContent = "Live";
          badge.className = "badge live";
        }

        const st = d.system_state || "IDLE";
        $("systemState").textContent = st;
        $("systemState").className = "big state-" + st;
        $("lockedSpeaker").textContent = d.locked_speaker || "(none)";
        $("uptime").textContent = (d.uptime_sec || 0) + "s";

        const angle = d.servo_angle ?? 90;
        const target = d.servo_target ?? angle;
        $("servoAngle").textContent = Math.round(angle);
        $("servoTarget").textContent = Math.round(target);
        $("servoMoving").textContent = d.servo_moving ? "moving" : "idle";
        $("gaugeFill").style.width = pct(angle, 0, 180) + "%";
        $("gaugeTarget").style.left = pct(target, 0, 180) + "%";

        $("faceCount").textContent = d.face_count ?? 0;
        $("confidence").textContent = (d.confidence ?? 0).toFixed(2);
        $("trackFps").textContent = (d.track_fps ?? 0).toFixed(1);
        $("recogFps").textContent = (d.recog_fps ?? 0).toFixed(1);
        $("frameNum").textContent = d.frame_number ?? 0;
        $("motorCmd").textContent = d.motor_command || "STOP";
        $("threshold").textContent = (d.threshold ?? 0.45).toFixed(2);
        $("lostFor").textContent = (d.lost_for_sec ?? 0).toFixed(1) + "s";

        $("mqttStatus").textContent = d.mqtt_connected ? "Connected" : "Offline";
        $("mqttStatus").style.color = d.mqtt_connected ? "#22c55e" : "#ef4444";
        $("mqttBroker").textContent = d.mqtt_broker || "—";

        if (d.face_center_x != null && d.frame_width > 0) {
          $("faceX").textContent = Math.round(d.face_center_x);
          const left = pct(d.face_center_x, 0, d.frame_width);
          $("faceDot").style.left = left + "%";
        } else {
          $("faceX").textContent = "—";
          $("faceDot").style.left = "50%";
        }
        $("faceError").textContent = d.face_error != null
          ? (d.face_error >= 0 ? "+" : "") + d.face_error.toFixed(2) : "—";

        drawChart(d.angle_history || []);
      } catch (e) {
        $("liveBadge").textContent = "Offline";
        $("liveBadge").className = "badge off";
      }
    }

    refresh();
    setInterval(refresh, 500);
  </script>
</body>
</html>
"""

_server: Optional[ThreadingHTTPServer] = None
_thread: Optional[threading.Thread] = None


class _DashboardHandler(BaseHTTPRequestHandler):
    def log_message(self, format: str, *args) -> None:
        return

    def do_GET(self) -> None:
        if self.path in ("/", "/index.html"):
            body = DASHBOARD_HTML.encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return

        if self.path == "/api/status":
            body = json.dumps(snapshot()).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return

        self.send_response(404)
        self.end_headers()


def start_dashboard(host: str = None, port: int = None) -> str:
    """Start background HTTP server. Returns URL."""
    global _server, _thread
    host = host or config.DASHBOARD_HOST
    port = port or config.DASHBOARD_PORT
    if _thread and _thread.is_alive():
        return f"http://{host}:{port}"

    _server = ThreadingHTTPServer((host, port), _DashboardHandler)
    _thread = threading.Thread(target=_server.serve_forever, daemon=True)
    _thread.start()
    return f"http://{host}:{port}"


def stop_dashboard() -> None:
    global _server, _thread
    if _server:
        _server.shutdown()
        _server = None
    _thread = None
