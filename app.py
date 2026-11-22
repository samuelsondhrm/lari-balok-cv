"""Lari Balok CV: wasit bantu berbasis computer vision.

    python app.py --source 0            # kamera USB/webcam indeks 0
    python app.py --source rekaman.mp4  # file video
lalu buka http://localhost:8000 (sumber bisa diganti atau ditambah dari dashboard)
"""
import argparse
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from pipeline import Pipeline

ROOT = Path(__file__).resolve().parent
WEB = ROOT / "web"
TYPES = {".html": "text/html; charset=utf-8", ".js": "text/javascript; charset=utf-8",
         ".css": "text/css; charset=utf-8", ".json": "application/json", ".jpg": "image/jpeg"}


class Handler(BaseHTTPRequestHandler):
    pipe = None

    def log_message(self, *args):  # keep the console for pipeline messages
        pass

    def do_GET(self):
        path, _, query = self.path.partition("?")
        if path == "/stream":
            return self.stream()
        if path == "/api/state":
            return self.send(200, self.pipe.state().encode(), TYPES[".json"])
        if path == "/api/sources":
            return self.send(200, self.pipe.sources().encode(), TYPES[".json"])
        if path == "/api/snapshot":
            jpg = self.pipe.snapshot("draft=1" in query)
            return self.send(200, jpg, TYPES[".jpg"]) if jpg else self.send_error(503)
        if path.startswith("/runs/"):
            return self.file(self.pipe.runs_dir, path[len("/runs/"):])
        return self.file(WEB, path.lstrip("/") or "index.html")

    def do_POST(self):
        if self.path != "/api/cmd":
            return self.send_error(404)
        try:
            n = int(self.headers.get("Content-Length", 0))
            if not 0 <= n <= 65536:
                raise ValueError("Permintaan terlalu besar.")
            body, code = dict(ok=True, **(self.pipe.command(json.loads(self.rfile.read(n) or b"{}")) or {})), 200
        except (ValueError, KeyError, TypeError) as e:
            body, code = dict(ok=False, error=str(e) or "Permintaan tidak valid."), 400
        self.send(code, json.dumps(body).encode(), TYPES[".json"])

    def send(self, code, data, ctype):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(data)

    def file(self, base, rel):
        p = (base / rel).resolve()
        if not p.is_relative_to(base.resolve()) or not p.is_file():  # no ../ escapes
            return self.send_error(404)
        self.send(200, p.read_bytes(), TYPES.get(p.suffix, "application/octet-stream"))

    def stream(self):
        self.send_response(200)
        self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        jpg = None
        try:
            while True:
                jpg = self.pipe.wait_frame(jpg)
                if jpg:
                    self.wfile.write(b"--frame\r\nContent-Type: image/jpeg\r\nContent-Length: %d\r\n\r\n%s\r\n"
                                     % (len(jpg), jpg))
        except OSError:  # the browser closed the stream
            pass


def main():
    ap = argparse.ArgumentParser(description="Lari Balok CV: wasit bantu berbasis computer vision.")
    ap.add_argument("--source", default="0", help="sumber awal: indeks kamera, URL kamera HP/IP, atau path video")
    ap.add_argument("--model", default="yolo11s-pose.pt",
                    help="bobot YOLO11-pose; yolo11n-pose.pt untuk laptop tanpa GPU")
    ap.add_argument("--imgsz", type=int, default=1280,
                    help="ukuran input model; 1280 menangkap pemain kecil di gambar, 640 untuk laptop tanpa GPU")
    ap.add_argument("--port", type=int, default=8000)
    a = ap.parse_args()
    Handler.pipe = pipe = Pipeline(a.source, a.model, a.imgsz, ROOT)
    threading.Thread(target=pipe.run, daemon=True).start()
    server = ThreadingHTTPServer(("127.0.0.1", a.port), Handler)
    server.daemon_threads = True
    print(f"Dashboard: http://localhost:{a.port}   (Ctrl+C untuk berhenti)")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        pipe.close()


if __name__ == "__main__":
    main()
