"""Camera or video file -> YOLO11-pose + ByteTrack -> judge -> annotated stream, race recording, evidence clips."""
import collections
import datetime
import json
import queue
import threading
import time
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

from judge import JOINTS, KP_CONF, Arena, Race, assign_lanes, block_mask, find_blocks

cv2.setNumThreads(-1)  # ultralytics turns OpenCV threading off at import (for DataLoader workers we never use)

LANE_BGR = [(68, 68, 239), (246, 130, 59), (21, 204, 250)]  # merah, biru, kuning; same colours as web/app.js
JOINT_BGR = {"aman": (94, 197, 34), "kandidat": (11, 158, 245), "kontak": (68, 68, 239)}
SKELETON = [(5, 7), (7, 9), (6, 8), (8, 10), (5, 6), (5, 11), (6, 12), (11, 12), (11, 13), (13, 15), (12, 14), (14, 16)]
STREAM_W = 1280                     # browser stream width; judging always uses the full-resolution frame
CLIP_BEFORE, CLIP_AFTER = 2.0, 1.0  # seconds of evidence kept around each event
COUNTDOWN_S = 3.0


def pt(p, s=1.0):
    return round(float(p[0]) * s), round(float(p[1]) * s)


def draw_arena(img, arena, s=1.0):
    """Lane bands, START line, chequered FINIS line and lane badges. s scales full-res coordinates to img."""
    z, over, half = img.shape[0] / 720, img.copy(), 0.5 / arena.lanes
    w = max(2, round(3 * z))
    for i, v in enumerate(arena.v):
        quad = [arena.img(0, v - half), arena.img(1, v - half), arena.img(1, v + half), arena.img(0, v + half)]
        cv2.fillPoly(over, [np.int32([pt(q, s) for q in quad])], LANE_BGR[i], cv2.LINE_AA)
    cv2.addWeighted(over, 0.18, img, 0.82, 0, img)
    for i, (a, b) in enumerate(arena.lines):
        cv2.line(img, pt(a, s), pt(b, s), LANE_BGR[i], max(1, round(z)), cv2.LINE_AA)
    cv2.line(img, pt(arena.img(0, 0), s), pt(arena.img(0, 1), s), (255, 255, 255), w, cv2.LINE_AA)
    for k in range(10):
        cv2.line(img, pt(arena.img(1, k / 10), s), pt(arena.img(1, (k + 1) / 10), s),
                 (255, 255, 255) if k % 2 else (25, 25, 25), w + 1, cv2.LINE_AA)
    font, fs, ft = cv2.FONT_HERSHEY_DUPLEX, 0.55 * z, max(1, round(1.5 * z))
    for label, u in (("START", 0), ("FINIS", 1)):
        (tw, _), _ = cv2.getTextSize(label, font, fs, ft)
        x, y = pt(arena.img(u, 1), s)
        cv2.putText(img, label, (x - tw // 2, y - round(10 * z)), font, fs, (255, 255, 255), ft, cv2.LINE_AA)
    for i, v in enumerate(arena.v):
        c = pt(arena.img(-0.045, v), s)
        cv2.circle(img, c, round(13 * z), LANE_BGR[i], -1, cv2.LINE_AA)
        (tw, th), _ = cv2.getTextSize(str(i + 1), font, fs, ft)
        cv2.putText(img, str(i + 1), (c[0] - tw // 2, c[1] + th // 2), font, fs,
                    (20, 20, 20) if i == 2 else (255, 255, 255), ft, cv2.LINE_AA)


def draw_person(img, k, s, color, joints):
    """Skeleton in the lane colour (thinner grey when outside every lane) with the judged joints marked."""
    z, active = img.shape[0] / 720, color is not None
    color, th = color or (200, 200, 200), max(1, round((3 if active else 2) * z))
    for a, b in SKELETON:
        if k[a, 2] >= KP_CONF and k[b, 2] >= KP_CONF:
            cv2.line(img, pt(k[a], s), pt(k[b], s), color, th, cv2.LINE_AA)
    if k[0, 2] >= KP_CONF:
        cv2.circle(img, pt(k[0], s), round(6 * z), color, -1, cv2.LINE_AA)
    if active:
        for j in JOINTS:
            if k[j, 2] >= KP_CONF:
                st = joints.get(j)
                hot = st in ("kandidat", "kontak")
                cv2.circle(img, pt(k[j], s), round((8 if hot else 4) * z), JOINT_BGR.get(st, (255, 255, 255)),
                           -1 if hot else max(1, round(2 * z)), cv2.LINE_AA)


class Recorder:
    """Writes the raw race video on its own thread: mp4 encoding costs ~8 ms per 1080p frame."""

    def __init__(self, path, fps, size):
        self.w = cv2.VideoWriter(str(path), cv2.VideoWriter.fourcc(*"mp4v"), fps, size)
        self.q = queue.Queue(maxsize=240)  # a full queue blocks the loop rather than dropping evidence
        self.thread = threading.Thread(target=self.run, daemon=True)
        self.thread.start()

    def run(self):
        while (frame := self.q.get()) is not None:
            self.w.write(frame)
        self.w.release()

    def write(self, frame):
        self.q.put(frame)

    def release(self):
        self.q.put(None)
        self.thread.join()


class Pipeline:
    def __init__(self, source, model, imgsz, root):
        self.source, self.model_name, self.imgsz = str(source), model, imgsz
        self.is_file = not self.source.isdigit()
        self.root = Path(root)
        self.calib_path, self.runs_dir = self.root / "calibration.json", self.root / "runs"
        self.cap = self.open()
        fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.fps = fps if 1 <= fps <= 240 else 30.0
        self.size = (int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        self.model = YOLO(model)
        self.cv = threading.Condition()  # guards everything below; notified on every new frame
        self.jpeg = self.frame = self.frozen = self.draft = None
        self.calib = self.arena = self.race = None
        self.t, self.idx, self.rate, self.eof, self.restart = 0.0, 0, 0.0, False, False
        self.ring = collections.deque(maxlen=int(self.fps * (CLIP_BEFORE + CLIP_AFTER + 1)))
        self.pending, self.clips, self.race_id = [], set(), 0
        self.writer = self.run_dir = None
        if self.calib_path.exists():
            try:
                c = json.loads(self.calib_path.read_text(encoding="utf-8"))
                if tuple(c["frame_size"]) != self.size:
                    raise ValueError(f"dibuat untuk resolusi {c['frame_size']}, sumber sekarang {list(self.size)}")
                self.apply(c)
            except (ValueError, KeyError, TypeError) as e:
                print(f"calibration.json diabaikan ({e}); kalibrasi ulang di dashboard.")

    def open(self):
        if self.is_file:
            cap = cv2.VideoCapture(self.source)
        else:  # DirectShow + MJPG lets common USB webcams deliver 1080p at 30 fps on Windows
            cap = cv2.VideoCapture(int(self.source), cv2.CAP_DSHOW)
            cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter.fourcc(*"MJPG"))
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
            cap.set(cv2.CAP_PROP_FPS, 30)
        if not cap.isOpened():
            raise SystemExit(f"Sumber video tidak bisa dibuka: {self.source}")
        return cap

    def run(self):
        wall0, last = time.perf_counter(), None
        while True:
            if self.restart:  # file: play again from the start for a new race
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                self.idx, self.eof, self.restart = 0, False, False
                wall0 = time.perf_counter()
            if self.is_file and (self.eof or self.frozen is not None):
                time.sleep(0.05)
                wall0 = time.perf_counter() - self.idx / self.fps  # resume real-time pacing where we paused
                continue
            ok, frame = self.cap.read()
            if not ok:
                if self.is_file:
                    with self.cv:
                        self.eof = True
                        if self.race:
                            self.race.end()
                        self.after_update()
                else:
                    time.sleep(0.05)  # camera hiccup; keep trying
                continue
            if self.is_file:  # play files in real time so the dashboard behaves like a live race
                t = self.idx / self.fps
                time.sleep(max(0.0, wall0 + t - time.perf_counter()))
            else:
                t = time.perf_counter()
            self.idx += 1
            if self.frozen is not None:  # live camera during calibration: keep the buffer drained, skip the work
                continue
            self.process(frame, t, self.idx - 1)
            now = time.perf_counter()
            if last:
                self.rate = 1 / (now - last) if not self.rate else 0.9 * self.rate + 0.1 / max(now - last, 1e-6)
            last = now

    def process(self, frame, t, idx):
        r = self.model.track(frame, persist=True, tracker="bytetrack.yaml", imgsz=self.imgsz, verbose=False)[0]
        people = []
        if r.boxes is not None and r.boxes.id is not None and r.keypoints is not None:
            people = list(zip(r.boxes.id.int().tolist(), r.boxes.xyxy.cpu().numpy(), r.keypoints.data.cpu().numpy()))
        with self.cv:
            obs = blocks = None
            if self.arena:
                blocks = find_blocks(frame, self.arena, self.calib["hsv"])
                obs = assign_lanes(self.arena, people)
                for e in self.race.update(t, idx, obs, blocks):
                    self.pending.append((e, self.race.t0 + e["t"] + CLIP_AFTER))
            jpg = cv2.imencode(".jpg", self.draw(frame, people, obs, blocks), [cv2.IMWRITE_JPEG_QUALITY, 80])[1]
            self.frame, self.t, self.jpeg = frame, t, jpg.tobytes()
            self.ring.append((idx, t, self.jpeg))
            if self.writer:
                self.writer.write(frame)
            self.after_update()
            self.cv.notify_all()

    def draw(self, frame, people, obs, blocks):
        """Overlay drawn on the downscaled stream image (cheaper than drawing at full resolution)."""
        s = min(1.0, STREAM_W / frame.shape[1])
        img = cv2.resize(frame, (round(frame.shape[1] * s), round(frame.shape[0] * s)), interpolation=cv2.INTER_AREA)
        if self.arena:
            draw_arena(img, self.arena, s)
            for i, boxes in enumerate(blocks):
                for x1, y1, x2, y2 in boxes:
                    cv2.rectangle(img, pt((x1, y1), s), pt((x2, y2), s), LANE_BGR[i], 2, cv2.LINE_AA)
        lane_of = {o[0]: i for i, o in enumerate(obs or []) if o}
        for tid, _, k in people:
            i = lane_of.get(tid)
            draw_person(img, k, s, None if i is None else LANE_BGR[i], {} if i is None else self.race.lanes[i]["joints"])
        return img

    def after_update(self):
        """Open/close the race recording with the race state; write evidence clips once their 'after' part exists."""
        state = self.race.state if self.race else "siap"
        if state in ("hitung", "berlaga") and self.run_dir is None:
            self.run_dir = self.runs_dir / datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            (self.run_dir / "clips").mkdir(parents=True, exist_ok=True)
            h, w = self.frame.shape[:2]
            self.writer = Recorder(self.run_dir / "race.mp4", self.fps, (w, h))
        if state == "selesai" and self.writer:
            self.writer.release()
            self.writer = None
            self.write_result()
        for item in [p for p in self.pending if self.t >= p[1] or self.eof]:
            self.pending.remove(item)
            self.save_clip(item[0])

    def save_clip(self, e):
        d = self.run_dir / "clips" / e["id"]
        d.mkdir(parents=True, exist_ok=True)
        t_ev, frames = self.race.t0 + e["t"], []
        for i, ts, jpg in self.ring:
            if t_ev - CLIP_BEFORE <= ts <= t_ev + CLIP_AFTER:
                name = f"{len(frames):04d}.jpg"
                (d / name).write_bytes(jpg)
                frames.append(dict(file=name, t=round(ts - self.race.t0, 3), frame=i))
        (d / "clip.json").write_text(json.dumps(dict(id=e["id"], frames=frames)), encoding="utf-8")
        self.clips.add(e["id"])

    def write_result(self):
        if self.run_dir and self.race:
            data = dict(source=self.source, model=self.model_name, fps=self.fps, calibration=self.calib,
                        **self.race.snapshot())
            (self.run_dir / "result.json").write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")

    def reset(self):
        """Close the current race (clips, recording, result.json) and get ready for the next one."""
        for e, _ in self.pending:
            self.save_clip(e)
        self.pending = []
        if self.writer:
            self.writer.release()
            self.writer = None
        self.write_result()
        self.run_dir, self.clips = None, set()
        self.ring.clear()  # a rewound file reuses timestamps; old frames must not leak into new clips
        if self.race:
            self.race.reset()
        if self.is_file:
            self.restart = True

    def close(self):
        with self.cv:
            self.reset()

    # ---- commands from the dashboard (all validated here: the browser is a trust boundary) ----

    def command(self, c):
        a = c.get("action")
        with self.cv:
            if a == "start":
                if not self.race:
                    raise ValueError("Kalibrasi arena dulu.")
                if self.frozen is not None:
                    raise ValueError("Selesaikan kalibrasi dulu.")
                if self.eof:
                    raise ValueError("Video sudah habis; tekan Reset untuk memutar ulang.")
                # a file starts on its next frame (its clock may just have been rewound); live gets a countdown
                self.race.begin(None if self.is_file else time.perf_counter() + COUNTDOWN_S)
                self.race_id += 1
            elif a == "end":
                if self.race:
                    self.race.end()
            elif a == "reset":
                self.reset()
            elif a == "review":
                if not self.race:
                    raise ValueError("Belum ada lomba.")
                self.race.review(str(c["id"]), c["decision"])
                self.write_result()
            elif a == "calib_begin":
                if self.race and self.race.state in ("hitung", "berlaga"):
                    raise ValueError("Tidak bisa kalibrasi saat lomba berjalan.")
                if self.frame is None:
                    raise ValueError("Belum ada gambar dari kamera.")
                self.frozen, self.draft = self.frame.copy(), dict(self.calib or {})
                return dict(calib=self.calib)
            elif a == "calib_draft":
                self.calib_draft(c)
            elif a == "calib_sample":
                return self.calib_sample(c)
            elif a == "calib_save":
                d = self.draft or {}
                if self.frozen is None or "pts" not in d or "hsv" not in d:
                    raise ValueError("Lengkapi 4 titik lantai dan warna balok dulu.")
                d["frame_size"] = list(self.frozen.shape[1::-1])
                self.reset()  # a new arena starts a new race
                self.apply(d)
                self.calib_path.write_text(json.dumps(d, indent=2), encoding="utf-8")
                self.frozen = self.draft = None
            elif a == "calib_cancel":
                self.frozen = self.draft = None
            else:
                raise ValueError("Perintah tidak dikenal.")
        return None

    def calib_draft(self, c):
        if self.frozen is None:
            raise ValueError("Mode kalibrasi belum aktif.")
        self.draft.pop("pts", None)  # "Ulangi titik", or a rejected click set, must not keep showing the old arena
        if not c.get("pts"):
            return
        h, w = self.frozen.shape[:2]
        pts = [[float(x), float(y)] for x, y in c["pts"]]
        if len(pts) != 4 or not all(0 <= x < w and 0 <= y < h for x, y in pts):
            raise ValueError("Butuh 4 titik di dalam gambar.")
        lanes, length, block_h = int(c["lanes"]), float(c["length_m"]), float(c["block_h_cm"])
        if lanes not in (1, 2, 3) or not 1 <= length <= 50 or not 2 <= block_h <= 50:
            raise ValueError("Isian di luar batas: lintasan 1-3, panjang 1-50 m, tinggi balok 2-50 cm.")
        Arena(pts, lanes, length, block_h)  # raises a readable message for bad clicks
        self.draft.update(pts=pts, lanes=lanes, length_m=length, block_h_cm=block_h)

    def calib_sample(self, c):
        if self.frozen is None:
            raise ValueError("Mode kalibrasi belum aktif.")
        x, y = int(c["x"]), int(c["y"])
        h, w = self.frozen.shape[:2]
        if not (0 <= x < w and 0 <= y < h):
            raise ValueError("Titik di luar gambar.")
        patch = self.frozen[max(0, y - 3):y + 4, max(0, x - 3):x + 4]
        hsv = [int(v) for v in np.median(cv2.cvtColor(patch, cv2.COLOR_BGR2HSV).reshape(-1, 3), axis=0)]
        if hsv[1] < 60 or hsv[2] < 50:
            raise ValueError("Warna terlalu pucat atau gelap; klik sisi balok yang berwarna mencolok.")
        self.draft["hsv"] = hsv
        b, g, r = (int(v) for v in np.median(patch.reshape(-1, 3), axis=0))
        return dict(hsv=hsv, rgb=[r, g, b])

    def apply(self, c):
        self.arena = Arena(c["pts"], c["lanes"], c["length_m"], c["block_h_cm"])
        self.calib, self.race = c, Race(self.arena, self.fps)

    # ---- read side for the web server ----

    def state(self):
        with self.cv:
            return json.dumps(dict(
                source=self.source, is_file=self.is_file, model=self.model_name, fps=round(self.rate, 1),
                frame=list(self.size), eof=self.eof, calibrating=self.frozen is not None, calib=self.calib,
                race=self.race.snapshot() if self.race else None, race_id=self.race_id,
                run=self.run_dir.name if self.run_dir else None, clips=sorted(self.clips)))

    def wait_frame(self, last):
        with self.cv:
            self.cv.wait_for(lambda: self.jpeg is not last, timeout=2)
            return self.jpeg

    def snapshot(self, draft):
        """Full-resolution still for calibration; with draft=True it previews lanes, lines and the block colour."""
        with self.cv:
            src = self.frozen if self.frozen is not None else self.frame
            if src is None:
                return None
            img, d = src.copy(), dict(self.draft) if draft and self.draft else {}
        arena = Arena(d["pts"], d["lanes"], d["length_m"], d["block_h_cm"]) if "pts" in d else None
        blocks = find_blocks(img, arena, d["hsv"]) if arena and "hsv" in d else []
        if "hsv" in d:
            m = block_mask(img, d["hsv"]) > 0
            img[m] = (img[m] * 0.35 + np.array([255, 0, 255]) * 0.65).astype(np.uint8)  # magenta = seen as block colour
        if arena:
            draw_arena(img, arena)
            for boxes in blocks:
                for x1, y1, x2, y2 in boxes:
                    cv2.rectangle(img, pt((x1, y1)), pt((x2, y2)), (255, 255, 255), max(2, round(img.shape[0] / 360)),
                                  cv2.LINE_AA)
        return cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 85])[1].tobytes()
