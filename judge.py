"""Arena geometry and the Lari Balok judging rules.

Pure logic on keypoints and boxes, so it can be tested without a camera or a model.
Image pixels: x right, y down. Arena coords: u = 0 at START .. 1 at FINIS, v = 0 near side .. 1 far side.
Lane index 0 is "Lintasan 1", the lane farthest from the camera (as in the slide's top view).
"""
import cv2
import numpy as np

# COCO-17 keypoint -> (label, height of the keypoint above the surface it rests on, cm)
JOINTS = {
    9: ("tangan kiri", 5), 10: ("tangan kanan", 5),
    11: ("pinggul kiri", 10), 12: ("pinggul kanan", 10),
    13: ("lutut kiri", 7), 14: ("lutut kanan", 7),
    15: ("kaki kiri", 8), 16: ("kaki kanan", 8),
}
ANKLES, WRISTS = (15, 16), (9, 10)
KP_CONF = 0.5           # weaker keypoints count as hidden and are not judged
MIN_CONTACT_S = 0.10    # a touch must last >= 100 ms (3 frames at 30 fps), as in the slide
STILL_CM = 6            # a touching point stays within this radius; a stepping foot leaves it
BLOCK_MARGIN_CM = 8     # heel or fingers may overhang a block's edge
BLOCK_CM2 = (40, 1500)  # plausible visible area of a block face; rejects specks and same-coloured clothing
FINISH_FRAMES = 2       # frames past the line before a finish counts (guards against one-frame keypoint glitches)
MERGE_S = 1.0           # touches of one lane closer than this in time form a single violation


class Arena:
    """Floor model from 4 clicked points: START near, START far, FINIS near, FINIS far."""

    def __init__(self, pts, lanes, length_m, block_h_cm):
        p = np.float32(pts)
        if p.shape != (4, 2):
            raise ValueError("Butuh tepat 4 titik lantai.")
        if not (p[0, 1] > p[1, 1] and p[2, 1] > p[3, 1]):
            raise ValueError("Titik DEKAT harus lebih bawah di gambar daripada titik JAUH.")
        if not cv2.isContourConvex(p[[0, 2, 3, 1]].astype(np.int32)):
            raise ValueError("Keempat titik tidak membentuk segi empat; ulangi sesuai urutan.")
        self.H = cv2.getPerspectiveTransform(p, np.float32([[0, 0], [0, 1], [1, 0], [1, 1]]))
        self.Hinv = np.linalg.inv(self.H)
        self.lanes, self.length_cm, self.block_h_cm = int(lanes), float(length_m) * 100, float(block_h_cm)
        self.v = [1 - (i + 0.5) / self.lanes for i in range(self.lanes)]
        self.lines = [(self.img(0, v), self.img(1, v)) for v in self.v]
        if min(abs(b[0] - a[0]) for a, b in self.lines) < 50:
            raise ValueError("START dan FINIS terlalu berdekatan di gambar; kamera harus menghadap sisi lintasan.")

    def img(self, u, v):
        x, y, w = self.Hinv @ (u, v, 1.0)
        return np.array([x / w, y / w])

    def arena(self, x, y):
        u, v, w = self.H @ (x, y, 1.0)
        return u / w, v / w

    def ground_y(self, i, x):
        """Image y of lane i's floor centre line at column x."""
        (x0, y0), (x1, y1) = self.lines[i]
        return y0 + (x - x0) * (y1 - y0) / (x1 - x0)

    def u_at(self, i, x):
        """Progress (0 START .. 1 FINIS) of lane i's floor point below image column x."""
        return self.arena(x, self.ground_y(i, x))[0]

    def scale(self, u, v):
        """Image pixels per cm along the track at arena point (u, v); also used as the vertical scale."""
        d = 0.01
        return float(np.linalg.norm(self.img(u + d, v) - self.img(u - d, v))) / (2 * d * self.length_cm)

    def px_per_cm(self, i, x):
        return self.scale(self.u_at(i, x), self.v[i])

    def locate(self, x, y, lift_cm):
        """(lane, distance from the lane centre in lane widths) of a point lift_cm above the floor; None outside."""
        with np.errstate(all="ignore"):
            u, v = self.arena(x, y)
            u, v = self.arena(x, y + lift_cm * self.scale(u, v))
        if not (-0.25 / self.lanes <= v <= 1 + 0.25 / self.lanes and -0.3 <= u <= 1.3):
            return None
        i = min(self.lanes - 1, max(0, int((1 - v) * self.lanes)))
        return i, abs(v - self.v[i]) * self.lanes


def assign_lanes(arena, people):
    """people: [(track_id, box_xyxy, kpts 17x3)] -> per lane (track_id, kpts) or None.
    Identity = lane: a player never leaves their lane, so a ByteTrack ID switch cannot move a result."""
    out, best = [None] * arena.lanes, [9.0] * arena.lanes
    for tid, box, k in people:
        feet = [k[j] for j in ANKLES if k[j, 2] >= KP_CONF]
        if feet:  # lowest ankle = the supporting foot; it stands on the floor or on a block, so assume half a block up
            x, y = max(feet, key=lambda p: p[1])[:2]
            hit = arena.locate(x, y, arena.block_h_cm / 2 + JOINTS[15][1])
        else:
            hit = arena.locate((box[0] + box[2]) / 2, box[3], arena.block_h_cm / 2)
        if hit and hit[1] < best[hit[0]]:
            out[hit[0]], best[hit[0]] = (tid, k), hit[1]
    return out


def block_mask(bgr, hsv, tol_h=12):
    """Pixels matching the sampled block colour (OpenCV HSV; hue wraps around at 180)."""
    h, s, v = (int(c) for c in hsv)
    img = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    lo_s, lo_v = max(40, int(s * 0.45)), max(40, int(v * 0.35))
    m = np.zeros(img.shape[:2], np.uint8)
    for a, b in ((h - tol_h, h + tol_h), (h - tol_h + 180, h + tol_h + 180), (h - tol_h - 180, h + tol_h - 180)):
        if max(a, 0) <= min(b, 179):
            m |= cv2.inRange(img, (max(a, 0), lo_s, lo_v), (min(b, 179), 255, 255))
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    return cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))


def find_blocks(bgr, arena, hsv, scale=0.5):
    """Coloured block faces whose bottom rests on a lane -> per lane [(x1, y1, x2, y2)] in full-res pixels.
    Blocks lifted in the air are dropped, which is fine: a hand carrying one is off the floor too."""
    small = cv2.resize(bgr, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    stats = cv2.connectedComponentsWithStats(block_mask(small, hsv))[2][1:]
    lanes = [[] for _ in range(arena.lanes)]
    for x, y, w, h, area in stats[stats[:, 4] >= 8]:  # drop specks before the per-blob geometry
        x1, y1, x2, y2 = x / scale, y / scale, (x + w) / scale, (y + h) / scale
        hit = arena.locate((x1 + x2) / 2, y2, 0)
        if hit is None:
            continue
        cm2 = area / scale ** 2 / arena.px_per_cm(hit[0], (x1 + x2) / 2) ** 2
        if BLOCK_CM2[0] <= cm2 <= BLOCK_CM2[1]:
            lanes[hit[0]].append((x1, y1, x2, y2))
    return lanes


class Race:
    """Per-lane judging: siap -> hitung (countdown) -> berlaga -> selesai. Event times are seconds after GO."""

    def __init__(self, arena, fps):
        self.arena, self.dt = arena, 1.0 / fps
        self.reset()

    def reset(self):
        self.state, self.go_at, self.t0, self.t_end, self.now = "siap", None, None, None, 0.0
        self.events = []
        self.lanes = [dict(present=False, seen=False, visible=False, progress=0.0, finish=None, finish_frame=None,
                           cross=None, prev=None, streaks={}, joints={}, tracks=[]) for _ in range(self.arena.lanes)]

    def begin(self, go_at=None):
        """Arm the race: GO when the source clock reaches go_at (live countdown), or on the next frame if None."""
        if self.state != "siap":
            raise ValueError("Lomba sudah dimulai; tekan Reset untuk lomba baru.")
        self.state, self.go_at = "hitung", go_at

    def end(self):
        if self.state in ("hitung", "berlaga"):
            self.state, self.t_end = "selesai", self.now

    def out(self, i):
        """Lane i is disqualified while it has a violation the referee has not overturned."""
        return any(e["lane"] == i and e["type"] == "pelanggaran" and e["review"] != "batal" for e in self.events)

    def review(self, eid, decision):
        if decision not in ("sah", "batal"):
            raise ValueError("Keputusan harus 'sah' atau 'batal'.")
        for e in self.events:
            if e["id"] == eid and e["type"] == "pelanggaran":
                e["review"] = decision
                return
        raise ValueError("Pelanggaran tidak ditemukan.")

    def update(self, t, frame, obs, blocks):
        """One frame. t: source clock (s); obs: per lane (track_id, kpts) or None; blocks: per lane boxes.
        Returns the events created by this frame."""
        self.now = t
        if self.state == "hitung" and (self.go_at is None or t >= self.go_at):
            self.state, self.t0 = "berlaga", t if self.go_at is None else self.go_at
        judging = self.state == "berlaga"
        rt = t - self.t0 if judging else 0.0
        new = []
        for i, (lane, o) in enumerate(zip(self.lanes, obs)):
            lane["present"], lane["joints"] = o is not None, {}
            feet = [] if o is None else [o[1][j] for j in ANKLES if o[1][j, 2] >= KP_CONF]
            lane["visible"] = bool(feet)
            u = max((self.arena.u_at(i, p[0]) for p in feet), default=None)  # front foot = first ankle at the line
            if u is not None and lane["finish"] is None:
                lane["progress"] = float(min(max(u, 0.0), 1.0))
            if not judging:
                continue
            if o is None:  # player not detected: nothing is judged, and a touch must be seen again from scratch
                lane["streaks"].clear()
                continue
            lane["seen"] = True
            if o[0] not in lane["tracks"]:
                lane["tracks"].append(int(o[0]))
            if u is not None and lane["finish"] is None:
                new += self._finish(i, lane, u, rt, frame)
            new += self._contacts(i, lane, o[1], blocks[i], rt, frame)
        seen = [lane for lane in self.lanes if lane["seen"]]
        if judging and seen and all(lane["finish"] is not None for lane in seen):
            self.state, self.t_end = "selesai", t
        return new

    def _finish(self, i, lane, u, rt, frame):
        prev, lane["prev"] = lane["prev"], (rt, u)
        if u < 1:
            lane["cross"] = None
            return []
        if lane["cross"] is None:  # first frame past the line: interpolate the crossing instant between frames
            tc = prev[0] + (1 - prev[1]) / (u - prev[1]) * (rt - prev[0]) if prev and prev[1] < 1 else rt
            lane["cross"] = [tc, frame, 0]
        lane["cross"][2] += 1
        if lane["cross"][2] < FINISH_FRAMES:
            return []
        lane["finish"], lane["finish_frame"], lane["progress"] = lane["cross"][0], lane["cross"][1], 1.0
        return [self._event(i, "finis", lane["finish"], lane["finish_frame"])]

    def _contacts(self, i, lane, k, boxes, rt, frame):
        a, new = self.arena, []
        # the line counts from the first frame past it, even while that finish still awaits confirmation
        fin = lane["finish"] if lane["finish"] is not None else (lane["cross"] or [None])[0]
        for j, (label, lift) in JOINTS.items():
            x, y, c = k[j]
            st = lane["streaks"].get(j)
            # hidden points are not judged; after the line only a touch that began before it still counts
            if c < KP_CONF or (fin is not None and (st is None or st["t"] >= fin)):
                lane["streaks"].pop(j, None)
                continue
            s = a.px_per_cm(i, x)
            m = BLOCK_MARGIN_CM * s
            # feet/knees/hips must be above a block's top edge; a hand may grip the block's side
            on_block = any(b[0] - m <= x <= b[2] + m and y < (b[3] if j in WRISTS else b[1]) for b in boxes)
            low = (a.ground_y(i, x) - y) / s < a.block_h_cm / 2 + lift  # halfway between "on floor" and "on block"
            if on_block or not low:
                lane["streaks"].pop(j, None)
                lane["joints"][j] = "aman"
                continue
            if st is None or np.hypot(x - st["x"], y - st["y"]) > STILL_CM * s:  # new touch, or the point moved on
                if fin is not None:
                    lane["streaks"].pop(j, None)
                    continue
                st = lane["streaks"][j] = dict(t=rt, frame=frame, x=x, y=y, flagged=False)
            if not st["flagged"] and rt - st["t"] + self.dt >= MIN_CONTACT_S - 1e-9:
                st["flagged"] = True
                new += self._violation(i, st["t"], st["frame"], label)
            lane["joints"][j] = "kontak" if st["flagged"] else "kandidat"
        return new

    def _violation(self, i, t, frame, label):
        for e in self.events:  # several body parts landing together is one fall, not several violations
            if e["lane"] == i and e["type"] == "pelanggaran" and e["review"] != "batal" and abs(e["t"] - t) < MERGE_S:
                if label not in e["joint"]:
                    e["joint"] += ", " + label
                return []
        return [self._event(i, "pelanggaran", t, frame, label)]

    def _event(self, i, kind, t, frame, joint=None):
        e = dict(id=f"e{len(self.events) + 1}", lane=i, type=kind, t=round(float(t), 3), frame=int(frame),
                 joint=joint, review="menunggu" if kind == "pelanggaran" else None)
        self.events.append(e)
        return e

    def standings(self):
        """lane -> (rank, tie): finished and not disqualified, by finish time; < 1 frame apart is a tie (slide rule)."""
        done = sorted((lane["finish"], i) for i, lane in enumerate(self.lanes)
                      if lane["finish"] is not None and not self.out(i))
        ranks, rank, prev = {}, 0, None
        for n, (t, i) in enumerate(done, 1):
            if prev is None or t - prev >= self.dt - 1e-9:
                rank = n
            ranks[i], prev = rank, t
        counts = list(ranks.values())
        return {i: (r, counts.count(r) > 1) for i, r in ranks.items()}

    def snapshot(self):
        ranks, lanes = self.standings(), []
        for i, lane in enumerate(self.lanes):
            if self.out(i):
                status = "gugur"
            elif lane["finish"] is not None:
                status = "finis"
            elif self.state in ("siap", "hitung"):
                status = "siap" if lane["present"] else "kosong"
            elif not lane["seen"]:
                status = "kosong"
            else:
                status = "berlaga" if self.state == "berlaga" else "dnf"
            rank, tie = ranks.get(i, (None, False))
            lanes.append(dict(lane=i + 1, status=status, progress=round(lane["progress"], 3),
                              finish=None if lane["finish"] is None else round(float(lane["finish"]), 3),
                              finish_frame=lane["finish_frame"], rank=rank, tie=tie, visible=lane["visible"],
                              tracks=lane["tracks"]))
        if self.state == "berlaga":
            time = self.now - self.t0
        elif self.state == "selesai" and self.t0 is not None:
            time = self.t_end - self.t0
        else:
            time = 0.0
        return dict(state=self.state, time=round(time, 3), lanes=lanes, events=self.events,
                    countdown=round(max(0.0, (self.go_at or self.now) - self.now), 3) if self.state == "hitung" else None)
