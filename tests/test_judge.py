"""Rule checks with synthetic keypoints. Run: python tests/test_judge.py (pytest also works)."""
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from judge import Arena, Race, assign_lanes, find_blocks  # noqa: E402

FPS = 30
# Side camera looking slightly down: near side lower and wider in the image. 2 lanes, 5 m, 10 cm blocks.
ARENA = Arena([(100, 600), (220, 300), (1180, 600), (1060, 300)], lanes=2, length_m=5, block_h_cm=10)
HEIGHTS = {**{j: 140 for j in range(9)}, 9: 80, 10: 80, 11: 90, 12: 90, 13: 50, 14: 50}  # cm above the floor


def at(lane, u, h_cm):
    """Image point h_cm above the floor at progress u on a lane's centre line."""
    gx, gy = ARENA.img(u, ARENA.v[lane])
    return gx, gy - h_cm * ARENA.px_per_cm(lane, gx)


def person(lane, u, ankle_h=18):
    """Keypoints of a player at progress u: ankles ankle_h above the floor (18 = standing on a block)."""
    k = np.zeros((17, 3))
    for j in range(17):
        k[j] = (*at(lane, u, HEIGHTS.get(j, ankle_h)), 0.9)
    return k


def block(lane, u, length_cm=25):
    """Image box of a block face lying on the floor, centred at progress u."""
    d = length_cm / 2 / 500
    (x1, y2), (x2, _), (_, y1) = at(lane, u - d, 0), at(lane, u + d, 0), at(lane, u, 10)
    return x1, y1, x2, y2


def race(frames):
    """frames: [(obs per lane, blocks per lane)] fed at 30 fps; GO at t = 0."""
    r = Race(ARENA, FPS)
    r.begin(0.0)
    for n, (obs, blocks) in enumerate(frames):
        r.update(n / FPS, n, obs, blocks)
    return r


def violations(r):
    return [e for e in r.events if e["type"] == "pelanggaran"]


def raises(fn):
    try:
        fn()
    except ValueError:
        return True
    return False


def test_valid_run_finishes_with_interpolated_time():
    step = 0.011
    r = race([([(1, person(0, n * step)), None], [[block(0, n * step)], []]) for n in range(100)])
    lane = r.snapshot()["lanes"][0]
    assert lane["status"] == "finis" and lane["rank"] == 1
    assert abs(lane["finish"] - 1 / step / FPS) < 1e-3  # crossed between frames 90 and 91
    assert not violations(r) and r.state == "selesai"


def test_touch_needs_three_frames():
    def run(touch_frames):
        frames = []
        for n in range(20):
            k = person(0, 0.3)
            if 5 <= n < 5 + touch_frames:  # right foot planted on the floor ahead of the block
                k[16] = (*at(0, 0.42, 8), 0.9)
            frames.append(([(1, k), None], [[block(0, 0.3)], []]))
        return race(frames)

    assert not violations(run(2))  # 67 ms
    r = run(3)  # 100 ms
    v = violations(r)
    assert len(v) == 1 and v[0]["joint"] == "kaki kanan" and abs(v[0]["t"] - 5 / FPS) < 1e-3  # times kept to ms
    assert r.snapshot()["lanes"][0]["status"] == "gugur" and r.snapshot()["lanes"][0]["rank"] is None


def test_hand_gripping_a_block_on_the_floor_is_safe():
    def run(with_gripped_block):
        frames = []
        for n in range(15):
            k = person(0, 0.5)
            k[10] = (*at(0, 0.6, 5), 0.9)  # right wrist at floor level
            boxes = [block(0, 0.5)] + ([block(0, 0.6)] if with_gripped_block else [])
            frames.append(([(1, k), None], [boxes, []]))
        return race(frames)

    assert not violations(run(True))
    assert violations(run(False))[0]["joint"] == "tangan kanan"  # same pose, hand on the bare floor


def test_fast_low_foot_is_a_step_not_a_touch():
    frames = []
    for n in range(12):
        k = person(0, 0.3)
        k[16] = (*at(0, 0.35 + n * 5 / 500, 8), 0.9)  # 5 cm per frame (150 cm/s) just above the floor
        frames.append(([(1, k), None], [[block(0, 0.3)], []]))
    assert not violations(race(frames))


def test_hidden_points_and_missing_players_are_not_judged():
    frames = []
    for n in range(15):
        k = person(0, 0.3)
        k[16] = (*at(0, 0.42, 8), 0.2)  # occluded foot on the floor
        frames.append(([(1, k) if n % 2 else None, None], [[block(0, 0.3)], []]))
    r = race(frames)
    assert not violations(r)
    hidden = person(0, 0.3)
    hidden[[15, 16], 2] = 0.1
    r.update(1.0, 30, [(1, hidden), None], [[], []])
    assert r.snapshot()["lanes"][0]["visible"] is False


def test_one_frame_glitch_past_the_line_is_not_a_finish():
    us = [0.5] * 5 + [1.2] + [0.5] * 5
    r = race([([(1, person(0, u)), None], [[], []]) for u in us])
    assert r.snapshot()["lanes"][0]["finish"] is None


def test_finishes_within_one_frame_tie():
    # lane 1 crosses at t = 1.000 s, lane 2 at t = 1.020 s: less than 1/30 s apart
    r = race([([(1, person(0, n / 30)), (2, person(1, n / 30.6))], [[], []]) for n in range(40)])
    assert r.standings() == {0: (1, True), 1: (1, True)}


def test_overturned_violation_restores_rank():
    frames = []
    for n in range(45):
        k0 = person(0, n / 30)  # faster player
        if 3 <= n < 8:
            k0[16] = (*at(0, 0.2, 8), 0.9)  # but planted a foot on the floor
        frames.append(([(1, k0), (2, person(1, n / 33))], [[], []]))
    r = race(frames)
    v = violations(r)
    assert len(v) == 1 and r.standings() == {1: (1, False)}
    r.review(v[0]["id"], "batal")
    assert r.standings() == {0: (1, False), 1: (2, False)}
    assert raises(lambda: r.review(v[0]["id"], "mungkin"))


def test_touch_before_the_line_counts_touch_after_it_does_not():
    def run(touch_from):
        frames = []
        for n in range(45):
            k = person(0, n / 30 + 0.005)  # front foot crosses on frame 30 (at ~0.995 s), confirmed on frame 31
            if n >= touch_from:
                k[15] = (*at(0, 0.9, 8), 0.9)  # back foot on the floor
            frames.append(([(1, k), (2, person(1, n / 60))], [[], []]))  # lane 2 keeps the race running
        return race(frames)

    r = run(29)  # touch starts before the line, reaches 100 ms after it
    v = violations(r)
    assert len(v) == 1 and v[0]["t"] < r.lanes[0]["finish"]
    assert not violations(run(30))  # starts on the crossing frame, while the finish awaits confirmation
    assert not violations(run(32))


def test_body_parts_landing_together_are_one_violation():
    frames = []
    for n in range(10):
        k = person(0, 0.3)
        if n >= 2:
            k[15] = (*at(0, 0.45, 8), 0.9)
            k[9] = (*at(0, 0.5, 5), 0.9)
        frames.append(([(1, k), None], [[block(0, 0.3)], []]))
    v = violations(race(frames))
    assert len(v) == 1 and "kaki kiri" in v[0]["joint"] and "tangan kiri" in v[0]["joint"]


def test_lane_assignment_by_feet_ignores_people_outside_the_arena():
    a, b = person(0, 0.4), person(1, 0.6)
    spectator = person(1, 0.5)
    spectator[:, 1] -= 400  # feet far behind the far lane
    box = lambda k: (k[:, 0].min(), k[:, 1].min(), k[:, 0].max(), k[:, 1].max())  # noqa: E731
    out = assign_lanes(ARENA, [(7, box(a), a), (9, box(spectator), spectator), (8, box(b), b)])
    assert out[0][0] == 7 and out[1][0] == 8


def test_find_blocks_keeps_lane_blocks_and_drops_clothing():
    img = np.full((720, 1280, 3), 90, np.uint8)
    green = (40, 200, 40)  # BGR
    for lane, u in ((0, 0.3), (1, 0.6)):
        x1, y1, x2, y2 = (int(round(c)) for c in block(lane, u))
        cv2.rectangle(img, (x1, y1), (x2, y2), green, -1)
    x, y = (int(c) for c in at(1, 0.85, 110))  # a green vest about 1 m above lane 2, clear of both blocks
    cv2.rectangle(img, (x - 60, y - 60), (x + 60, y + 60), green, -1)
    hsv = cv2.cvtColor(np.uint8([[green]]), cv2.COLOR_BGR2HSV)[0, 0]
    assert [len(b) for b in find_blocks(img, ARENA, hsv)] == [1, 1]


def test_file_start_begins_on_the_next_frame_whatever_the_clock():
    r = Race(ARENA, FPS)
    r.begin(None)  # a rewound video file: its clock says nothing about "now"
    assert r.snapshot()["countdown"] == 0.0
    r.update(42.0, 0, [None, None], [[], []])
    assert r.state == "berlaga" and r.t0 == 42.0


def test_arena_rejects_bad_clicks():
    assert raises(lambda: Arena([(220, 300), (100, 600), (1180, 600), (1060, 300)], 2, 5, 10))  # near/far swapped
    assert raises(lambda: Arena([(100, 600), (1060, 300), (1180, 600), (220, 300)], 2, 5, 10))  # crossed quad


if __name__ == "__main__":
    tests = [f for name, f in sorted(globals().items()) if name.startswith("test_")]
    for f in tests:
        f()
        print("ok ", f.__name__)
    print(f"{len(tests)} tests passed")
