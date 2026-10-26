"""RETRACTED premise, kept for the record: the world model does NOT hide the drift. The generated frames show the car
leaving the road; the 'drawn offset' comes from the lane-line readout, which saturates near half a lane. Do not use.

'The world model hides the drift' video (4:5, 1080x1350): a montage of steering-pull rollouts of the never-steer car.

Per clip (5 s, the 5 Hz generated frames flow-interpolated to 30 fps):
  camera   comma's world model's generated narrow-camera frames, with the REAL road (the human's path, placed where it
           really is relative to the car's true pose) drawn on top; when it leaves the frame an edge arrow points to it
  counters real lateral distance from the road (physics) vs the offset the world model draws (comma's lane-line head on
           the generated frame, minus the same readout on the real drive: measure/calibrate_forgiveness.py)
  bottom   chase-cam top-down view (true scale) and a graph of both numbers over time

  python make_hidden.py OUT.mp4 FRAMES_DIR results_pull_fixed.jsonl forgiveness_pull.jsonl calib_pull.jsonl SEG:START:KICK ...
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from render import (AMBER, BG, CAM_HEIGHT, FPS_OUT, FPS_SRC, GREEN, INK, MUTED, PANEL, RED, Canvas, Rollout, Writer,  # noqa: E402
                    ego_to_device, project, upscale)

W, H = 1080, 1350
CYAN = (64, 224, 255)
LANE_HALF = 1.8  # m, half a lane


def series(meas: dict, calib: dict) -> tuple[np.ndarray, np.ndarray]:
    """Per generated frame k: |real lateral offset| (m) and |drawn offset| (m), the latter lightly smoothed."""
    real = np.abs(np.asarray(meas["physics_offset_m"], float))
    drawn = np.array([np.nan if a is None or b is None else abs(a - b)
                      for a, b in zip(meas["rendered_offset_m"], calib["real_offset_m"])], float)
    first = np.flatnonzero(np.isfinite(drawn))
    drawn[: first[0]] = 0.0  # lane readout warm-up frames: nothing drawn yet
    idx = np.arange(len(drawn))
    ok = np.isfinite(drawn)
    drawn = np.interp(idx, idx[ok], drawn[ok])
    drawn = np.convolve(np.pad(drawn, 1, mode="edge"), np.ones(3) / 3, mode="valid")  # 3-frame smoothing (0.6 s)
    return real, drawn


def at(arr: np.ndarray, t: float) -> float:
    return float(np.interp(t, np.arange(len(arr)), arr))


def road_overlay(cam: np.ndarray, r: Rollout, t: float) -> tuple[np.ndarray, int]:
    """Draw the real road (human path, lane-wide band) onto the upscaled camera. Returns the image and the side the
    road has left the frame to (-1 left, +1 right, 0 visible)."""
    scale = cam.shape[1] / 256.0
    ss = 2
    layer = np.zeros((cam.shape[0] * ss, cam.shape[1] * ss, 4), np.uint8)
    hp = r.human_ahead(t)
    hp = hp[(hp[:, 0] > 3.0) & (hp[:, 0] < 80)]
    side = 0
    if len(hp) > 1:
        d = np.gradient(hp, axis=0)
        d /= np.linalg.norm(d, axis=1, keepdims=True) + 1e-9
        nrm = np.stack([-d[:, 1], d[:, 0]], axis=1)
        L = project(ego_to_device(hp - nrm * LANE_HALF))
        R = project(ego_to_device(hp + nrm * LANE_HALF))
        C = project(ego_to_device(hp))
        ok = np.isfinite(L).all(1) & np.isfinite(R).all(1)
        if ok.sum() > 1:
            pl = lambda p: np.round(p * scale * ss).astype(np.int32)  # noqa: E731
            poly = np.vstack([L[ok], R[ok][::-1]])
            cv2.fillPoly(layer, [pl(poly)], (*CYAN, 70), cv2.LINE_AA)
            for edge in (L[ok], R[ok]):
                cv2.polylines(layer, [pl(edge)], False, (*CYAN, 255), 3 * ss, cv2.LINE_AA)
            for p in C[ok][::2]:
                cv2.circle(layer, tuple(pl(p)), 3 * ss, (*INK, 230), -1, cv2.LINE_AA)
        # the road point 25 m ahead (or the nearest available): off-frame -> arrow
        near = C[np.argmin(np.abs(hp[:, 0] - 25.0))]
        if not np.isfinite(near).all() or near[0] < 0:
            side = -1
        elif near[0] > 256:
            side = 1
    else:
        side = -1 if r.human_ahead(t)[-1, 1] < 0 else 1
    layer = cv2.resize(layer, (cam.shape[1], cam.shape[0]), interpolation=cv2.INTER_AREA)
    a = layer[..., 3:4].astype(np.float32) / 255.0
    return (cam.astype(np.float32) * (1 - a) + layer[..., :3].astype(np.float32) * a).astype(np.uint8), side


def chase_bev(r: Rollout, t: float, w: int, h: int) -> np.ndarray:
    """Top-down chase view at true scale: the road (lane band around the human path) and the car."""
    img = np.full((h, w, 3), PANEL, np.uint8)
    pos, head, _ = r.pose(t)
    span_y = 34.0  # metres shown across
    px_m = w / span_y
    ref = np.interp(pos[0], r.human[:, 0], r.human[:, 1])  # road centre beside the car

    def to_px(p):
        p = np.atleast_2d(p)
        u = w / 2 + (p[:, 1] - (ref + pos[1]) / 2) * px_m
        v = h * 0.72 - (p[:, 0] - pos[0]) * px_m
        return np.stack([u, v], 1).astype(np.int32)

    hum = r.human
    d = np.gradient(hum, axis=0)
    d /= np.linalg.norm(d, axis=1, keepdims=True) + 1e-9
    nrm = np.stack([-d[:, 1], d[:, 0]], axis=1)
    for k, col in ((3 * LANE_HALF, (44, 48, 56)),):
        poly = np.vstack([to_px(hum - nrm * k), to_px(hum + nrm * k)[::-1]])
        cv2.fillPoly(img, [poly], col, cv2.LINE_AA)
    poly = np.vstack([to_px(hum - nrm * LANE_HALF), to_px(hum + nrm * LANE_HALF)[::-1]])
    cv2.fillPoly(img, [poly], (30, 70, 86), cv2.LINE_AA)
    for s in (-1, 1):
        cv2.polylines(img, [to_px(hum + s * nrm * LANE_HALF)], False, CYAN, 2, cv2.LINE_AA)
    i = int(np.clip(np.floor(t + 1), 0, len(r.pts) - 1))
    cv2.polylines(img, [to_px(np.vstack([r.pts[: i + 1], pos[None]]))], False, RED, 4, cv2.LINE_AA)
    # car body 4.6 x 1.9 m
    c, s = np.cos(head), np.sin(head)
    body = np.array([[2.3, -0.95], [2.3, 0.95], [-2.3, 0.95], [-2.3, -0.95]]) @ np.array([[c, s], [-s, c]]) + pos
    cv2.fillPoly(img, [to_px(body)], RED, cv2.LINE_AA)
    cv2.polylines(img, [to_px(body)], True, INK, 2, cv2.LINE_AA)
    return img


def chart(real: np.ndarray, drawn: np.ndarray, t: float, w: int, h: int, ymax: float) -> np.ndarray:
    img = np.full((h, w, 3), PANEL, np.uint8)
    pad_l, pad_r, pad_t, pad_b = 16, 16, 18, 18
    n = len(real) - 1

    def to_px(x, y):
        return (int(pad_l + x / n * (w - pad_l - pad_r)), int(h - pad_b - y / ymax * (h - pad_t - pad_b)))

    for gy in range(0, int(ymax) + 1, 5):
        cv2.line(img, to_px(0, gy), to_px(n, gy), (44, 48, 56), 1, cv2.LINE_AA)
    ts = np.linspace(0, t, max(2, int(t * 12) + 2))
    for arr, col in ((real, RED), (drawn, GREEN)):
        pts = np.array([to_px(x, at(arr, x)) for x in ts], np.int32)
        cv2.polylines(img, [pts], False, col, 5, cv2.LINE_AA)
        cv2.circle(img, tuple(pts[-1]), 8, col, -1, cv2.LINE_AA)
    return img


def frame(r: Rollout, real, drawn, t: float, ymax: float, header: str) -> np.ndarray:
    cv_ = Canvas(W, H, BG)
    cv_.text((W // 2, 34), header, 30, INK, anchor="mt")
    cv_.text((W // 2, 76), "everything below is generated by comma's world model", 24, MUTED, bold=False, anchor="mt")
    cam = upscale(r.image(t)[..., :3], W)
    cam, side = road_overlay(cam, r, t)
    y0 = 118
    cv_.paste(cam, (0, y0))
    ch = cam.shape[0]
    d = cv_.d
    # tags on the camera
    cv_.rrect((20, y0 + 16, 300, y0 + 62), 10, fill=(0, 0, 0, 150))
    cv_.text((34, y0 + 24), "what the model shows", 26, INK)
    if side:
        x = 24 if side < 0 else W - 24
        ym = y0 + ch // 2
        bx = x - 46 * side  # arrow tip at the frame edge, base 46 px inside
        tri = [(x, ym), (bx, ym - 34), (bx, ym + 34)]
        d.polygon(tri, fill=(*CYAN, 255))
        cv_.text((x + 60 if side < 0 else x - 60, ym), "real road", 34, CYAN, anchor="lm" if side < 0 else "rm")
    else:
        cv_.rrect((W - 230, y0 + 16, W - 20, y0 + 62), 10, fill=(0, 0, 0, 150))
        cv_.text((W - 206, y0 + 24), "real road", 26, CYAN)
    # counters
    yc = y0 + ch + 22
    rv, dv = at(real, t), at(drawn, t)
    cv_.rrect((20, yc, W // 2 - 10, yc + 190), 18, fill=PANEL)
    cv_.rrect((W // 2 + 10, yc, W - 20, yc + 190), 18, fill=PANEL)
    cv_.text((44, yc + 22), "car really is", 28, MUTED, bold=False)
    cv_.text((44, yc + 58), f"{rv:4.1f} m", 96, RED)
    cv_.text((44, yc + 156), "off the road", 26, MUTED, bold=False)
    cv_.text((W // 2 + 34, yc + 22), "world model draws", 28, MUTED, bold=False)
    cv_.text((W // 2 + 34, yc + 58), f"{dv:4.1f} m", 96, GREEN)
    cv_.text((W // 2 + 34, yc + 156), "off the road", 26, MUTED, bold=False)
    # bottom: chase view + chart
    yb = yc + 210
    hb = H - yb - 54
    cv_.paste(chase_bev(r, t, W // 2 - 30, hb), (20, yb))
    cv_.paste(chart(real, drawn, t, W // 2 - 30, hb, ymax), (W // 2 + 10, yb))
    cv_.text((34, yb + 14), "top-down, true scale", 22, MUTED, bold=False)
    cv_.text((W // 2 + 24, yb + 14), "metres off the road", 22, MUTED, bold=False)
    cv_.text((W // 2, H - 22), "5 Hz generated frames, interpolated to 30 fps · drawn offset read by comma's lane detector",
             19, MUTED, bold=False, anchor="ms")
    return cv_.array()


def main() -> None:
    out, frames_dir, res_p, meas_p, calib_p = sys.argv[1], Path(sys.argv[2]), sys.argv[3], sys.argv[4], sys.argv[5]
    picks = [s.split(":") for s in sys.argv[6:]]
    rows = {(r["segment"], r["start"], r["policy"], r["kick"]): r for r in map(json.loads, open(res_p))}
    meas = {(m["segment"], m["start"], m["policy"], m["kick"]): m for m in map(json.loads, open(meas_p))}
    calib = {(c["segment"], c["start"]): c for c in map(json.loads, open(calib_p))}
    clips = []
    for seg, start, kick in picks:
        key = (seg, int(start), "straight", kick)
        r = Rollout(rows[key], frames_dir / f"{seg}_{start}_straight_{kick}.npz")
        real, drawn = series(meas[key], calib[(seg, int(start))])
        side = "right" if rows[key]["pull_dps"] > 0 else "left"
        clips.append((r, real, drawn, f"nobody steering · steering pulled to the {side}"))
    ymax = max(c[1].max() for c in clips) * 1.08
    wr = Writer(Path(out), W, H)
    prev = None
    for r, real, drawn, header in clips:
        n = r.n - 1
        steps = int(round(n / FPS_SRC * FPS_OUT))
        for i in range(steps + 1):
            f = frame(r, real, drawn, i * n / steps, ymax, header)
            if prev is not None and i < 8:  # quick crossfade between clips
                a = (i + 1) / 9
                f = (prev.astype(np.float32) * (1 - a) + f.astype(np.float32) * a).astype(np.uint8)
            wr.write(f)
        last = f
        wr.hold(last, 0.8)
        prev = last
    wr.close()
    print(f"wrote {out}: {wr.n} frames, {wr.n / FPS_OUT:.1f} s")


if __name__ == "__main__":
    main()
