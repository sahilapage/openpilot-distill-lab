"""Video renderer for the DAgger before/after showcase.

Inputs are exactly what eval_closed_loop.py writes: results.jsonl rows (policy_path, human_path, actions, metrics) and
one frames npz per (segment, start, policy) holding `frames` (T, H, W, 6: narrow RGB | wide RGB, uint8) and `plan`
(T, N, >=3: the world model's own plan in the device frame).

Conventions (same as the repo): path points are in the rollout's start ego frame, x forward, y RIGHT (openpilot's
calibrated frame, positive curvature = right turn), so they map straight onto the device frame (x forward, y right,
z down); the camera sits 1.22 m above the road. Rows written before 2026-10-26 hold a mirrored human path (y left);
Rollout mirrors those back (no "ref_frame" field). The narrow-camera projection is the repo's
(rl/env.py draw_worldmodel_outputs), re-implemented without openpilot and checked against it to 0.0 px.

The generated frames are 5 Hz, so playback blends neighbouring frames along optical flow and interpolates every
overlay, which is why the output is smooth at 30 fps. Say so in the post: it is interpolated, not generated at 30 fps.
"""
from __future__ import annotations

import json
import subprocess
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

FPS_SRC = 5
FPS_OUT = 30
CAM_HEIGHT = 1.22
K = np.array([[455.0, 0.0, 128.0], [0.0, 455.0, 23.8], [0.0, 0.0, 1.0]])  # narrow camera at 256x128

# palette
BG = (10, 12, 16)
PANEL = (20, 23, 29)
INK = (236, 239, 244)
MUTED = (140, 148, 160)
GREEN = (72, 224, 128)
ORANGE = (255, 160, 48)
RED = (255, 84, 84)
AMBER = (255, 196, 64)
BLUE = (96, 170, 255)

_FONTS: dict = {}


def font(size: int, bold: bool = True, mono: bool = False) -> ImageFont.FreeTypeFont:
    key = (size, bold, mono)
    if key not in _FONTS:
        if mono:
            _FONTS[key] = ImageFont.truetype("/System/Library/Fonts/Menlo.ttc", size, index=1 if bold else 0)
        else:
            _FONTS[key] = ImageFont.truetype("/System/Library/Fonts/Avenir Next.ttc", size, index=2 if bold else 7)
    return _FONTS[key]


# ----------------------------------------------------------------------------------------------- geometry
def project(device_pts: np.ndarray) -> np.ndarray:
    """Device-frame points (x fwd, y right, z down, camera at origin) -> narrow-camera pixels (256x128 space)."""
    v = np.stack([device_pts[:, 1], device_pts[:, 2], device_pts[:, 0]], axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        uv = v[:, :2] / v[:, 2:3]
    return (np.column_stack([uv, np.ones(len(uv))]) @ K.T)[:, :2]


def ego_to_device(pts: np.ndarray, height: float = CAM_HEIGHT) -> np.ndarray:
    """(n,2) ego path (x fwd, y right) on the road -> device frame points below the camera."""
    return np.column_stack([pts[:, 0], pts[:, 1], np.full(len(pts), height)])


def ribbon(center_left: np.ndarray, half_width: float = 0.9) -> tuple[np.ndarray, np.ndarray] | None:
    """The two edges of a road-width ribbon around an ego path (x fwd, y right), as camera pixels."""
    c = center_left[center_left[:, 0] > 1.5]
    if len(c) < 2:
        return None
    d = np.gradient(c, axis=0)
    d /= np.linalg.norm(d, axis=1, keepdims=True) + 1e-9
    normal = np.stack([-d[:, 1], d[:, 0]], axis=1)  # right normal (y right)
    return project(ego_to_device(c + normal * half_width)), project(ego_to_device(c - normal * half_width))


# ----------------------------------------------------------------------------------------------- rollout data
class Rollout:
    """One (segment, start, policy) rollout, queryable at any continuous time t (in 5 Hz frame units)."""

    def __init__(self, row: dict, npz_path: Path, name: str = ""):
        z = np.load(npz_path)
        self.frames = z["frames"]
        self.plan = z["plan"]
        self.name = name or row["policy"]
        self.row = row
        self.n = len(self.frames)
        pts = np.asarray(row["policy_path"], float)  # (T+1, 2)
        acts = np.asarray(row["actions"], float)
        chord = np.linalg.norm(np.diff(pts, axis=0), axis=1)  # = speed/5 exactly (rl.env.Physics)
        yaw = chord * acts[: len(chord), 0]
        self.pts = pts
        self.heading = np.concatenate([[0.0], np.cumsum(yaw)])
        self.speed = np.concatenate([[chord[0] * FPS_SRC], chord * FPS_SRC])
        self.human = np.asarray(row["human_path"], float)
        if row.get("ref_frame") != "y_right":  # old rows: human path was written mirrored (y left)
            self.human = self.human * np.array([1.0, -1.0])
        self.err = np.linalg.norm(pts - self.human, axis=1)  # drift vs the real human path at every step
        self.success = bool(row["success"])

    # frame k shows the world AFTER action k, i.e. pose index k+1
    def pose(self, t: float) -> tuple[np.ndarray, float, float]:
        i = np.clip(t + 1.0, 0, len(self.pts) - 1)
        a, f = int(np.floor(i)), i - np.floor(i)
        b = min(a + 1, len(self.pts) - 1)
        pos = self.pts[a] * (1 - f) + self.pts[b] * f
        head = self.heading[a] * (1 - f) + self.heading[b] * f
        spd = self.speed[a] * (1 - f) + self.speed[b] * f
        return pos, float(head), float(spd)

    def err_at(self, t: float) -> float:
        i = np.clip(t + 1.0, 0, len(self.err) - 1)
        a = int(np.floor(i))
        b = min(a + 1, len(self.err) - 1)
        return float(self.err[a] * (1 - (i - a)) + self.err[b] * (i - a))

    def _to_ego(self, world_pts: np.ndarray, pos, head) -> np.ndarray:
        c, s = np.cos(-head), np.sin(-head)
        r = (world_pts - pos) @ np.array([[c, s], [-s, c]])  # rotate by -heading
        return r

    def executed(self, t: float) -> np.ndarray:
        """Where the student's actually-executed commands lead from time t, in the current ego frame."""
        pos, head, _ = self.pose(t)
        i = t + 1.0
        future = self.pts[int(np.floor(i)) + 1 :]
        return self._to_ego(np.vstack([pos[None], future]), pos, head)

    def human_ahead(self, t: float) -> np.ndarray:
        pos, head, _ = self.pose(t)
        future = self.human[int(np.floor(t + 1.0)) + 1 :]
        return self._to_ego(np.vstack([pos[None], future]), pos, head)

    def expert_plan(self, t: float) -> np.ndarray:
        a = int(np.clip(np.floor(t), 0, self.n - 1))
        b = min(a + 1, self.n - 1)
        f = np.clip(t - a, 0, 1)
        return (self.plan[a][:, :3] * (1 - f) + self.plan[b][:, :3] * f).astype(float)

    def image(self, t: float) -> np.ndarray:
        """(H, W, 6) frame at continuous time t: flow-warped blend of the two neighbouring generated frames."""
        t = float(np.clip(t, 0, self.n - 1))
        a = int(np.floor(t))
        b = min(a + 1, self.n - 1)
        f = t - a
        if f < 1e-3 or a == b:
            return self.frames[a]
        out = np.empty_like(self.frames[a])
        for sl in (slice(0, 3), slice(3, 6)):
            ia, ib = self.frames[a][..., sl], self.frames[b][..., sl]
            ga, gb = cv2.cvtColor(ia, cv2.COLOR_RGB2GRAY), cv2.cvtColor(ib, cv2.COLOR_RGB2GRAY)
            dis = _dis()
            fwd = dis.calc(ga, gb, None)  # a -> b
            bwd = dis.calc(gb, ga, None)
            h, w = ga.shape
            gx, gy = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
            wa = cv2.remap(ia, gx - fwd[..., 0] * f, gy - fwd[..., 1] * f, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
            wb = cv2.remap(ib, gx - bwd[..., 0] * (1 - f), gy - bwd[..., 1] * (1 - f), cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
            out[..., sl] = cv2.addWeighted(wa, 1 - f, wb, f, 0)
        return out


_DIS = None


def _dis():
    global _DIS
    if _DIS is None:
        _DIS = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_MEDIUM)
    return _DIS


def load_results(path: Path, kick: str | None = None) -> dict:
    """Rows keyed by (segment, start, policy). Shove-test files hold a left and a right kick per key: pick one."""
    rows = {}
    for line in Path(path).read_text().splitlines():
        r = json.loads(line)
        if kick is not None and r.get("kick") != kick:
            continue
        rows[(r["segment"], r["start"], r["policy"])] = r
    return rows


def load_rollout(rows: dict, frames_dir: Path, segment: str, start: int, policy: str, label: str = "") -> Rollout:
    row = rows[(segment, start, policy)]
    suffix = f"_{row['kick']}" if "kick" in row else ""  # shove test: one npz per kick direction
    return Rollout(row, Path(frames_dir) / f"{segment}_{start}_{policy}{suffix}.npz", label or policy)


# ----------------------------------------------------------------------------------------------- drawing
def upscale(img: np.ndarray, width: int) -> np.ndarray:
    h = int(round(img.shape[0] * width / img.shape[1]))
    big = cv2.resize(img, (width, h), interpolation=cv2.INTER_CUBIC)
    blur = cv2.GaussianBlur(big, (0, 0), 1.2)
    return cv2.addWeighted(big, 1.35, blur, -0.35, 0)  # mild unsharp mask: the source is only 256 px wide


def draw_overlays(cam: np.ndarray, r: Rollout, t: float, show_human: bool = True) -> np.ndarray:
    """Draw ribbons/lines on an upscaled narrow-camera image (any width, 2:1)."""
    scale = cam.shape[1] / 256.0
    ss = 2  # supersample for clean edges
    layer = np.zeros((cam.shape[0] * ss, cam.shape[1] * ss, 4), np.uint8)

    def px(p):
        return np.round(p * scale * ss).astype(np.int32)

    rb = ribbon(r.executed(t))
    if rb is not None:
        left, right = rb
        poly = np.vstack([left, right[::-1]])
        if np.isfinite(poly).all():
            cv2.fillPoly(layer, [px(poly)], (*GREEN, 120), cv2.LINE_AA)
            cv2.polylines(layer, [px(left)], False, (*GREEN, 255), max(2, int(2.2 * scale * ss / 3)), cv2.LINE_AA)
            cv2.polylines(layer, [px(right)], False, (*GREEN, 255), max(2, int(2.2 * scale * ss / 3)), cv2.LINE_AA)
    plan = r.expert_plan(t).copy()
    plan = plan[plan[:, 0] > 1.5]
    if len(plan) > 1:
        dev = plan.copy()
        dev[:, 2] += CAM_HEIGHT
        pix = project(dev)
        if np.isfinite(pix).all():
            cv2.polylines(layer, [px(pix)], False, (*ORANGE, 255), max(2, int(2.6 * scale * ss / 3)), cv2.LINE_AA)
    if show_human:
        hp = r.human_ahead(t)
        hp = hp[hp[:, 0] > 1.5]
        if len(hp) > 1:
            pix = project(ego_to_device(hp))
            for p in pix[::2]:
                if np.isfinite(p).all():
                    cv2.circle(layer, tuple(px(p)), max(2, int(2.0 * scale * ss / 2)), (*INK, 235), -1, cv2.LINE_AA)
    layer = cv2.resize(layer, (cam.shape[1], cam.shape[0]), interpolation=cv2.INTER_AREA)
    alpha = layer[..., 3:4].astype(np.float32) / 255.0
    return (cam.astype(np.float32) * (1 - alpha) + layer[..., :3].astype(np.float32) * alpha).astype(np.uint8)


def draw_bev(r: Rollout, t: float, size: int, rollouts_extra=()) -> np.ndarray:
    """Top-down inset: human path (white), student trail so far (coloured), car as a marker. 4 s ahead window."""
    img = np.full((size, size, 3), 14, np.uint8)
    cv2.rectangle(img, (0, 0), (size - 1, size - 1), (46, 52, 62), 1)
    all_pts = np.vstack([r.human, r.pts] + [e.pts for e in rollouts_extra])
    x0, x1 = all_pts[:, 0].min() - 3, all_pts[:, 0].max() + 3
    y0, y1 = all_pts[:, 1].min(), all_pts[:, 1].max()
    span = max(x1 - x0, (y1 - y0) + 8, 24)
    ymid = 0.5 * (y0 + y1)

    def to_px(p):
        p = np.atleast_2d(p)
        u = (p[:, 1] - ymid) / span * (size - 20)  # lateral -> screen x (right positive -> screen right)
        v = (p[:, 0] - x0) / span * (size - 20)
        return np.stack([size / 2 + u, size - 10 - v], axis=1).astype(np.int32)

    cv2.polylines(img, [to_px(r.human)], False, INK, 2, cv2.LINE_AA)
    for e in rollouts_extra:
        cv2.polylines(img, [to_px(e.pts)], False, (70, 76, 88), 1, cv2.LINE_AA)
    i = int(np.clip(np.floor(t + 1), 0, len(r.pts) - 1))
    col = err_color(r.err_at(t))
    cv2.polylines(img, [to_px(r.pts[: i + 1])], False, col, 3, cv2.LINE_AA)
    pos, _, _ = r.pose(t)
    cv2.circle(img, tuple(to_px(pos)[0]), 6, col, -1, cv2.LINE_AA)
    cv2.circle(img, tuple(to_px(r.human[min(i, len(r.human) - 1)])[0]), 4, INK, 1, cv2.LINE_AA)
    return img


def err_color(e: float):
    return GREEN if e < 1.0 else AMBER if e < 2.0 else RED


# ----------------------------------------------------------------------------------------------- text helpers
class Canvas:
    """A PIL-backed frame so text and shapes stay crisp; convert to/from numpy once per frame."""

    def __init__(self, w: int, h: int, bg=BG):
        self.img = Image.new("RGB", (w, h), bg)
        self.d = ImageDraw.Draw(self.img, "RGBA")
        self.w, self.h = w, h

    def paste(self, arr: np.ndarray, xy):
        self.img.paste(Image.fromarray(arr), xy)

    def text(self, xy, s, size=32, fill=INK, bold=True, mono=False, anchor="la"):
        self.d.text(xy, s, font=font(size, bold, mono), fill=fill, anchor=anchor)

    def rrect(self, box, r=14, fill=PANEL, outline=None, width=2):
        self.d.rounded_rectangle(box, r, fill=fill, outline=outline, width=width)

    def array(self) -> np.ndarray:
        return np.asarray(self.img)


def ease(x: float) -> float:
    x = float(np.clip(x, 0, 1))
    return x * x * (3 - 2 * x)


# ----------------------------------------------------------------------------------------------- video io
class Writer:
    def __init__(self, path: Path, w: int, h: int, fps: int = FPS_OUT, crf: int = 15):
        self.p = subprocess.Popen(
            ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{w}x{h}", "-r", str(fps),
             "-i", "-", "-an", "-c:v", "libx264", "-preset", "slow", "-crf", str(crf), "-pix_fmt", "yuv420p",
             "-movflags", "+faststart", str(path)],
            stdin=subprocess.PIPE,
        )
        self.w, self.h = w, h
        self.n = 0

    def write(self, frame: np.ndarray):
        assert frame.shape == (self.h, self.w, 3), frame.shape
        self.p.stdin.write(np.ascontiguousarray(frame).tobytes())
        self.n += 1

    def hold(self, frame: np.ndarray, seconds: float):
        for _ in range(int(round(seconds * FPS_OUT))):
            self.write(frame)

    def close(self):
        self.p.stdin.close()
        self.p.wait()


def crossfade(w: Writer, a: np.ndarray, b: np.ndarray, seconds: float = 0.5):
    n = max(2, int(seconds * FPS_OUT))
    for i in range(n):
        f = ease((i + 1) / n)
        w.write((a.astype(np.float32) * (1 - f) + b.astype(np.float32) * f).astype(np.uint8))
