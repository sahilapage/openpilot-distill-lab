"""'Same road, never seen' video (4:5, 1080x1350, 20 fps = the camera's own rate) from the open-loop eval.

The human drives. Every model watches the same REAL narrow-camera video of a drive none of the students trained on
(repo val split) and outputs a steering command at 5 Hz (eval_open_loop.py, source raw). On the camera: each model's
commanded path, the constant-curvature arc it steers for the next ~2.5 s, projected with the comma four narrow camera
intrinsics and the drive's own calibration; white dots: the human's current arc. Below: lateral acceleration over the
clip, every model vs the human, with a playhead. Commands are linearly interpolated between the 5 Hz steps.

  python make_realroad.py OUT.mp4 REAL_DIR results_open_loop_more.jsonl unseen_all.txt SEG:START:STEPS ...
REAL_DIR/<seg>/ holds fcamera.hevc and localizer.safetensors copied from the box (comma1M: never commit these).
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parent))
from render import Writer, font  # noqa: E402

W, H, SS = 1080, 1350, 2
CAM_W, CAM_H, CAM_K = 1344, 760, np.array([[1141.5, 0.0, 672.0], [0.0, 1141.5, 380.0], [0.0, 0.0, 1.0]])
CROP = (60, 40, 1284, 660)  # x0, y0, x1, y1 of the camera frame shown (hood cropped off)
CAM_Y = 150
PANEL_Y0, PANEL_Y1 = 800, 1225
HEIGHT = 1.22
BG = (16, 16, 15)
INK, INK2, MUTED, GRID = (240, 239, 236), (195, 194, 183), (130, 129, 122), (52, 52, 49)
POLICIES = [  # name in results, label, colour (dataviz reference palette, dark steps)
    ("teacher", "comma's model", (25, 190, 130)),
    ("morefinal", "my student, 360 drives", (57, 135, 229)),
    ("student40k", "my student, 128 drives", (230, 100, 50)),
]


def load_tensor(path: Path, name: str) -> np.ndarray:
    """One tensor from a .safetensors file (8-byte header length, JSON header, raw little-endian data)."""
    b = path.read_bytes()
    n = int.from_bytes(b[:8], "little")
    h = json.loads(b[8 : 8 + n])[name]
    dtype = {"F64": "<f8", "F32": "<f4", "I64": "<i8", "I32": "<i4"}[h["dtype"]]
    lo, hi = h["data_offsets"]
    return np.frombuffer(b[8 + n + lo : 8 + n + hi], dtype).reshape(h["shape"])


def rot_from_euler(rpy) -> np.ndarray:
    r, p, y = rpy
    rx = np.array([[1, 0, 0], [0, np.cos(r), -np.sin(r)], [0, np.sin(r), np.cos(r)]])
    ry = np.array([[np.cos(p), 0, np.sin(p)], [0, 1, 0], [-np.sin(p), 0, np.cos(p)]])
    rz = np.array([[np.cos(y), -np.sin(y), 0], [np.sin(y), np.cos(y), 0], [0, 0, 1]])
    return rz @ ry @ rx


def arc(curv: float, length: float, n: int = 40, s0: float = 5.0) -> np.ndarray:
    """Ground points (x forward, y right, z down = camera height) of a constant-curvature path."""
    s = np.linspace(s0, max(length, s0 + 1), n)
    if abs(curv) < 1e-6:
        x, y = s, 0.5 * curv * s**2
    else:
        x, y = np.sin(curv * s) / curv, (1 - np.cos(curv * s)) / curv
    return np.stack([x, y, np.full_like(x, HEIGHT)], 1)


def project(pts_calib: np.ndarray, dev_from_calib: np.ndarray, scale: float) -> np.ndarray:
    dev = pts_calib @ dev_from_calib.T
    view = np.stack([dev[:, 1], dev[:, 2], dev[:, 0]], 1)
    uv = (view @ CAM_K.T)[:, :2] / view[:, 2:3]
    return (uv - np.array(CROP[:2])) * scale


def decode(hevc: Path, first: int, count: int) -> np.ndarray:
    """Frames first..first+count-1 of the raw HEVC stream as RGB."""
    cmd = ["ffmpeg", "-loglevel", "error", "-f", "hevc", "-i", str(hevc), "-vf", f"select=between(n\\,{first}\\,{first + count - 1})",
           "-fps_mode", "passthrough", "-f", "rawvideo", "-pix_fmt", "rgb24", "-"]
    raw = subprocess.run(cmd, capture_output=True, check=True).stdout
    frames = np.frombuffer(raw, np.uint8).reshape(-1, CAM_H, CAM_W, 3)
    if len(frames) != count:
        raise RuntimeError(f"decoded {len(frames)} frames, wanted {count}")
    return frames


def interp(a: np.ndarray, t: float) -> float:
    return float(np.interp(t, np.arange(len(a)), a))


def draw_path(d: ImageDraw.ImageDraw, uv: np.ndarray, colour, width_near: float, dotted: bool = False) -> None:
    ymax = (CAM_Y + (CROP[3] - CROP[1]) * W / (CROP[2] - CROP[0]) - 2) * SS
    pts = [(float(u) * SS, float(v) * SS + CAM_Y * SS) for u, v in uv]
    pts = [q for q in pts if q[1] <= ymax]
    n = len(pts)
    for i in range(n - 1):
        w = max(2.0, width_near * (1 - i / n)) * SS
        if dotted:
            if i % 3 == 0:
                d.ellipse([pts[i][0] - w * 0.6, pts[i][1] - w * 0.6, pts[i][0] + w * 0.6, pts[i][1] + w * 0.6], fill=colour)
        else:
            d.line([pts[i], pts[i + 1]], fill=colour, width=int(w))


def main() -> None:
    out, real_dir, res_p, val_p = sys.argv[1], Path(sys.argv[2]), sys.argv[3], sys.argv[4]
    picks = [s.split(":") for s in sys.argv[5:]]
    val = set(open(val_p).read().split())
    rows = {(r["policy"], r["segment"]): r for r in map(json.loads, open(res_p)) if r["source"] == "raw"}
    wr = Writer(Path(out), W, H, fps=20)
    scale = W / (CROP[2] - CROP[0])
    for ci, (seg, start, steps) in enumerate(picks):
        start, steps = int(start), int(steps)
        assert seg in val, f"{seg} is not a never-seen drive"
        rpy = load_tensor(real_dir / seg / "localizer.safetensors", "rpy")
        dev_from_calib = rot_from_euler(np.asarray(rpy, float).reshape(-1)[:3])
        frames = decode(real_dir / seg / "fcamera.hevc", start * 4, steps * 4)
        speeds = np.asarray(rows[("teacher", seg)]["speeds"], float)
        human = np.asarray(rows[("teacher", seg)]["human_actions"], float)[:, 0]
        curv = {p: np.asarray(rows[(p, seg)]["actions"], float)[:, 0] for p, _, _ in POLICIES}
        lat = {p: speeds[: len(c)] ** 2 * c[: len(speeds)] for p, c in curv.items()}
        lat_h = speeds[: len(human)] ** 2 * human[: len(speeds)]
        sl = slice(start, start + steps)
        vmax = max(1.0, max(np.abs(lat_h[sl]).max(), *(np.abs(v[sl]).max() for v in lat.values())) * 1.15)
        last = None
        for n in range(steps * 4):
            t = start + n / 4
            img = Image.new("RGB", (W * SS, H * SS), BG)
            crop = frames[n][CROP[1] : CROP[3], CROP[0] : CROP[2]]
            cam = Image.fromarray(crop).resize((W * SS, int((CROP[3] - CROP[1]) * scale) * SS), Image.BILINEAR)
            img.paste(cam, (0, CAM_Y * SS))
            d = ImageDraw.Draw(img, "RGBA")
            v = interp(speeds, t)
            length = min(max(v * 2.5, 15.0), 55.0)
            draw_path(d, project(arc(interp(human, t), length), dev_from_calib, scale), (255, 255, 255, 230), 9, dotted=True)
            for p, _, colour in reversed(POLICIES):
                draw_path(d, project(arc(interp(curv[p], t), length), dev_from_calib, scale), (*colour, 235), 12)
            # header
            f = lambda size, bold=True: font(size * SS, bold)
            d.text((44 * SS, 34 * SS), "same road. none of my students ever saw it", font=f(44), fill=INK)
            d.text((44 * SS, 94 * SS), "the human drives. every model watches the same real video and steers", font=f(26, False), fill=INK2)
            # legend chips on the camera
            for i, (p, label, colour) in enumerate(POLICIES):
                y = (CAM_Y + 22 + i * 46) * SS
                d.rounded_rectangle([22 * SS, y, 22 * SS + 76 * SS + f(24).getlength(label) + 16 * SS, y + 38 * SS], 10 * SS, fill=(0, 0, 0, 165))
                d.rounded_rectangle([36 * SS, y + 15 * SS, 64 * SS, y + 23 * SS], 4 * SS, fill=colour)
                d.text((76 * SS, y + 19 * SS), label, font=f(24), fill=INK, anchor="lm")
            y = (CAM_Y + 22 + 3 * 46) * SS
            d.rounded_rectangle([22 * SS, y, 22 * SS + 76 * SS + f(24).getlength('the human') + 16 * SS, y + 38 * SS], 10 * SS, fill=(0, 0, 0, 165))
            for k in range(3):
                d.ellipse([(38 + k * 10) * SS, y + 15 * SS, (46 + k * 10) * SS, y + 23 * SS], fill=(255, 255, 255))
            d.text((76 * SS, y + 19 * SS), "the human", font=f(24), fill=INK, anchor="lm")
            # steering trace panel
            x0, x1 = 44 * SS, (W - 44) * SS
            y0, y1 = PANEL_Y0 * SS, PANEL_Y1 * SS
            ym = (y0 + y1) / 2
            d.text((x0, y0 - 46 * SS), "steering (lateral acceleration) over this clip", font=f(26), fill=INK2)
            d.line([(x0, ym), (x1, ym)], fill=GRID, width=2 * SS)
            d.text((x1, y0 - 4 * SS), "right", font=f(20, False), fill=MUTED, anchor="rt")
            d.text((x1, y1 + 4 * SS), "left", font=f(20, False), fill=MUTED, anchor="rb")
            X = lambda tt: x0 + (x1 - x0) * (tt - start) / (steps - 1)
            Y = lambda val_: ym - (y1 - y0) / 2 * val_ / vmax
            ts = np.arange(start, start + steps)
            hx = [(X(tt), Y(lat_h[tt])) for tt in ts]
            for k in range(0, len(hx), 1):
                d.ellipse([hx[k][0] - 4 * SS, hx[k][1] - 4 * SS, hx[k][0] + 4 * SS, hx[k][1] + 4 * SS], fill=(255, 255, 255, 200))
            for p, _, colour in reversed(POLICIES):
                d.line([(X(tt), Y(lat[p][tt])) for tt in ts], fill=(*colour, 255), width=4 * SS, joint="curve")
            px = X(min(t, start + steps - 1))
            d.rectangle([px, y0, x1, y1], fill=(16, 16, 15, 150))  # future dimmed
            d.line([(px, y0), (px, y1)], fill=(255, 255, 255, 200), width=2 * SS)
            d.text((44 * SS, (H - 52) * SS), "open loop: the human drives; models only steer in their heads, on the same real frames  ·  clip "
                   f"{ci + 1}/{len(picks)}", font=f(19, False), fill=MUTED)
            frame = np.asarray(img.resize((W, H), Image.LANCZOS))
            if last is not None and n < 8:  # crossfade from the previous clip
                a = (n + 1) / 9
                frame = (last.astype(np.float32) * (1 - a) + frame.astype(np.float32) * a).astype(np.uint8)
            wr.write(frame)
            prev_frame = frame
        last = prev_frame
        for _ in range(16):  # hold 0.8 s on the last frame
            wr.write(last)
    wr.close()
    print(f"wrote {out}: {wr.n} frames, {wr.n / 20:.1f} s")


if __name__ == "__main__":
    main()
