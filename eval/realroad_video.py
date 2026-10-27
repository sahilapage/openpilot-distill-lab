"""Real-road video: every model watches the same logged drive, and its steering is drawn on the camera.

The human drives. Each policy in an `eval.open_loop run` results file sees the real camera frame by frame and commands a
curvature at 5 Hz. On the narrow camera we draw, for every policy, the constant-curvature arc it is steering for the
next ~2.5 s (projected with the comma four narrow camera intrinsics and the drive's own calibration), and the human's
current arc as white dots. Commands are interpolated between the 5 Hz steps; the video runs at the camera's 20 fps.

  python -m eval.realroad_video out.mp4 open_loop.jsonl SEGMENT:START:STEPS [SEGMENT:START:STEPS ...] \\
      --policy teacher="comma's model" --policy student="my student"

START and STEPS are in 5 Hz steps. data/<segment>/ must hold fcamera.hevc and localizer.safetensors. Needs ffmpeg.
"""

import argparse
import json
import subprocess
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from safetensors.numpy import load_file

W, SS, FPS = 1080, 2, 20  # output width, supersampling, frames per second
CAM_W, CAM_H = 1344, 760
CAM_K = np.array([[1141.5, 0.0, 672.0], [0.0, 1141.5, 380.0], [0.0, 0.0, 1.0]])  # comma four narrow camera
CROP = (60, 40, 1284, 652)  # x0, y0, x1, y1 of the camera frame shown (hood cropped off): exactly 2:1
H = int((CROP[3] - CROP[1]) * W / (CROP[2] - CROP[0]))
CAM_HEIGHT = 1.22  # metres above the road
INK = (240, 239, 236)
COLOURS = [(25, 190, 130), (57, 135, 229), (230, 100, 50), (200, 90, 200), (235, 190, 40)]  # fixed order, never cycled
FONTS = ["/System/Library/Fonts/Avenir Next.ttc", "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
         "/usr/share/fonts/TTF/DejaVuSans-Bold.ttf"]


def font(size: int) -> ImageFont.FreeTypeFont:
    for path in FONTS:
        if Path(path).exists():
            return ImageFont.truetype(path, size, index=2 if path.endswith(".ttc") else 0)
    return ImageFont.load_default(size)


def rot_from_euler(rpy) -> np.ndarray:
    r, p, y = rpy
    rx = np.array([[1, 0, 0], [0, np.cos(r), -np.sin(r)], [0, np.sin(r), np.cos(r)]])
    ry = np.array([[np.cos(p), 0, np.sin(p)], [0, 1, 0], [-np.sin(p), 0, np.cos(p)]])
    rz = np.array([[np.cos(y), -np.sin(y), 0], [np.sin(y), np.cos(y), 0], [0, 0, 1]])
    return rz @ ry @ rx


def arc(curv: float, length: float, n: int = 40, s0: float = 5.0) -> np.ndarray:
    """Road points (x forward, y right, z down to the road) of a constant-curvature path."""
    s = np.linspace(s0, max(length, s0 + 1), n)
    if abs(curv) < 1e-6:
        x, y = s, 0.5 * curv * s**2
    else:
        x, y = np.sin(curv * s) / curv, (1 - np.cos(curv * s)) / curv
    return np.stack([x, y, np.full_like(x, CAM_HEIGHT)], 1)


def project(pts_calib: np.ndarray, dev_from_calib: np.ndarray) -> np.ndarray:
    dev = pts_calib @ dev_from_calib.T
    view = np.stack([dev[:, 1], dev[:, 2], dev[:, 0]], 1)
    uv = (view @ CAM_K.T)[:, :2] / view[:, 2:3]
    return (uv - np.array(CROP[:2])) * W / (CROP[2] - CROP[0])


def decode(hevc: Path, first: int, count: int) -> np.ndarray:
    """Camera frames first..first+count-1 of a raw HEVC stream, as RGB."""
    cmd = ["ffmpeg", "-loglevel", "error", "-f", "hevc", "-i", str(hevc), "-vf", f"select=between(n\\,{first}\\,{first + count - 1})",
           "-fps_mode", "passthrough", "-f", "rawvideo", "-pix_fmt", "rgb24", "-"]
    frames = np.frombuffer(subprocess.run(cmd, capture_output=True, check=True).stdout, np.uint8).reshape(-1, CAM_H, CAM_W, 3)
    if len(frames) != count:
        raise RuntimeError(f"decoded {len(frames)} frames, wanted {count}")
    return frames


def draw_path(d: ImageDraw.ImageDraw, uv: np.ndarray, colour, width_near: float, dotted: bool = False) -> None:
    pts = [(float(u) * SS, float(v) * SS) for u, v in uv if v <= H - 2]
    for i in range(len(pts) - 1):
        w = max(2.0, width_near * (1 - i / len(pts))) * SS
        if not dotted:
            d.line([pts[i], pts[i + 1]], fill=colour, width=int(w))
        elif i % 3 == 0:
            d.ellipse([pts[i][0] - w * 0.6, pts[i][1] - w * 0.6, pts[i][0] + w * 0.6, pts[i][1] + w * 0.6], fill=colour)


def legend(d: ImageDraw.ImageDraw, i: int, label: str, colour=None) -> None:
    f = font(24 * SS)
    y = (22 + i * 46) * SS
    d.rounded_rectangle([22 * SS, y, 98 * SS + f.getlength(label) + 16 * SS, y + 38 * SS], 10 * SS, fill=(0, 0, 0, 165))
    if colour is None:  # the human: three dots
        for k in range(3):
            d.ellipse([(38 + k * 10) * SS, y + 15 * SS, (46 + k * 10) * SS, y + 23 * SS], fill=(255, 255, 255))
    else:
        d.rounded_rectangle([36 * SS, y + 15 * SS, 64 * SS, y + 23 * SS], 4 * SS, fill=colour)
    d.text((76 * SS, y + 19 * SS), label, font=f, fill=INK, anchor="lm")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("out", type=Path)
    p.add_argument("results", type=Path, help="jsonl from python -m eval.open_loop run")
    p.add_argument("clips", nargs="+", help="SEGMENT:START:STEPS")
    p.add_argument("--policy", action="append", required=True, help='name="label", drawn front to back in this order')
    p.add_argument("--data", type=Path, default=Path("data"))
    args = p.parse_args()
    policies = [(name, label, COLOURS[i]) for i, (name, label) in enumerate(s.split("=", 1) for s in args.policy)]
    rows = {(r["policy"], r["segment"]): r for r in map(json.loads, args.results.read_text().splitlines())}

    ffmpeg = subprocess.Popen(
        ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}", "-r", str(FPS),
         "-i", "-", "-an", "-c:v", "libx264", "-preset", "slow", "-crf", "15", "-pix_fmt", "yuv420p", "-movflags", "+faststart",
         str(args.out)], stdin=subprocess.PIPE)
    written, last = 0, None
    for clip in args.clips:
        seg, start, steps = clip.split(":")
        start, steps = int(start), int(steps)
        rpy = load_file(str(args.data / seg / "localizer.safetensors"))["rpy"]
        dev_from_calib = rot_from_euler(np.asarray(rpy, float).reshape(-1)[:3])
        frames = decode(args.data / seg / "fcamera.hevc", start * 4, steps * 4)
        ref = rows[(policies[0][0], seg)]
        speeds, human = np.asarray(ref["speeds"], float), np.asarray(ref["human_actions"], float)[:, 0]
        curv = {name: np.asarray(rows[(name, seg)]["actions"], float)[:, 0] for name, _, _ in policies}
        at = lambda a, t: float(np.interp(t, np.arange(len(a)), a))  # noqa: E731
        for n in range(steps * 4):
            t = start + n / 4
            crop = frames[n][CROP[1] : CROP[3], CROP[0] : CROP[2]]
            img = Image.fromarray(crop).resize((W * SS, H * SS), Image.BILINEAR)
            d = ImageDraw.Draw(img, "RGBA")
            length = min(max(at(speeds, t) * 2.5, 15.0), 55.0)
            draw_path(d, project(arc(at(human, t), length), dev_from_calib), (255, 255, 255, 230), 9, dotted=True)
            for name, _, colour in reversed(policies):
                draw_path(d, project(arc(at(curv[name], t), length), dev_from_calib), (*colour, 235), 12)
            for i, (_, label, colour) in enumerate(policies):
                legend(d, i, label, colour)
            legend(d, len(policies), "the human")
            frame = np.asarray(img.resize((W, H), Image.LANCZOS))
            if last is not None and n < 8:  # crossfade from the previous clip
                a = (n + 1) / 9
                frame = (last.astype(np.float32) * (1 - a) + frame.astype(np.float32) * a).astype(np.uint8)
            ffmpeg.stdin.write(np.ascontiguousarray(frame).tobytes())
            written += 1
            current = frame
        last = current
        for _ in range(16):  # hold 0.8 s on the last frame
            ffmpeg.stdin.write(last.tobytes())
            written += 1
    ffmpeg.stdin.close()
    ffmpeg.wait()
    print(f"wrote {args.out}: {written} frames, {written / FPS:.1f} s")


if __name__ == "__main__":
    main()
