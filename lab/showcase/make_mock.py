"""Synthetic stand-in for eval output (results.jsonl + frames npz) so the renderer can be built before the real run ends.
NOT real data: roads are drawn from the same projection the overlay uses, policies are a human path plus a bias."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from render import K, ego_to_device, project  # noqa: E402

FPS = 5
T = 20
SEG, START = "mockseg0000000000000000000000000", 42


def integrate(speed0, acts):
    speed, x, y, h = float(speed0), 0.0, 0.0, 0.0
    pts = [(0.0, 0.0)]
    for c, a in acts:
        speed = max(speed + a / FPS, 0.0)
        yaw = speed * c / FPS
        dist = speed / FPS
        dx, dy = dist * np.cos(yaw / 2), dist * np.sin(yaw / 2)
        x += dx * np.cos(h) - dy * np.sin(h)
        y += dx * np.sin(h) + dy * np.cos(h)
        h += yaw
        pts.append((x, y))
    return np.array(pts)


def ego(world, pos, head):
    c, s = np.cos(-head), np.sin(-head)
    return (world - pos) @ np.array([[c, s], [-s, c]])


def road_frame(human_world, pos, head, rng):
    img = np.zeros((128, 256, 3), np.uint8)
    img[:24] = (150, 180, 215)
    img[24:] = (92, 112, 78)
    xs = np.linspace(1.6, 90, 90)
    # dense human path in world coords
    dense = np.stack([np.interp(xs + pos[0], human_world[:, 0], human_world[:, 0]), np.interp(xs + pos[0], human_world[:, 0], human_world[:, 1])], 1)
    e = ego(dense, pos, head)
    e = e[e[:, 0] > 1.6]
    if len(e) > 2:
        d = np.gradient(e, axis=0)
        d /= np.linalg.norm(d, axis=1, keepdims=True) + 1e-9
        n = np.stack([-d[:, 1], d[:, 0]], 1)
        L = project(ego_to_device(e + n * 1.9))
        R = project(ego_to_device(e - n * 1.9))
        poly = np.vstack([L, R[::-1]]).astype(np.int32)
        cv2.fillPoly(img, [poly], (68, 70, 74), cv2.LINE_AA)
        cv2.polylines(img, [project(ego_to_device(e)).astype(np.int32)], False, (200, 190, 120), 1, cv2.LINE_AA)
    img = np.clip(img.astype(np.float32) + rng.normal(0, 4, img.shape), 0, 255).astype(np.uint8)
    return img


def build(out: Path, policies: dict[str, float], seg: str = SEG, seed: int = 0, rows_all: list | None = None, curve: float = 1.0):
    rng = np.random.default_rng(seed)
    out.mkdir(parents=True, exist_ok=True)
    (out / "frames").mkdir(exist_ok=True)
    speed0 = 10.0
    t = np.arange(T) / FPS
    human_c = curve * (0.02 * np.sin(t * 0.6) + 0.012 + 0.012 * (t > 2))  # gentle bendy road
    acts_h = np.stack([human_c, np.zeros(T)], 1)
    human = integrate(speed0, acts_h)
    rows = []
    for name, bias in policies.items():
        c = human_c + bias * (0.2 + t / t[-1] * 0.8) * (0.012 + 0.01 * np.sin(t))
        acts = np.stack([c, np.zeros(T)], 1)
        pts = integrate(speed0, acts)
        err = np.linalg.norm(pts[1:] - human[1:], axis=1)
        heads = np.concatenate([[0], np.cumsum(np.linalg.norm(np.diff(pts, axis=0), axis=1) * c)])
        frames, plans = [], []
        for k in range(T):
            pos, head = pts[k + 1], heads[k + 1]
            narrow = road_frame(human, pos, head, rng)
            wide = cv2.resize(cv2.GaussianBlur(narrow, (0, 0), 1.2), (256, 128))
            frames.append(np.concatenate([narrow, wide], -1))
            fut = human[k + 1 :]
            e = ego(np.vstack([human[k + 1][None], fut]), pos, head)
            e = e[e[:, 0] > 0]
            xs = np.linspace(2, 60, 33)
            ys = np.interp(xs, e[:, 0], e[:, 1]) if len(e) > 1 else np.zeros(33)
            plans.append(np.stack([xs, -ys, np.zeros(33)], 1))  # device frame y = right
        np.savez_compressed(out / "frames" / f"{seg}_{START}_{name}.npz", frames=np.stack(frames), plan=np.stack(plans))
        rows.append({"segment": seg, "start": START, "policy": name, "frames": T, "ade": float(err.mean()), "fde": float(err[-1]),
                     "max_err": float(err.max()), "success": bool(err.max() <= 2.0), "policy_path": np.round(pts, 2).tolist(),
                     "human_path": np.round(human, 2).tolist(), "actions": np.round(acts, 5).tolist()})
    if rows_all is not None:
        rows_all.extend(rows)
        return
    (out / "results.jsonl").write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    print("mock written to", out, {r["policy"]: round(r["max_err"], 2) for r in rows})


if __name__ == "__main__":
    out = Path(sys.argv[1])
    rows_all: list = []
    rng = np.random.default_rng(5)
    for i in range(12):
        k = rng.uniform(0.5, 1.1)
        seg = f"mockseg{i:02d}" + "0" * 23
        sign = rng.choice([-1.0, 1.0])
        build(out, {n: sign * v * k for n, v in {"student": 1.0, "dagger10": 0.8, "dagger50": 0.55, "dagger100": 0.32, "dagger200": 0.13, "teacher": 0.06, "straight": 0.0}.items()},
              seg=seg, seed=i, rows_all=rows_all, curve=rng.uniform(0.6, 1.4) * sign)
    (out / "results.jsonl").write_text("\n".join(json.dumps(r) for r in rows_all) + "\n")
    print("mock written to", out, len(rows_all), "rows")
