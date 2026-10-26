"""Synthetic stand-in for shove-test output (results_shove.jsonl + frames npz), to test the layout. NOT real data:
a simple feedback controller with a per-policy gain plays the model, a 15 degree kick is forced for 3 frames."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from make_mock import FPS, START, ego, integrate, road_frame  # noqa: E402

T = 25
KICK_DEG, KICK_FRAMES = 15.0, 3


def drive(human, human_c, speed0, gain, kick_sign):
    speed, pos, head = speed0, np.zeros(2), 0.0
    pts, heads, acts = [pos.copy()], [0.0], []
    tang = np.gradient(human, axis=0)
    for k in range(T):
        if k < KICK_FRAMES:
            c = np.deg2rad(KICK_DEG) * kick_sign / KICK_FRAMES * FPS / max(speed, 1.0)
        else:
            i = int(np.argmin(np.linalg.norm(human - pos, axis=1)))
            th = np.arctan2(tang[i, 1], tang[i, 0])
            n_left = np.array([-np.sin(th), np.cos(th)])
            e, herr = float((pos - human[i]) @ n_left), head - th
            c = human_c[k] - gain * (0.012 * e + 0.09 * herr)
        yaw, dist = speed * c / FPS, speed / FPS
        dx, dy = dist * np.cos(yaw / 2), dist * np.sin(yaw / 2)
        pos = pos + np.array([dx * np.cos(head) - dy * np.sin(head), dx * np.sin(head) + dy * np.cos(head)])
        head += yaw
        pts.append(pos.copy())
        heads.append(head)
        acts.append((c, 0.0))
    return np.array(pts), np.array(heads), acts


def main(out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    (out / "frames").mkdir(exist_ok=True)
    rng = np.random.default_rng(3)
    speed0 = 12.0
    t = np.arange(T) / FPS
    human_c = 0.015 * np.sin(t * 0.5) + 0.006
    human = integrate(speed0, np.stack([human_c, np.zeros(T)], 1))
    rows = []
    seg = "mockshove" + "0" * 23
    for kname, sign in (("L", 1.0), ("R", -1.0)):
        for name, gain in (("straight", 0.0), ("student", 0.35), ("dagger1400", 2.0)):
            pts, heads, acts = drive(human, human_c, speed0, gain, sign)
            err = np.linalg.norm(pts[1:] - human[1:], axis=1)
            frames, plans = [], []
            for k in range(T):
                pos, head = pts[k + 1], heads[k + 1]
                narrow = road_frame(human, pos, head, rng)
                wide = cv2.resize(cv2.GaussianBlur(narrow, (0, 0), 1.2), (256, 128))
                frames.append(np.concatenate([narrow, wide], -1))
                e = ego(human[k + 1 :], pos, head)
                e = e[e[:, 0] > 0]
                xs = np.linspace(2, 60, 33)
                ys = np.interp(xs, e[:, 0], e[:, 1]) if len(e) > 1 else np.zeros(33)
                plans.append(np.stack([xs, -ys, np.zeros(33)], 1))
            np.savez_compressed(out / "frames" / f"{seg}_{START}_{name}_{kname}.npz", frames=np.stack(frames), plan=np.stack(plans))
            tail = err[-FPS:]
            rows.append({"segment": seg, "start": START, "policy": name, "kick": kname, "kick_deg": KICK_DEG * sign,
                         "kick_frames": KICK_FRAMES, "frames": T, "ade": float(err.mean()), "fde": float(err[-1]),
                         "max_err": float(err.max()), "success": bool(err.max() <= 2.0), "peak_err": float(err.max()),
                         "tail_err": float(tail.mean()), "recovered": bool(tail.max() < 1.0), "t_recover": 0.0,
                         "policy_path": np.round(pts, 2).tolist(), "human_path": np.round(human, 2).tolist(),
                         "actions": np.round(np.asarray(acts), 5).tolist()})
    (out / "results_shove.jsonl").write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    print("mock shove written to", out, {(r["policy"], r["kick"]): (round(r["peak_err"], 1), round(r["tail_err"], 1), r["recovered"]) for r in rows})


if __name__ == "__main__":
    main(Path(sys.argv[1]))
