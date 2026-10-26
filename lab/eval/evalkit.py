"""Shared helpers for the closed-loop evaluation: human reference path, policy path integration, metrics.

Conventions: 5 Hz steps, ego frame at the start of the rollout, in the REPO's frame: openpilot's calibrated frame, x forward,
y RIGHT, so a positive curvature is a RIGHT turn. That is the frame of rl.env.Physics, of the world model's pose input and
of get_curvature_from_plan (supervised/dataset.py builds plans in calib_from_device frame, yaw about z-down).
Before 2026-10-26 the human path here was y LEFT, i.e. mirrored against every policy path. Rows written since then carry
"ref_frame": "y_right"; human_path(row) mirrors older rows and rescore(row) recomputes their path metrics.
Run the self-test on CPU: PYTHONPATH=. python ../eval/evalkit.py <segment>...
"""
import sys

import numpy as np

FPS = 5
FRAME_SKIP = 20 // FPS  # localizer / camera frames are 20 Hz
SUCCESS_ERR_M = 2.0  # a rollout "succeeds" if it never strays more than this from the human path
REF_FRAME = "y_right"  # tag written on every row whose human_path is in the repo frame


def split_of(segment: str) -> str:
    """Same split rule as supervised.dataset.Comma1MDataset."""
    return "val" if hash(int(segment, 16)) % 10 == 0 else "train"


def human_reference(loc: dict, origin: int, n: int) -> tuple[np.ndarray, np.ndarray]:
    """Path (n+1, 2) and speed (n+1,) of the real drive from 5 Hz frame `origin`, in the origin ego frame."""
    from openpilot.common.transformations.orientation import rot_from_euler, rot_from_quat

    states = loc["frame_states"]
    idx = (origin + np.arange(n + 1)) * FRAME_SKIP
    pos = states[idx, 0:3]
    calib_from_device = rot_from_euler(loc["rpy"]).T
    device_from_ecef = rot_from_quat(states[idx[0], 3:7]).T
    rel = (pos - pos[0]) @ (calib_from_device @ device_from_ecef).T  # calib frame: x fwd, y right, z down
    path = rel[:, :2].copy()  # repo frame: y right
    speed = np.linalg.norm(states[idx, 7:10], axis=1)
    return path, speed


def integrate(speed0: float, actions) -> tuple[np.ndarray, np.ndarray]:
    """Replicates rl.env.Physics: returns path (n+1, 2) and speeds (n+1,) for (curvature, accel) actions."""
    speed, x, y, heading = float(speed0), 0.0, 0.0, 0.0
    pts, speeds = [(0.0, 0.0)], [speed]
    for curvature, accel in actions:
        speed = max(speed + accel / FPS, 0.0)
        yaw = speed * curvature / FPS
        dist = speed / FPS
        dx, dy = dist * np.cos(yaw / 2), dist * np.sin(yaw / 2)
        x += dx * np.cos(heading) - dy * np.sin(heading)
        y += dx * np.sin(heading) + dy * np.cos(heading)
        heading += yaw
        pts.append((x, y))
        speeds.append(speed)
    return np.asarray(pts), np.asarray(speeds)


def sensor_actions(loc: dict, origin: int, n: int, sign: float = 1.0) -> np.ndarray:
    """The human's (curvature, accel) per 5 Hz step, from the localizer's gyro and speed (not from path geometry).
    sign=-1 flips the turn direction; only used to show the self-test can detect a wrong convention."""
    from openpilot.common.transformations.orientation import rot_from_euler

    states = loc["frame_states"]
    calib_from_device = rot_from_euler(loc["rpy"]).T
    gyro_z = (states[:, 10:13] @ calib_from_device.T)[:, 2]  # calib frame is z-down: right turn is positive
    speed = np.linalg.norm(states[:, 7:10], axis=1)
    actions, prev = [], float(speed[origin * FRAME_SKIP])
    for k in range(n):
        lo, hi = (origin + k) * FRAME_SKIP, (origin + k + 1) * FRAME_SKIP
        new_speed = float(speed[hi])
        yaw_rate_right = float(gyro_z[lo:hi].mean())
        curvature = sign * yaw_rate_right / max(new_speed, 1.0)
        actions.append((curvature, (new_speed - prev) * FPS))
        prev = new_speed
    return np.asarray(actions)


def human_path(row: dict) -> np.ndarray:
    """The row's human path in the repo frame (y right), whatever convention the row was written with."""
    h = np.asarray(row["human_path"], float)
    return h if row.get("ref_frame") == REF_FRAME else h * np.array([1.0, -1.0])


def path_errors(row: dict) -> np.ndarray:
    return np.linalg.norm(np.asarray(row["policy_path"], float)[1:] - human_path(row)[1:], axis=1)


def rescore(row: dict) -> dict:
    """Recompute every human-path metric of a row in the repo frame (speed and comfort metrics do not depend on it)."""
    p, h = np.asarray(row["policy_path"], float), human_path(row)
    err = path_errors(row)
    out = dict(row)
    out |= {"ade": float(err.mean()), "fde": float(err[-1]), "max_err": float(err.max()),
            "lat_final": float(p[-1, 1] - h[-1, 1]), "long_final": float(p[-1, 0] - h[-1, 0]),
            "success": bool(err.max() <= SUCCESS_ERR_M), "human_path": np.round(h, 2).tolist(), "ref_frame": REF_FRAME}
    if "kick_frames" in row:  # shove / pull rows (eval_shove.recovery_metrics)
        kf = int(row["kick_frames"])
        tail = err[-FPS:]
        over = np.nonzero(err[kf:] > 1.0)[0]
        out |= {"peak_err": float(err.max()), "tail_err": float(tail.mean()), "recovered": bool(tail.max() < 1.0),
                "t_recover": float((over.max() + 1 + kf) / FPS) if len(over) else float(kf / FPS)}
        disturb = row.get("pull_dps") or row.get("kick_deg") or 0.0
        out["side"] = "right" if disturb > 0 else "left"  # + curvature is a right turn (the old "L" key is a right pull)
    return out


def _rms(x: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(x)))) if len(x) else 0.0


def rollout_metrics(policy_pts, policy_speed, actions, human_pts, human_speed) -> dict:
    err = np.linalg.norm(policy_pts[1:] - human_pts[1:], axis=1)
    a = np.asarray(actions, dtype=np.float64)
    lat_acc = policy_speed[1:] ** 2 * a[:, 0]
    return {
        "ade": float(err.mean()),
        "fde": float(err[-1]),
        "max_err": float(err.max()),
        "lat_final": float(policy_pts[-1, 1] - human_pts[-1, 1]),
        "long_final": float(policy_pts[-1, 0] - human_pts[-1, 0]),
        "speed_rmse": _rms(policy_speed[1:] - human_speed[1:]),
        "lat_acc_rms": _rms(lat_acc),
        "jerk_lat_rms": _rms(np.diff(lat_acc)) * FPS,
        "jerk_long_rms": _rms(np.diff(a[:, 1])) * FPS,
        "success": bool(err.max() <= SUCCESS_ERR_M),
    }


def self_test(segments: list[str], root: str = ".", n: int = 20) -> None:
    """CPU-only checks. Human actions come from the gyro + speed; integrating them must land near the human path
    (validates that path and gyro agree; it can NOT catch a frame mirrored consistently in both, which is how the old y-left
    convention slipped through: the policy side must be checked against real-frame actions, see check_policy_frame). The same test with the turn direction flipped
    must be clearly worse, otherwise the test could not catch a sign error. Also prints a human comfort baseline."""
    from pathlib import Path

    from safetensors.numpy import load_file

    right, flipped, jerk_lat, jerk_long, lat_acc, curvy = [], [], [], [], [], []
    for seg in segments:
        loc = load_file(Path(root) / "data" / seg / "localizer.safetensors")
        for origin in range(1, len(loc["frame_states"]) // FRAME_SKIP - n - 1, 7):
            path, speed = human_reference(loc, origin, n)
            if speed[0] < 3.0:
                continue
            acts = sensor_actions(loc, origin, n)
            pts, speeds = integrate(speed[0], acts)
            err = np.linalg.norm(pts[1:] - path[1:], axis=1)
            bad_pts, _ = integrate(speed[0], sensor_actions(loc, origin, n, sign=-1.0))
            bad = np.linalg.norm(bad_pts[1:] - path[1:], axis=1)
            right.append(err)
            flipped.append(bad)
            curvy.append(abs(path[-1, 1]))  # final lateral offset: how much the path bends
            lat_acc.append(speeds[1:] ** 2 * acts[:, 0])
            jerk_lat.append(np.diff(lat_acc[-1]) * FPS)
            jerk_long.append(np.diff(acts[:, 1]) * FPS)
    right, flipped, curvy = np.array(right), np.array(flipped), np.array(curvy)
    bendy = curvy > 3.0  # windows where the path really turns
    print(f"segments={len(segments)} windows={len(right)} (bendy={int(bendy.sum())})")
    print(f"gyro-driven path vs human path, final error: median {np.median(right[:, -1]):.2f} m, "
          f"mean ADE {right.mean():.2f} m; on bendy windows ADE {right[bendy].mean():.2f} m")
    print(f"same with turn direction FLIPPED:           median {np.median(flipped[:, -1]):.2f} m, "
          f"mean ADE {flipped.mean():.2f} m; on bendy windows ADE {flipped[bendy].mean():.2f} m  (must be much worse)")
    print(f"human comfort baseline: lat_acc rms {_rms(np.concatenate(lat_acc)):.3f} m/s^2, "
          f"lat jerk rms {_rms(np.concatenate(jerk_lat)):.3f}, long jerk rms {_rms(np.concatenate(jerk_long)):.3f} m/s^3")


if __name__ == "__main__":
    self_test(sys.argv[1:])
