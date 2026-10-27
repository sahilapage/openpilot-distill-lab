"""Does a model steer like the human on roads it never trained on?

`run`: every policy watches the real, logged drive frame by frame (narrow + wide camera, warped exactly as the world
model's episodes are) and outputs its (curvature, accel) at 5 Hz. The human's own curvature comes from the localizer
gyro and speed. Rows are appended to a jsonl file; finished (segment, policy) pairs are skipped on rerun.

`summary`: steering gain = slope of the policy's lateral acceleration (v^2 * curvature) on the human's, at the lag
where they match best (policies plan ahead), plus correlation, with a 95% bootstrap over drives. Gain 1.0 = turns
the wheel as much as the human. Reported separately for drives in the training split and for unseen drives (the
supervised loader's validation split, hash(segment) % 10 == 0), so memorisation shows up as a gap.

  python -m rl.server &
  python -m eval.open_loop run --segments eval.txt --out open_loop.jsonl \\
      --policy teacher=teacher --policy student=student:outputs/supervised/supercombo.pt
  python -m eval.open_loop summary open_loop.jsonl
"""

import argparse
import json
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

FPS, FRAME_SKIP = 5, 4
WARM, MAX_LAG, MIN_SPEED = 10, 8, 5.0  # skip the first 2 s (no history yet), search lags up to 1.6 s, ignore < 5 m/s


def is_unseen(segment: str) -> bool:
    return hash(int(segment, 16)) % 10 == 0  # same rule as supervised/dataset.py's validation split


def human_actions(loc: dict, n: int) -> np.ndarray:
    """The human's (curvature, accel) per 5 Hz step from the localizer (calibrated frame, positive = right turn)."""
    from openpilot.common.transformations.orientation import rot_from_euler

    states = loc["frame_states"]
    gyro_z = (states[:, 10:13] @ rot_from_euler(loc["rpy"]))[:, 2]
    speed = np.linalg.norm(states[:, 7:10], axis=1)
    out, prev = [], float(speed[0])
    for k in range(n):
        lo, hi = k * FRAME_SKIP, (k + 1) * FRAME_SKIP
        new = float(speed[hi])
        out.append((float(gyro_z[lo:hi].mean()) / max(new, 1.0), (new - prev) * FPS))
        prev = new
    return np.asarray(out)


class CaptureEncode:
    """Runtime stand-in for Episode: keeps the warped camera frames it is asked to encode."""

    def __init__(self, rt):
        self.rt, self.vae, self.frames = rt, rt.vae, []

    def encode(self, frames):
        self.frames.append(np.asarray(frames).copy())
        return self.rt.encode(frames)


def run(args) -> None:
    from safetensors.numpy import load_file

    from rl.actor import SupercomboActor
    from rl.env import Episode
    from rl.server import SERVER_URL, RuntimeClient

    rt = RuntimeClient(SERVER_URL, 15, 2.0)
    actors = {}
    for spec in args.policy:
        name, rest = spec.split("=", 1)
        kind, *path = rest.split(":", 1)
        actors[name] = SupercomboActor.from_onnx() if kind == "teacher" else SupercomboActor.from_torch_checkpoint(path[0])
    done = set()
    if args.out.exists():
        done = {(r["segment"], r["policy"]) for r in map(json.loads, args.out.read_text().splitlines())}
    segments = args.segments.read_text().split()
    t0 = time.time()
    with args.out.open("a") as out:
        for si, seg in enumerate(segments, 1):
            todo = [n for n in actors if (seg, n) not in done]
            if not todo:
                continue
            cap = CaptureEncode(rt)
            ep = Episode(cap, seg, cache_dir=None)  # no cache: forces decode + warp, so the real frames are captured
            frames = np.concatenate(cap.frames)
            n = len(frames) - 1
            human = human_actions(load_file(Path("data") / seg / "localizer.safetensors"), n)
            speeds = np.asarray(ep.speeds, float)
            for name in todo:
                actor = actors[name]
                actor.reset()
                acts = [actor.act(np.asarray(frames[t]), float(speeds[t])) for t in range(n)]
                out.write(json.dumps({"segment": seg, "policy": name, "actions": np.round(acts, 6).tolist(),
                                      "human_actions": np.round(human, 6).tolist(),
                                      "speeds": np.round(speeds[: n + 1], 3).tolist()}) + "\n")
                out.flush()
            print(f"[{si}/{len(segments)}] {seg[:8]}: {len(todo)} policies, {(time.time() - t0) / 60:.1f} min", flush=True)


def pairs(row: dict, lag: int):
    a = np.asarray(row["actions"], float)[:, 0]
    h = np.asarray(row["human_actions"], float)[:, 0]
    v = np.asarray(row["speeds"], float)
    n = min(len(a), len(h), len(v)) - lag
    idx = np.arange(WARM, n)
    idx = idx[v[idx] > MIN_SPEED]
    return v[idx] ** 2 * a[idx], v[idx + lag] ** 2 * h[idx + lag]


def table(rows: list[dict], title: str) -> None:
    groups = defaultdict(list)
    for r in rows:
        groups[r["policy"]].append(r)
    rng = np.random.default_rng(0)
    print(f"\n{title}: {len({r['segment'] for r in rows})} drives\n{'policy':>14}  lag   steering gain [95% CI]   corr")
    for pol, rs in sorted(groups.items()):
        lag = max(range(MAX_LAG + 1), key=lambda L: np.corrcoef(*np.concatenate([np.stack(pairs(r, L)) for r in rs], axis=1))[0, 1])
        per = [pairs(r, lag) for r in rs]

        def gain(items):
            a, h = np.concatenate([i[0] for i in items]), np.concatenate([i[1] for i in items])
            return float(np.sum(a * h) / np.sum(h * h))

        boot = [gain([per[i] for i in rng.integers(len(per), size=len(per))]) for _ in range(2000)]
        a, h = np.concatenate([i[0] for i in per]), np.concatenate([i[1] for i in per])
        print(f"{pol:>14}  {lag / FPS:.1f}s  {gain(per):6.2f} [{np.percentile(boot, 2.5):.2f}, {np.percentile(boot, 97.5):.2f}]"
              f"     {np.corrcoef(a, h)[0, 1]:+.2f}")


def summary(args) -> None:
    rows = [json.loads(line) for line in args.results.read_text().splitlines() if line.strip()]
    table([r for r in rows if is_unseen(r["segment"])], "UNSEEN drives (never trained on)")
    table([r for r in rows if not is_unseen(r["segment"])], "training drives")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--segments", type=Path, required=True)
    r.add_argument("--out", type=Path, required=True)
    r.add_argument("--policy", action="append", required=True, help="name=teacher or name=student:PATH")
    s = sub.add_parser("summary")
    s.add_argument("results", type=Path)
    args = p.parse_args()
    run(args) if args.cmd == "run" else summary(args)


if __name__ == "__main__":
    main()
