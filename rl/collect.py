"""Collect world-model rollouts for on-policy distillation (DAgger).

A driver (your student, or the openpilot teacher) drives for a few seconds inside the world model, starting from a
random moment of a random logged segment. In most rollouts a constant steering pull is added on top of what the driver
commands (DART-style noise), so the car drifts off-centre and the data covers the states a policy must recover from.

Every rollout is saved as <out-dir>/r{index:06d}.npz: the world model's frames (the 9 real history frames, then every
generated frame, exactly what the policy sees in closed loop) plus a little metadata. No labels are stored: training
runs the teacher on the same frames online (WM_DATA=<out-dir> python -m supervised.train ...).
Rollout i is seeded by (seed, i), so a killed run resumes exactly where it stopped.

  python -m rl.server &
  python -m rl.collect --segments train.txt --out-dir outputs/wm_data/iter1 --rollouts 600 \\
      --driver student=student:outputs/supervised/supercombo.pt --driver teacher=teacher --driver-weights 0.5,0.5
"""

import argparse
import io
import json
import os
import time
from pathlib import Path

import numpy as np

from rl.actor import SupercomboActor
from rl.env import ENCODE_BATCH_SIZE, FRAME_SKIP, ROLLOUT_FRAMES, Env, Episode, rollout
from rl.server import SERVER_URL, RuntimeClient


class PullActor:
    """Adds a constant yaw rate (rad/s) to the actor's steering from frame `onset` on."""

    def __init__(self, actor, pull: float, onset: int):
        self.actor, self.pull, self.onset, self.i = actor, pull, onset, 0

    def act(self, frame=None, speed=0.0):
        curvature, accel = self.actor.act(frame, speed)
        self.i += 1
        if self.i > self.onset:
            curvature += self.pull / max(speed, 1.0)
        return curvature, accel

    @property
    def model_inputs(self):
        return self.actor.model_inputs

    @property
    def model_outputs(self):
        return self.actor.model_outputs


def build_actor(spec: str) -> SupercomboActor:
    kind, *path = spec.split(":", 1)
    if kind == "teacher":
        return SupercomboActor.from_onnx()
    if kind == "student":
        return SupercomboActor.from_torch_checkpoint(path[0])
    raise SystemExit(f"unknown driver {spec!r} (use teacher or student:PATH)")


def left_hand_traffic(segment: str) -> bool:
    import reverse_geocoder
    from openpilot.common.transformations.coordinates import ecef2geodetic
    from safetensors.numpy import load_file

    from supervised.dataset import LHT_COUNTRIES

    loc = load_file(f"data/{segment}/localizer.safetensors")
    lat, lon, _ = ecef2geodetic(loc["frame_states"][-1, 0:3])
    return reverse_geocoder.search((lat, lon), mode=1, verbose=False)[0]["cc"] in LHT_COUNTRIES


def save(path: Path, frames: np.ndarray, meta: dict) -> None:
    buf = io.BytesIO()
    np.savez(buf, frames=frames, meta=np.frombuffer(json.dumps(meta).encode(), np.uint8))
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_bytes(buf.getvalue())
    os.replace(tmp, path)  # atomic: a kill never leaves a half-written rollout


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--segments", type=Path, required=True, help="training segments only, one id per line")
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--rollouts", type=int, required=True)
    p.add_argument("--driver", action="append", required=True, help="name=teacher or name=student:PATH")
    p.add_argument("--driver-weights", help="comma list, one per --driver (default: equal)")
    p.add_argument("--rollout-frames", type=int, default=25, help="5 Hz frames per rollout")
    p.add_argument("--pull-prob", type=float, default=0.75)
    p.add_argument("--pull-min-dps", type=float, default=0.5)
    p.add_argument("--pull-max-dps", type=float, default=3.0)
    p.add_argument("--onset-max", type=int, default=8)
    p.add_argument("--min-speed", type=float, default=5.0)
    p.add_argument("--sampling-steps", type=int, default=15)
    p.add_argument("--cfg", type=float, default=2.0)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    rt = RuntimeClient(SERVER_URL, args.sampling_steps, args.cfg)
    if not 2 <= args.rollout_frames <= ROLLOUT_FRAMES - rt.history_frames - rt.future_frames:
        raise SystemExit("--rollout-frames out of range for the world model's context")
    drivers = {name: build_actor(spec) for name, spec in (d.split("=", 1) for d in args.driver)}
    names = list(drivers)
    weights = np.ones(len(names)) if args.driver_weights is None else np.array([float(w) for w in args.driver_weights.split(",")])
    weights = weights / weights.sum()

    segments = args.segments.read_text().split()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    episodes, lht = {}, {}
    t0, made = time.time(), 0
    for i in range(args.rollouts):
        path = args.out_dir / f"r{i:06d}.npz"
        if path.exists():
            continue
        rng = np.random.default_rng([args.seed, i])
        seg = segments[int(rng.integers(len(segments)))]
        if seg not in episodes:
            episodes[seg] = Episode(rt, seg, encode_batch_size=ENCODE_BATCH_SIZE, cache_dir=Path("outputs/rl/cache"))
            lht[seg] = left_hand_traffic(seg)
        ep = episodes[seg]
        last_start = len(ep.latents) - rt.future_frames - args.rollout_frames
        starts = [s for s in range(rt.history_frames + 1, last_start + 1) if ep.speeds[s - 1] >= args.min_speed]
        if not starts:
            save(path, np.zeros((0, 128, 256, 6), np.uint8), {"segment": seg, "skipped": "no fast start"})
            continue
        start = int(starts[int(rng.integers(len(starts)))])
        name = names[int(rng.choice(len(names), p=weights))]
        pull_dps, onset = 0.0, 0
        if rng.random() < args.pull_prob:
            pull_dps = float(rng.uniform(args.pull_min_dps, args.pull_max_dps) * rng.choice([-1.0, 1.0]))
            onset = int(rng.integers(0, args.onset_max + 1))

        actor = drivers[name]
        actor.reset()
        driver = PullActor(actor, np.deg2rad(pull_dps), onset) if pull_dps else actor
        env = Env(rt, ep, start * FRAME_SKIP, (start + args.rollout_frames) * FRAME_SKIP)
        history = np.asarray(rt.decode(ep.latents[start - rt.history_frames : start]))
        result = rollout(env, driver)
        frames = np.concatenate((history, np.stack([np.asarray(f) for f in result.worldmodel_outputs["frames"]])))
        save(path, frames, {"segment": seg, "start": start, "driver": name, "pull_dps": pull_dps, "onset": onset,
                            "lht": lht[seg], "n_history": len(history)})
        made += 1
        if made == 1 or made % 10 == 0:
            print(f"[{i + 1}/{args.rollouts}] {seg[:8]} {name} pull {pull_dps:+.1f} deg/s, "
                  f"{(time.time() - t0) / made:.1f} s/rollout", flush=True)
    print(f"done: {made} new rollouts in {(time.time() - t0) / 60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
