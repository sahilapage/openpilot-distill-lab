"""DAgger v2, step 1: collect world-model rollouts for full-network on-policy distillation.

Each rollout starts at a random 5 Hz frame of a random TRAIN drive. A driver picked per rollout (--driver name=kind:path
with --driver-weights; usually the current student, sometimes comma's teacher) drives for --rollout-frames frames inside
comma's world model. In --pull-prob of the rollouts a constant steering pull (a uniform random yaw rate, either side,
from a random onset frame) is added to whatever the driver commands, DART style, so the drives go off-centre and the
data contains the states a student must recover from.

Saved per rollout (outputs atomic, <out-dir>/r{index:06d}.npz):
  frames   uint8 (history + rollout, 128, 256, 6): the world model's VAE decode of the 9 real history frames, then every
           generated frame; narrow camera RGB in [..., :3], wide in [..., 3:], exactly what the actor sees in closed loop
  meta     json: segment, start, driver, pull_dps, onset, lht (left-hand traffic), speeds, n_history
No labels are stored: training runs comma's teacher on the same frames online (supervised/train.py with WM_DATA).
Rollout i uses its own seed (seed, i), so a killed run resumes exactly where it stopped.

  PYTHONPATH=. python collect_wm.py --segments TRAIN.txt --out-dir outputs/wm_data/iter1 --rollouts 600 \
      --driver student=student:outputs/supervised_more/supercombo.pt --driver teacher=teacher --driver-weights 0.5,0.5
"""
import argparse
import io
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from eval_closed_loop import build_actor  # noqa: E402
from train_dagger import PullActor  # noqa: E402

from rl.env import ENCODE_BATCH_SIZE, FRAME_SKIP, ROLLOUT_FRAMES, Env, Episode, rollout  # noqa: E402
from rl.server import SERVER_URL, RuntimeClient  # noqa: E402


def lht_of(segment: str) -> bool:
    """Left-hand traffic, decided the way supervised.dataset does it (reverse geocode of the drive's last position)."""
    import reverse_geocoder
    from openpilot.common.transformations.coordinates import ecef2geodetic
    from safetensors.numpy import load_file

    from supervised.dataset import LHT_COUNTRIES

    loc = load_file(f"data/{segment}/localizer.safetensors")
    lat, lon, _ = ecef2geodetic(loc["frame_states"][-1, 0:3])
    return reverse_geocoder.search((lat, lon), mode=1, verbose=False)[0]["cc"] in LHT_COUNTRIES


def atomic_savez(path: Path, **arrays) -> None:
    buf = io.BytesIO()
    np.savez(buf, **arrays)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_bytes(buf.getvalue())
    os.replace(tmp, path)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--segments", type=Path, required=True, help="TRAIN drives only (never the eval drives)")
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--rollouts", type=int, required=True)
    p.add_argument("--driver", action="append", required=True, help="name=kind[:path]")
    p.add_argument("--driver-weights", default=None, help="comma list, same order as --driver (default: equal)")
    p.add_argument("--rollout-frames", type=int, default=25)
    p.add_argument("--pull-prob", type=float, default=0.75)
    p.add_argument("--pull-min-dps", type=float, default=0.5)
    p.add_argument("--pull-max-dps", type=float, default=3.0)
    p.add_argument("--onset-max", type=int, default=8, help="pull starts at a uniform frame in [0, onset-max]")
    p.add_argument("--min-speed", type=float, default=5.0)
    p.add_argument("--sampling-steps", type=int, default=15)
    p.add_argument("--cfg", type=float, default=2.0)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    rt = RuntimeClient(SERVER_URL, args.sampling_steps, args.cfg)
    max_frames = ROLLOUT_FRAMES - rt.history_frames - rt.future_frames
    if not 2 <= args.rollout_frames <= max_frames:
        raise SystemExit(f"--rollout-frames must be in [2, {max_frames}]")
    drivers = {}
    for spec in args.driver:
        name, rest = spec.split("=", 1)
        kind, *paths = rest.split(":")
        drivers[name] = build_actor(kind, paths)
    names = list(drivers)
    weights = np.ones(len(names)) if args.driver_weights is None else np.asarray([float(w) for w in args.driver_weights.split(",")])
    if len(weights) != len(names):
        raise SystemExit("--driver-weights needs one weight per --driver")
    weights = weights / weights.sum()

    segments = [s.strip() for s in args.segments.read_text().split() if s.strip()]
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "config.json").write_text(json.dumps({k: str(v) for k, v in vars(args).items()}, indent=1))
    episodes: dict[str, Episode] = {}
    lht: dict[str, bool] = {}
    t0, made = time.time(), 0
    for i in range(args.rollouts):
        path = args.out_dir / f"r{i:06d}.npz"
        if path.exists():
            continue
        rng = np.random.default_rng([args.seed, i])
        seg = segments[int(rng.integers(len(segments)))]
        if seg not in episodes:
            episodes[seg] = Episode(rt, seg, encode_batch_size=ENCODE_BATCH_SIZE, cache_dir=Path("outputs/rl/cache"))
            lht[seg] = lht_of(seg)
        ep = episodes[seg]
        last_start = len(ep.latents) - rt.future_frames - args.rollout_frames
        candidates = [s for s in range(rt.history_frames + 1, last_start + 1) if ep.speeds[s - 1] >= args.min_speed]
        if not candidates:
            atomic_savez(path, frames=np.zeros((0, 128, 256, 6), np.uint8),
                         meta=np.frombuffer(json.dumps({"segment": seg, "skipped": "no fast start"}).encode(), np.uint8))
            continue
        start = int(candidates[int(rng.integers(len(candidates)))])
        name = names[int(rng.choice(len(names), p=weights))]
        pull_dps, onset = 0.0, 0
        if rng.random() < args.pull_prob:
            pull_dps = float(rng.uniform(args.pull_min_dps, args.pull_max_dps) * rng.choice([-1.0, 1.0]))
            onset = int(rng.integers(0, args.onset_max + 1))
        actor = drivers[name]
        actor.reset()
        driver = PullActor(actor, np.deg2rad(pull_dps), onset) if pull_dps else actor
        env = Env(rt, ep, start * FRAME_SKIP, (start + args.rollout_frames) * FRAME_SKIP)
        speeds = [float(env.physics.speed)]
        history = rt.decode(ep.latents[start - rt.history_frames : start])
        result = rollout(env, driver)
        speeds.append(float(env.physics.speed))
        frames = np.concatenate((np.asarray(history), np.stack([np.asarray(f) for f in result.worldmodel_outputs["frames"]])))
        if frames.dtype != np.uint8 or frames.shape[1:] != (128, 256, 6):
            raise RuntimeError(f"unexpected frame format {frames.shape} {frames.dtype}")
        meta = {"segment": seg, "start": start, "driver": name, "pull_dps": pull_dps, "onset": onset, "lht": lht[seg],
                "n_history": int(len(history)), "speed_start": speeds[0], "speed_end": speeds[-1]}
        atomic_savez(path, frames=frames, meta=np.frombuffer(json.dumps(meta).encode(), np.uint8))
        made += 1
        if made % 10 == 0 or made == 1:
            rate = (time.time() - t0) / made
            print(f"[{i + 1}/{args.rollouts}] {seg[:8]} {name} pull {pull_dps:+.1f} {rate:.1f} s/rollout", flush=True)
    print(f"done: {made} new rollouts in {(time.time() - t0) / 60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
