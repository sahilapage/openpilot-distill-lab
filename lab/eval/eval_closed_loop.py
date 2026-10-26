"""Closed-loop evaluation in the world model, paired across policies (same segments, same start points).

Needs the world-model server running (python -m rl.server). Resumable: finished (segment, start, policy) rows in --out
are skipped. Run from the repo root with PYTHONPATH=. so `rl` and `supervised` import.

  python ../eval/eval_closed_loop.py --segments val-file.txt --out results.jsonl \
      --policy straight=straight --policy teacher=teacher \
      --policy student=student:outputs/supervised/supercombo.pt \
      --policy dagger50=dagger:outputs/supervised/supercombo.pt:outputs/dagger/on_policy_step50.pt
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from evalkit import FPS, FRAME_SKIP, REF_FRAME, human_reference, integrate, rollout_metrics  # noqa: E402

from rl.actor import ConstantActor, SupercomboActor  # noqa: E402
from rl.env import Env, Episode, rollout  # noqa: E402
from rl.server import SERVER_URL, RuntimeClient  # noqa: E402


class Recorder:
    """Wraps an actor and records every (curvature, accel) it returns, in order."""

    def __init__(self, actor):
        self.actor = actor
        self.actions: list[tuple[float, float]] = []

    def act(self, frame=None, speed=0.0):
        action = self.actor.act(frame, speed)
        self.actions.append((float(action[0]), float(action[1])))
        return action

    @property
    def model_inputs(self):
        return self.actor.model_inputs

    @property
    def model_outputs(self):
        return self.actor.model_outputs


def build_actor(kind: str, paths: list[str]):
    if kind == "straight":
        return ConstantActor(0.0, 0.0)
    if kind == "teacher":
        return SupercomboActor.from_onnx()
    if kind == "student":
        return SupercomboActor.from_torch_checkpoint(paths[0])
    if kind == "dagger":
        return SupercomboActor.from_torch_checkpoint(paths[0], on_policy=paths[1])
    raise SystemExit(f"unknown policy kind {kind!r}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--segments", type=Path, required=True, help="file with one segment id per line")
    p.add_argument("--policy", action="append", required=True, help="name=kind[:path[:on_policy_path]]")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--starts-per-segment", type=int, default=6)
    p.add_argument("--rollout-frames", type=int, default=20, help="5 Hz frames per rollout (20 = 4 s)")
    p.add_argument("--min-speed", type=float, default=3.0, help="skip starts where the human is slower (m/s)")
    p.add_argument("--sampling-steps", type=int, default=15)
    p.add_argument("--cfg", type=float, default=2.0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--save-frames", type=int, default=0, help="save generated frames for the first K starts")
    p.add_argument("--frames-dir", type=Path, default=Path("outputs/eval/frames"))
    args = p.parse_args()

    rt = RuntimeClient(SERVER_URL, args.sampling_steps, args.cfg)
    max_frames = 50 - rt.history_frames - rt.future_frames
    if not 2 <= args.rollout_frames <= max_frames:
        raise SystemExit(f"--rollout-frames must be in [2, {max_frames}]")

    policies = {}
    for spec in args.policy:
        name, rest = spec.split("=", 1)
        kind, *paths = rest.split(":")
        policies[name] = (kind, paths)
    actors = {name: build_actor(kind, paths) for name, (kind, paths) in policies.items()}

    done = set()
    if args.out.exists():
        text = args.out.read_text()
        if text and not text.endswith("\n"):  # killed mid-write: drop the half line so appends start clean
            text = text[: text.rfind("\n") + 1]
            args.out.write_text(text)
        for line in text.splitlines():
            r = json.loads(line)
            done.add((r["segment"], r["start"], r["policy"]))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    segments = [s.strip() for s in args.segments.read_text().split() if s.strip()]
    t0 = time.time()
    n_done = 0
    with args.out.open("a") as out:
        for si, seg in enumerate(segments, 1):
            ep = Episode(rt, seg, cache_dir=Path("outputs/rl/cache"))
            last_start = len(ep.latents) - rt.future_frames - args.rollout_frames
            candidates = [s for s in range(rt.history_frames + 1, last_start + 1) if ep.speeds[s - 1] >= args.min_speed]
            rng = np.random.default_rng(args.seed * 100003 + int(seg[:8], 16))
            starts = sorted(rng.choice(candidates, min(args.starts_per_segment, len(candidates)), replace=False).tolist()) if candidates else []
            from safetensors.numpy import load_file

            loc = load_file(Path("data") / seg / "localizer.safetensors")
            for start in starts:
                human_pts, human_speed = human_reference(loc, start - 1, args.rollout_frames)
                for name, actor in actors.items():
                    if (seg, start, name) in done:
                        continue
                    if hasattr(actor, "reset"):  # ConstantActor is stateless
                        actor.reset()
                    rec = Recorder(actor)
                    env = Env(rt, ep, start * FRAME_SKIP, (start + args.rollout_frames) * FRAME_SKIP)
                    res = rollout(env, rec)
                    actions = rec.actions[: args.rollout_frames]
                    pts, speeds = integrate(float(ep.speeds[start - 1]), actions)
                    if abs(speeds[-1] - env.physics.speed) > 1e-6:
                        raise RuntimeError(f"integrator disagrees with env physics: {speeds[-1]} vs {env.physics.speed}")
                    row = {"segment": seg, "start": int(start), "policy": name, "frames": args.rollout_frames}
                    row |= rollout_metrics(pts, speeds, actions, human_pts, human_speed)
                    row["policy_path"] = np.round(pts, 2).tolist()
                    row["human_path"] = np.round(human_pts, 2).tolist()
                    row["ref_frame"] = REF_FRAME
                    row["actions"] = np.round(np.asarray(actions), 5).tolist()
                    if args.save_frames and starts.index(start) < args.save_frames:
                        args.frames_dir.mkdir(parents=True, exist_ok=True)
                        final = args.frames_dir / f"{seg}_{start}_{name}.npz"
                        tmp = final.with_name(final.stem + ".tmp.npz")
                        np.savez_compressed(
                            tmp,
                            frames=np.stack([np.asarray(f) for f in res.worldmodel_outputs["frames"]]),
                            plan=np.asarray(res.worldmodel_outputs["plan"]),
                        )
                        os.replace(tmp, final)  # frames land before their row, and never half-written
                    out.write(json.dumps(row) + "\n")
                    out.flush()
                    n_done += 1
            print(f"[{si}/{len(segments)}] {seg}: {len(starts)} starts, {n_done} rollouts, {(time.time() - t0) / 60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
