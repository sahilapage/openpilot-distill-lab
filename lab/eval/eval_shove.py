"""The shove test: can a policy get back on the road after being knocked off it?

Two disturbances. `shove` (default): a one-off heading kick, below. `pull` (--pull-dps > 0): the steering pulls to one side
for the whole rollout (a bent steering rack / crosswind), adding a constant yaw rate on top of whatever the policy asks
for, so the policy has to keep counter-steering. Calibration showed the world model re-centres the scene within a frame
once a kick ends (the expert plan drops back to ~0), so after a shove nobody, not even the teacher, sees a reason to
correct; the sustained pull keeps the disturbance visible in every frame.

For the first KICK_FRAMES steps of a rollout the steering is forced to turn the car by KICK_DEG in total (left and right
kicks, speed independent), while the policy still sees every frame. Then the policy takes over and has to bring the
car back to the human's path. DAgger is trained on exactly this situation (states the policy's own mistakes lead to),
supervised cloning is not, so this is the sharpest test of what DAgger buys.

Anchor test (--anchor-frames 10,20,max): the world model is conditioned at every step on the REAL drive's frames just
after the rollout end (rl/env.py `future`), which show the car in the human's lane. To see whether that anchor pulls the
rendered scene back, the rollout length stays --rollout-frames while the anchor sits A frames after the start; keys
become e.g. "L@20" so frames and rows of every anchor are kept apart. Default (no flag): anchor at the rollout end.

Same server, actors and metrics as eval_closed_loop.py. Resumable: finished (segment, start, policy, kick) rows are skipped.

  python ../eval/eval_shove.py --segments ../eval_segments.txt --out ../results_shove.jsonl --save-frames 1 \
      --policy straight=straight --policy student=student:outputs/supervised/supercombo.pt \
      --policy dagger1400=dagger:outputs/supervised/supercombo.pt:outputs/dagger/on_policy_step1400.pt
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from eval_closed_loop import Recorder, build_actor  # noqa: E402
from evalkit import FPS, FRAME_SKIP, REF_FRAME, human_reference, integrate, rollout_metrics  # noqa: E402

from rl.env import Env, Episode, rollout  # noqa: E402
from rl.server import SERVER_URL, RuntimeClient  # noqa: E402


class KickActor:
    """Runs the real actor every step (so its history stays correct) but overrides the steering for the first frames."""

    def __init__(self, actor, kick_deg: float, kick_frames: int, pull_dps: float = 0.0):
        self.actor, self.kick_rad, self.kick_frames, self.i = actor, np.deg2rad(kick_deg), kick_frames, 0
        self.pull = np.deg2rad(pull_dps)  # signed yaw rate added every step, rad/s

    def act(self, frame=None, speed=0.0):
        curvature, accel = self.actor.act(frame, speed)
        if self.pull:
            return curvature + self.pull / max(speed, 1.0), accel  # yaw rate = speed * curvature
        if self.i < self.kick_frames:
            self.i += 1
            # per-step yaw = speed * curvature / FPS  ->  curvature that turns kick_rad / kick_frames in one step
            curvature = self.kick_rad / self.kick_frames * FPS / max(speed, 1.0)
        return curvature, accel

    @property
    def model_inputs(self):
        return self.actor.model_inputs

    @property
    def model_outputs(self):
        return self.actor.model_outputs


def recovery_metrics(pts, human_pts, kick_frames: int) -> dict:
    err = np.linalg.norm(pts[1:] - human_pts[1:], axis=1)
    tail = err[-FPS:]  # last second
    over = np.nonzero(err[kick_frames:] > 1.0)[0]
    return {
        "peak_err": float(err.max()),
        "tail_err": float(tail.mean()),
        "recovered": bool(tail.max() < 1.0),  # back within 1 m of the human path for the final second
        "t_recover": float((over.max() + 1 + kick_frames) / FPS) if len(over) else float(kick_frames / FPS),
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--segments", type=Path, required=True)
    p.add_argument("--policy", action="append", required=True, help="name=kind[:path[:on_policy_path]]")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--starts-per-segment", type=int, default=3)
    p.add_argument("--rollout-frames", type=int, default=25, help="5 Hz frames per rollout (25 = 5 s)")
    p.add_argument("--kick-deg", type=float, default=15.0, help="heading change forced into the car, left and right")
    p.add_argument("--kick-frames", type=int, default=3)
    p.add_argument("--pull-dps", type=float, default=0.0, help="pull mode: constant yaw rate added, deg/s (0 = shove mode)")
    p.add_argument("--min-speed", type=float, default=5.0)
    p.add_argument("--sampling-steps", type=int, default=15)
    p.add_argument("--cfg", type=float, default=2.0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--save-frames", type=int, default=0, help="save generated frames for the first K starts")
    p.add_argument("--frames-dir", type=Path, default=Path("outputs/shove/frames"))
    p.add_argument("--anchor-frames", default="", help="comma list of anchor distances in 5 Hz frames ('max' = the most "
                   "the world model allows); default: anchor at the rollout end")
    args = p.parse_args()

    rt = RuntimeClient(SERVER_URL, args.sampling_steps, args.cfg)
    max_frames = 50 - rt.history_frames - rt.future_frames
    if not args.kick_frames + 5 <= args.rollout_frames <= max_frames:
        raise SystemExit(f"--rollout-frames must be in [{args.kick_frames + 5}, {max_frames}]")
    anchors = [None]
    if args.anchor_frames:
        anchors = [max_frames if a == "max" else int(a) for a in args.anchor_frames.split(",")]
        if not all(args.rollout_frames <= a <= max_frames for a in anchors):
            raise SystemExit(f"--anchor-frames must be in [{args.rollout_frames}, {max_frames}]")
        print(f"anchor test: rollouts of {args.rollout_frames} frames, anchors {anchors} (max {max_frames})", flush=True)
    span = max(a or args.rollout_frames for a in anchors)

    policies = {}
    for spec in args.policy:
        name, rest = spec.split("=", 1)
        kind, *paths = rest.split(":")
        policies[name] = (kind, paths)
    actors = {name: build_actor(kind, paths) for name, (kind, paths) in policies.items()}
    pull = args.pull_dps > 0
    # keys are file/resume names only. "L" carries the positive value, which is a RIGHT turn in the repo frame (y right);
    # the row's "side" field says the physical side. Kept as is so interrupted runs resume onto the same keys.
    kicks = {"L": args.pull_dps if pull else args.kick_deg, "R": -(args.pull_dps if pull else args.kick_deg)}
    kick_frames = 0 if pull else args.kick_frames

    done = set()
    if args.out.exists():
        text = args.out.read_text()
        if text and not text.endswith("\n"):  # killed mid-write: drop the half line so appends start clean
            text = text[: text.rfind("\n") + 1]
            args.out.write_text(text)
        for line in text.splitlines():
            r = json.loads(line)
            done.add((r["segment"], r["start"], r["policy"], r["kick"]))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    segments = [s.strip() for s in args.segments.read_text().split() if s.strip()]
    t0 = time.time()
    n_done = 0
    with args.out.open("a") as out:
        for si, seg in enumerate(segments, 1):
            ep = Episode(rt, seg, cache_dir=Path("outputs/rl/cache"))
            last_start = len(ep.latents) - rt.future_frames - span
            candidates = [s for s in range(rt.history_frames + 1, last_start + 1) if ep.speeds[s - 1] >= args.min_speed]
            rng = np.random.default_rng(args.seed * 100003 + int(seg[:8], 16))
            starts = sorted(rng.choice(candidates, min(args.starts_per_segment, len(candidates)), replace=False).tolist()) if candidates else []
            from safetensors.numpy import load_file

            loc = load_file(Path("data") / seg / "localizer.safetensors")
            for start in starts:
                human_pts, human_speed = human_reference(loc, start - 1, args.rollout_frames)
                for (kbase, kdeg), anchor in [(k, a) for k in kicks.items() for a in anchors]:
                    kname = kbase if anchor is None else f"{kbase}@{anchor}"
                    for name, actor in actors.items():
                        if (seg, start, name, kname) in done:
                            continue
                        if hasattr(actor, "reset"):  # ConstantActor is stateless
                            actor.reset()
                        rec = Recorder(KickActor(actor, kdeg, args.kick_frames, kdeg if pull else 0.0))
                        env = Env(rt, ep, start * FRAME_SKIP, (start + (anchor or args.rollout_frames)) * FRAME_SKIP)
                        env.end = (start + args.rollout_frames) * FRAME_SKIP  # stop here; the anchor stays where it was set
                        res = rollout(env, rec)
                        actions = rec.actions[: args.rollout_frames]
                        pts, speeds = integrate(float(ep.speeds[start - 1]), actions)
                        if abs(speeds[-1] - env.physics.speed) > 1e-6:
                            raise RuntimeError(f"integrator disagrees with env physics: {speeds[-1]} vs {env.physics.speed}")
                        row = {"segment": seg, "start": int(start), "policy": name, "kick": kname, "mode": "pull" if pull else "shove",
                               "kick_deg": 0.0 if pull else kdeg, "pull_dps": kdeg if pull else 0.0,
                               "kick_frames": kick_frames, "frames": args.rollout_frames,
                               "anchor_frames": anchor or args.rollout_frames}
                        row |= rollout_metrics(pts, speeds, actions, human_pts, human_speed)
                        row |= recovery_metrics(pts, human_pts, kick_frames)
                        row["policy_path"] = np.round(pts, 2).tolist()
                        row["human_path"] = np.round(human_pts, 2).tolist()
                        row["ref_frame"] = REF_FRAME
                        row["side"] = "right" if kdeg > 0 else "left"  # + curvature is a RIGHT turn in the repo frame
                        row["actions"] = np.round(np.asarray(actions), 5).tolist()
                        if args.save_frames and starts.index(start) < args.save_frames:
                            args.frames_dir.mkdir(parents=True, exist_ok=True)
                            final = args.frames_dir / f"{seg}_{start}_{name}_{kname}.npz"
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
