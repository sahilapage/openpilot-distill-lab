"""Open-loop check: do the policies steer like the human on the REAL drive, with no world model in the loop?

Every policy watches each held-out drive frame by frame, twice: once on the raw camera frames (warped exactly as rl.env
Episode does before encoding) and once on the same frames passed through the world model's VAE (what the policies see
in closed loop, minus the closed loop). Its (curvature, accel) at every 5 Hz frame is stored next to the human's own,
taken from the localizer gyro + speed (evalkit.sensor_actions, repo frame: positive curvature = right turn).

  student steers on raw frames, not on VAE frames  -> the VAE look is the problem
  student steers on both open loop, not closed loop -> the closed loop is the problem
  student does not steer even on raw frames         -> distillation never taught it to take curves
Needs the world-model server (encode/decode only). Resumable: finished (segment, policy, source) rows are skipped.

  PYTHONPATH=. python ../eval/eval_open_loop.py --segments S.txt --out O.jsonl --policy teacher=teacher \
      --policy student=student:outputs/supervised/supercombo.pt --policy d800=dagger:SUP:outputs/dagger/on_policy_step800.pt
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from eval_closed_loop import build_actor  # noqa: E402
from evalkit import REF_FRAME, sensor_actions  # noqa: E402

from rl.env import Episode  # noqa: E402
from rl.server import SERVER_URL, RuntimeClient  # noqa: E402


class CaptureEncode:
    """Runtime stand-in for Episode: keeps the warped camera frames it is asked to encode."""

    def __init__(self, rt: RuntimeClient):
        self.rt, self.vae, self.frames = rt, rt.vae, []

    def encode(self, frames: np.ndarray):
        self.frames.append(np.asarray(frames).copy())
        return self.rt.encode(frames)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--segments", type=Path, required=True)
    p.add_argument("--policy", action="append", required=True, help="name=kind[:path[:on_policy_path]]")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--decode-batch", type=int, default=24)
    args = p.parse_args()

    rt = RuntimeClient(SERVER_URL, 15, 2.0)
    specs = [s.split("=", 1) for s in args.policy]
    actors = {}
    for name, spec in specs:
        kind, *paths = spec.split(":")
        actors[name] = build_actor(kind, paths)
    segments = args.segments.read_text().split()

    done = set()
    if args.out.exists():
        lines = args.out.read_text().splitlines()
        good = []
        for line in lines:
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                break  # a half-written last line from a kill
            good.append(line)
            done.add((r["segment"], r["policy"], r["source"]))
        args.out.write_text("".join(g + "\n" for g in good))

    from safetensors.numpy import load_file

    t0 = time.time()
    with args.out.open("a") as out:
        for si, seg in enumerate(segments, 1):
            todo = [(n, s) for n in actors for s in ("raw", "vae") if (seg, n, s) not in done]
            if not todo:
                continue
            cap = CaptureEncode(rt)
            ep = Episode(cap, seg, cache_dir=None)  # no cache: forces decode + warp, so the raw frames are captured
            raw = np.concatenate(cap.frames)
            vae = np.concatenate([rt.decode(ep.latents[i : i + args.decode_batch])
                                  for i in range(0, len(ep.latents), args.decode_batch)])
            if raw.shape != vae.shape or raw.dtype != vae.dtype:
                raise RuntimeError(f"raw {raw.shape} {raw.dtype} vs vae {vae.shape} {vae.dtype}: frame formats differ")
            n = len(raw) - 1
            loc = load_file(Path("data") / seg / "localizer.safetensors")
            human = sensor_actions(loc, 0, n)
            speeds = np.asarray(ep.speeds, float)
            for name, source in todo:
                frames = raw if source == "raw" else vae
                actor = actors[name]
                if hasattr(actor, "reset"):
                    actor.reset()
                acts = [actor.act(np.asarray(frames[t]), float(speeds[t])) for t in range(n)]
                row = {"segment": seg, "policy": name, "source": source, "ref_frame": REF_FRAME,
                       "actions": np.round(np.asarray(acts, float), 6).tolist(),
                       "human_actions": np.round(human, 6).tolist(), "speeds": np.round(speeds[: n + 1], 3).tolist()}
                out.write(json.dumps(row) + "\n")
                out.flush()
            print(f"[{si}/{len(segments)}] {seg[:8]}: {n} frames x {len(todo)} runs, {(time.time() - t0) / 60:.1f} min",
                  flush=True)


if __name__ == "__main__":
    main()
