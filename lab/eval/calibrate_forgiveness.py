"""Instrument baseline for measure_forgiveness.py: the same lane-geometry readout on the REAL drive (logged latents decoded
through the same VAE the world model uses), over the same time windows as the rollouts. On real frames the car follows
the human path by definition, so this gives the instrument's bias and noise. Needs the world-model server (decode only).

  CUDA_VISIBLE_DEVICES= PYTHONPATH=. python ../eval/calibrate_forgiveness.py --results R.jsonl --out C.jsonl
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from measure_forgiveness import centre_line, lane_geometry  # noqa: E402

from rl.actor import SupercomboActor  # noqa: E402
from rl.env import Episode  # noqa: E402
from rl.server import SERVER_URL, RuntimeClient  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--results", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--providers", default="CPUExecutionProvider")
    args = p.parse_args()

    rt = RuntimeClient(SERVER_URL, 15, 2.0)
    teacher = SupercomboActor.from_onnx(providers=args.providers.split(","))
    rows = [json.loads(line) for line in args.results.read_text().splitlines() if line.strip()]
    windows = sorted({(r["segment"], r["start"], r["frames"]) for r in rows})
    with args.out.open("w") as out:
        for seg, start, n in windows:
            ep = Episode(rt, seg, cache_dir=Path("outputs/rl/cache"))
            frames = rt.decode(ep.latents[start : start + n])  # frame k of a rollout shows time start + k
            speeds = ep.speeds[start : start + n]
            teacher.reset()
            hs, os_, cs = [], [], []
            for k, frame in enumerate(frames):
                teacher.act(np.asarray(frame), float(speeds[k]))
                if teacher.model_outputs is None:
                    hs.append(None)
                    os_.append(None)
                    cs.append(None)
                    continue
                ll = teacher.model_outputs["lane_lines"][0].float().numpy()
                cs.append(centre_line(ll))
                h, o = lane_geometry(ll)
                hs.append(round(float(np.rad2deg(h)), 2))
                os_.append(round(o, 2))
            out.write(json.dumps({"segment": seg, "start": start, "real_heading_deg": hs, "real_offset_m": os_,
                                 "real_centre_y": cs}) + "\n")
            out.flush()
            print(seg[:8], start, "real hdg", hs[::4], "real off", os_[::4], flush=True)


if __name__ == "__main__":
    main()
