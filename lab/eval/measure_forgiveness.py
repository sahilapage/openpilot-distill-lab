"""How much of a commanded deviation does the world model actually render?

Runs comma's teacher (ONNX, CPU by default so it never competes for the GPU) as a measuring instrument over generated
rollouts saved by eval_shove.py / eval_closed_loop.py (--save-frames). For every generated frame it reads the teacher's
lane-line head and turns it into the car's heading relative to the RENDERED road and its offset from the rendered
ego-lane centre. The same two quantities are computed from physics (integrated actions) relative to the human path.
If the world renders deviations faithfully the two agree; if the generated road bends back under the car, the rendered
numbers stay near zero while physics keeps growing ("forgiveness").

  CUDA_VISIBLE_DEVICES= PYTHONPATH=. python ../eval/measure_forgiveness.py --results R.jsonl --frames DIR --out F.jsonl
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from evalkit import human_path  # noqa: E402
from openpilot.selfdrive.modeld.constants import ModelConstants  # noqa: E402

from rl.actor import SupercomboActor  # noqa: E402

X_IDXS = np.asarray(ModelConstants.X_IDXS)
NEAR = (X_IDXS >= 4.0) & (X_IDXS <= 20.0)  # range where the rendered lane lines are well resolved


def lane_geometry(lane_lines: np.ndarray) -> tuple[float, float]:
    """(heading_rad, offset_m) of the car relative to its rendered ego lane, RIGHT-positive (the repo frame).
    lane_lines: raw head output (528,) = mean (4 lines, 33 points, [y, z]) then log-std; device frame, y to the right."""
    mu = lane_lines[:264].reshape(4, 33, 2)
    left, right = mu[1, :, 0], mu[2, :, 0]  # inner lines: left and right of the ego lane
    centre = (left + right) / 2
    slope = np.polyfit(X_IDXS[NEAR], centre[NEAR], 1)[0]  # dy/dx of the lane centre, y right
    # car turned right of the road -> the road ahead appears to go left (slope < 0); lane centre to the left of the car
    # (centre < 0) -> the car is right of centre. Both reported right-positive, like physics and the actions.
    heading_right = -float(np.arctan(slope))
    offset_right = -float(np.interp(0.0, X_IDXS, centre))
    return heading_right, offset_right


def centre_line(lane_lines: np.ndarray) -> list[float]:
    """Rendered ego-lane centre y (device frame, right positive) at the 33 X_IDXS points."""
    mu = lane_lines[:264].reshape(4, 33, 2)
    return np.round((mu[1, :, 0] + mu[2, :, 0]) / 2, 3).tolist()


def physics_relative(row: dict) -> tuple[np.ndarray, np.ndarray]:
    """Heading (rad) and lateral offset (m) of the car relative to the human path, both right-positive, per pose index."""
    pts = np.asarray(row["policy_path"], float)
    hum = human_path(row)  # repo frame, whatever convention the row was written with
    acts = np.asarray(row["actions"], float)
    chord = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    heading = np.concatenate([[0.0], np.cumsum(chord * acts[: len(chord), 0])])
    hd = np.diff(hum, axis=0)
    hum_heading = np.concatenate([[0.0], np.arctan2(hd[:, 1], hd[:, 0])])
    rel_heading = heading - hum_heading
    # lateral offset: signed distance from the human point at the same time, measured across the human's heading
    n = np.stack([-np.sin(hum_heading), np.cos(hum_heading)], axis=1)
    rel_offset = np.sum((pts - hum) * n, axis=1)
    return rel_heading, rel_offset


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--results", type=Path, required=True)
    p.add_argument("--frames", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--providers", default="CPUExecutionProvider")
    args = p.parse_args()

    teacher = SupercomboActor.from_onnx(providers=args.providers.split(","))
    rows = [json.loads(line) for line in args.results.read_text().splitlines() if line.strip()]
    with args.out.open("w") as out:
        for row in rows:
            suffix = f"_{row['kick']}" if "kick" in row else ""
            npz = args.frames / f"{row['segment']}_{row['start']}_{row['policy']}{suffix}.npz"
            if not npz.exists():
                continue
            frames = np.load(npz)["frames"]
            phys_h, phys_o = physics_relative(row)
            speeds = np.linalg.norm(np.diff(np.asarray(row["policy_path"], float), axis=0), axis=1) * 5
            teacher.reset()
            rend_h, rend_o, rend_c = [], [], []
            for k, frame in enumerate(frames):
                teacher.act(frame, float(speeds[min(k, len(speeds) - 1)]))
                out_k = teacher.model_outputs
                if out_k is None:
                    rend_h.append(None)
                    rend_o.append(None)
                    rend_c.append(None)
                    continue
                ll = out_k["lane_lines"][0].float().numpy()
                rend_c.append(centre_line(ll))
                h, o = lane_geometry(ll)
                rend_h.append(h)
                rend_o.append(o)
            rec = {k: row[k] for k in ("segment", "start", "policy") if k in row}
            rec |= {k: row[k] for k in ("kick", "pull_dps", "mode") if k in row}
            # frame k shows the world after action k, i.e. pose index k+1
            rec["physics_heading_deg"] = np.round(np.rad2deg(phys_h[1 : len(frames) + 1]), 2).tolist()
            rec["physics_offset_m"] = np.round(phys_o[1 : len(frames) + 1], 2).tolist()
            rec["rendered_heading_deg"] = [None if v is None else round(float(np.rad2deg(v)), 2) for v in rend_h]
            rec["rendered_offset_m"] = [None if v is None else round(v, 2) for v in rend_o]
            rec["rendered_centre_y"] = rend_c
            rec["executed_curvature"] = np.round(np.asarray(row["actions"], float)[: len(frames), 0], 6).tolist()
            out.write(json.dumps(rec) + "\n")
            out.flush()
            print(rec["policy"], rec.get("kick", ""), "phys hdg", rec["physics_heading_deg"][::4],
                  "rend hdg", rec["rendered_heading_deg"][::4], flush=True)


if __name__ == "__main__":
    main()
