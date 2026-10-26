"""Summarise measure_forgiveness.py output against the real-frame baseline from calibrate_forgiveness.py.

Rendering gain = slope of (rendered - real baseline) on physics, over all frames, for heading relative to the road.
1 = the world shows the car's deviation as it physically is; 0 = the generated road re-aligns under the car completely.
Bootstrap CI over rollouts.

  python analyze_forgiveness.py forgiveness.jsonl calib.jsonl
"""
import json
import sys

import numpy as np


def load(path):
    return [json.loads(line) for line in open(path) if line.strip()]


def main() -> None:
    meas, calib = load(sys.argv[1]), load(sys.argv[2])
    base = {(c["segment"], c["start"]): c for c in calib}
    real_h = np.array([v for c in calib for v in c["real_heading_deg"] if v is not None])
    print(f"instrument on REAL frames: heading {real_h.mean():+.2f} ± {real_h.std():.2f} deg (n={len(real_h)} frames)")
    per = []  # (physics, rendered - baseline) per rollout
    for m in meas:
        c = base.get((m["segment"], m["start"]))
        ph, rh = np.array(m["physics_heading_deg"], float), m["rendered_heading_deg"]
        bh = c["real_heading_deg"] if c else [None] * len(rh)
        keep = [k for k in range(min(len(rh), len(bh), len(ph))) if rh[k] is not None and bh[k] is not None]
        x = ph[keep]
        y = np.array([rh[k] - bh[k] for k in keep])
        per.append((m["policy"], m.get("kick", ""), x, y))
        print(f"{m['policy']:>10} {m.get('kick', ''):1}  physics heading end {x[-1]:+6.1f} deg   rendered (minus real) end {y[-1]:+6.1f} deg"
              f"   max|physics| {np.abs(x).max():5.1f}  max|rendered| {np.abs(y).max():5.1f}")

    def gain(items):
        x = np.concatenate([i[2] for i in items])
        y = np.concatenate([i[3] for i in items])
        return float(np.sum(x * y) / np.sum(x * x))  # through the origin: no deviation -> no rendered deviation

    g = gain(per)
    rng = np.random.default_rng(0)
    boots = [gain([per[i] for i in rng.integers(len(per), size=len(per))]) for _ in range(2000)]
    lo, hi = np.percentile(boots, [2.5, 97.5])
    print(f"\nrendering gain (heading vs road): {g:.2f}  [95% CI {lo:.2f}, {hi:.2f}]  over {len(per)} rollouts")
    print("1.0 = deviations rendered faithfully, 0.0 = the generated road fully re-aligns under the car")


if __name__ == "__main__":
    main()
