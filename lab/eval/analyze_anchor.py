"""Anchor test: does the real-future anchor pull the world model's scene back onto the human's lane?

Input: measure_forgiveness.py output for eval_shove.py --anchor-frames runs (kick keys "L@10", "R@31", ...) and
calibrate_forgiveness.py output (the same lane readout on the real frames, the instrument baseline).

For `straight` the executed motion is identical under every anchor (same pull, no policy), so any difference in what is
rendered is caused by the anchor alone. Per anchor and time step, pull-normalised (+ = toward the pull):
  rendered offset  - real,  rendered heading - real     (what the scene shows)
  physics offset,           physics heading             (what really happened)
Then paired differences far-anchor minus near-anchor at each time, cluster bootstrap over drives: if the anchor pulls the
scene back, the near anchor shows LESS drift (difference > 0). For the teacher, the motion differs by anchor (it reacts
to what it sees): final drift per anchor tells whether a near anchor makes it correct less.

  python -I analyze_anchor.py forgiveness_anchor.jsonl calib_anchor.jsonl [results_anchor.jsonl]
"""
import json
import sys
from collections import defaultdict

import numpy as np

DT = 0.2


def load(path):
    return [json.loads(line) for line in open(path) if line.strip()]


def main() -> None:
    meas, calib = load(sys.argv[1]), load(sys.argv[2])
    base = {(c["segment"], c["start"]): c for c in calib}
    rng = np.random.default_rng(0)
    # series[(policy, anchor)][(segment, start, side)] = array (T, 4): phys off, rend off-real, phys hdg, rend hdg-real
    series = defaultdict(dict)
    for m in meas:
        side, anchor = m["kick"].split("@")
        c = base[(m["segment"], m["start"])]
        sg = 1.0 if m["pull_dps"] > 0 else -1.0
        T = len(m["rendered_offset_m"])
        a = np.full((T, 4), np.nan)
        for t in range(T):
            ro, rh, co, ch = m["rendered_offset_m"][t], m["rendered_heading_deg"][t], c["real_offset_m"][t], c["real_heading_deg"][t]
            a[t, 0], a[t, 2] = sg * m["physics_offset_m"][t], sg * m["physics_heading_deg"][t]
            if None not in (ro, co):
                a[t, 1] = sg * (ro - co)
            if None not in (rh, ch):
                a[t, 3] = sg * (rh - ch)
        series[(m["policy"], int(anchor))][(m["segment"], m["start"], side)] = a
    anchors = sorted({a for _, a in series})
    pols = sorted({p for p, _ in series})
    T = min(v.shape[0] for d in series.values() for v in d.values())
    ts = list(range(1, T))
    print(f"anchors (frames after start, 5 Hz): {anchors}; rollout {T} frames ({T * DT:.1f} s)")
    for pol in pols:
        print("=" * 100)
        print(f"{pol}: pull-normalised means (+ = toward the pull)")
        print(f"{'anchor':>7} {'what':>22} " + " ".join(f"{t * DT:>6.1f}s" for t in ts))
        for an in anchors:
            arr = np.array([v[:T] for v in series[(pol, an)].values()])
            for j, what in [(0, "physics offset m"), (1, "rendered offset m"), (2, "physics heading deg"), (3, "rendered heading deg")]:
                print(f"{an:>7} {what:>22} " + " ".join(f"{np.nanmean(arr[:, t, j]):>+7.2f}" for t in ts))
        # paired far - near, cluster bootstrap over drives
        near = anchors[0]
        for far in anchors[1:]:
            keys = sorted(set(series[(pol, near)]) & set(series[(pol, far)]))
            segs = sorted({k[0] for k in keys})
            for j, what in [(1, "rendered offset"), (3, "rendered heading")]:
                d = {s: np.array([series[(pol, far)][k][:T, j] - series[(pol, near)][k][:T, j] for k in keys if k[0] == s]) for s in segs}
                est = np.nanmean(np.concatenate(list(d.values())), axis=0)
                bs = []
                for _ in range(2000):
                    pick = [segs[i] for i in rng.integers(len(segs), size=len(segs))]
                    bs.append(np.nanmean(np.concatenate([d[s] for s in pick]), axis=0))
                lo, hi = np.nanpercentile(bs, [2.5, 97.5], axis=0)
                cells = " ".join(f"{est[t]:>+6.2f}{'*' if lo[t] > 0 or hi[t] < 0 else ' '}" for t in ts)
                print(f"  {what:>16} anchor {far} - {near}: {cells}")
        print("  (* = 95% CI excludes 0; > 0 means the near anchor shows less drift)")
    if len(sys.argv) > 3:
        rows = [r for r in load(sys.argv[3]) if r.get("policy") == "teacher"]
        print("=" * 100)
        print("teacher final drift toward the pull by anchor (m): a near anchor that hides drift -> less correction")
        for an in anchors:
            fin = []
            for r in rows:
                if int(r["kick"].split("@")[1]) != an:
                    continue
                p, h = np.asarray(r["policy_path"], float), np.asarray(r["human_path"], float)
                fin.append((1 if r["pull_dps"] > 0 else -1) * (p[-1, 1] - h[-1, 1]))
            print(f"  anchor {an:>3}: {np.mean(fin):+.2f} m  (n={len(fin)})")


if __name__ == "__main__":
    main()
