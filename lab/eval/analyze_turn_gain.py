"""Turn gain: how much of the human's turn does each policy reproduce in the world model, and does it end up inside
(overshoot) or outside (undershoot) the human's curve? Rollouts where the human turned >= MIN_DEG.
  turn gain   = policy heading change / human heading change   (0 = drove straight, 1 = matched the human)
  inside      = signed final lateral error toward the inside of the human's turn (m); >0 overshoot, <0 undershoot
Cluster bootstrap over drives.   python analyze_turn_gain.py results.jsonl [MIN_DEG]
"""
import json, sys
from collections import defaultdict
import numpy as np

def hchange(path):
    d = np.diff(np.asarray(path, float), axis=0); h = np.unwrap(np.arctan2(d[:, 1], d[:, 0])); return h[-1] - h[0]

rows = [json.loads(l) for l in open(sys.argv[1]) if l.strip()]
MIN = float(sys.argv[2]) if len(sys.argv) > 2 else 3.7
per = defaultdict(lambda: defaultdict(list))
for r in rows:
    hh = hchange(r["human_path"])
    if abs(np.rad2deg(hh)) < MIN: continue
    s = np.sign(hh)
    gain = hchange(r["policy_path"]) / hh
    inside = s * r["lat_final"]
    per[r["policy"]][r["segment"]].append((gain, inside, np.sign(hchange(r["policy_path"])) == s))
rng = np.random.default_rng(0)
print(f"rollouts with human turn >= {MIN} deg")
for p in sorted(per):
    segs = list(per[p]); allv = np.array([v for s in segs for v in per[p][s]], float)
    boots = []
    for _ in range(3000):
        pick = [per[p][segs[i]] for i in rng.integers(len(segs), size=len(segs))]
        b = np.array([v for g in pick for v in g], float); boots.append([np.median(b[:, 0]), b[:, 1].mean()])
    lo, hi = np.percentile(boots, [2.5, 97.5], axis=0)
    print(f"{p:>10}  n={len(allv):3d}  turn gain median {np.median(allv[:,0]):+5.2f} [{lo[0]:+5.2f},{hi[0]:+5.2f}]  "
          f"turned the right way {allv[:,2].mean():4.0%}  ended inside the curve by {allv[:,1].mean():+5.2f} m [{lo[1]:+5.2f},{hi[1]:+5.2f}]  "
          f"overshoot share {np.mean(allv[:,1] > 0):4.0%}")
