"""Rank steering-pull drives for the 'nobody steering vs comma's model' montage: bright scene, never-steer car far off
the road, teacher much closer. Writes contact sheets: never-steer (top) and teacher (bottom) at 0 / 2.5 / 5 s.
  python pick_pairs.py FRAMES_DIR forgiveness_pull.jsonl OUT_DIR
"""
import json
import sys
from pathlib import Path

import cv2
import numpy as np

frames_dir, meas_p, out = Path(sys.argv[1]), sys.argv[2], Path(sys.argv[3])
meas = {(m["segment"], m["start"], m["policy"], m["kick"]): m for m in map(json.loads, open(meas_p))}
cands = []
for (seg, st, pol, kick), m in meas.items():
    if pol != "straight" or (seg, st, "teacher", kick) not in meas:
        continue
    a, b = frames_dir / f"{seg}_{st}_straight_{kick}.npz", frames_dir / f"{seg}_{st}_teacher_{kick}.npz"
    if not (a.exists() and b.exists()):
        continue
    s_off = abs(m["physics_offset_m"][-1])
    t_off = abs(meas[(seg, st, "teacher", kick)]["physics_offset_m"][-1])
    fa = np.load(a)["frames"][..., :3]
    bright = float(fa.mean())
    score = (s_off - 2 * t_off) * (0.4 + min(bright, 110) / 110)
    cands.append((score, seg, st, kick, s_off, t_off, bright, a, b))
cands.sort(reverse=True)
print(f"{'score':>6} seg      start kick nobody_off teacher_off bright")
for c in cands[:14]:
    print(f"{c[0]:6.1f} {c[1][:8]} {c[2]:5d} {c[3]:>4}  {c[4]:8.1f}  {c[5]:9.1f}  {c[6]:5.0f}")
tiles = []
for c in cands[:10]:
    fa, fb = np.load(c[7])["frames"][..., :3], np.load(c[8])["frames"][..., :3]
    pick = lambda f: np.hstack([cv2.resize(f[i], (320, 160)) for i in (0, len(f) // 2, len(f) - 1)])  # noqa: E731
    tile = np.vstack([pick(fa), pick(fb)])
    cv2.putText(tile, f"{c[1][:8]} {c[2]} {c[3]}  nobody {c[4]:.0f}m / comma {c[5]:.0f}m", (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
    tiles.append(np.vstack([tile, np.zeros((6, tile.shape[1], 3), np.uint8)]))
for k in range(0, len(tiles), 4):
    cv2.imwrite(str(out / f"pairs{k // 4}.jpg"), cv2.cvtColor(np.vstack(tiles[k:k + 4]), cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 85])
