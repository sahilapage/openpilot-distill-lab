"""RETRACTED premise, kept for the record: the world model does NOT hide the drift. The generated frames show the car
leaving the road; the 'drawn offset' comes from the lane-line readout, which saturates near half a lane. Do not use.

Rank steering-pull clips of the never-steer car for the 'world model hides the drift' video: bright scene, valid lane
readout, big real drift, small drawn drift. Writes a contact sheet (first / middle / last frame) of the top clips.
  python pick_hidden.py FRAMES_DIR forgiveness_pull.jsonl calib_pull.jsonl results_pull_fixed.jsonl OUT_DIR
"""
import json
import sys
from pathlib import Path

import cv2
import numpy as np

frames_dir, meas_p, calib_p, res_p, out = map(Path, sys.argv[1:6])
calib = {(c["segment"], c["start"]): c for c in map(json.loads, open(calib_p))}
rows = {(r["segment"], r["start"], r["policy"], r["kick"]): r for r in map(json.loads, open(res_p))}
cands = []
for m in map(json.loads, open(meas_p)):
    if m["policy"] != "straight":
        continue
    npz = frames_dir / f"{m['segment']}_{m['start']}_straight_{m['kick']}.npz"
    if not npz.exists():
        continue
    fr = np.load(npz)["frames"][..., :3]
    c = calib[(m["segment"], m["start"])]
    rend = [abs(a - b) for a, b in zip(m["rendered_offset_m"], c["real_offset_m"]) if a is not None and b is not None]
    phys = abs(m["physics_offset_m"][-1])
    bright = float(fr.mean())
    sharp = float(np.mean([cv2.Laplacian(cv2.cvtColor(f, cv2.COLOR_RGB2GRAY), cv2.CV_32F).var() for f in fr[::5]]))
    score = (phys - 3 * max(rend)) * (0.5 + min(bright, 120) / 120)
    cands.append((score, m["segment"], m["start"], m["kick"], phys, max(rend), np.mean(rend), bright, sharp, npz))
cands.sort(reverse=True)
print(f"{'score':>6} seg      start kick  real_off  drawn_max drawn_mean bright sharp")
for c in cands[:16]:
    print(f"{c[0]:6.1f} {c[1][:8]} {c[2]:5d} {c[3]:>4}  {c[4]:7.1f}  {c[5]:8.2f} {c[6]:9.2f} {c[7]:6.0f} {c[8]:6.0f}")
tiles = []
for c in cands[:12]:
    fr = np.load(c[9])["frames"][..., :3]
    row = np.hstack([cv2.resize(fr[i], (384, 192)) for i in (0, len(fr) // 2, len(fr) - 1)])
    cv2.putText(row, f"{c[1][:8]} {c[2]} {c[3]} real {c[4]:.1f}m drawn {c[5]:.2f}m", (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
    tiles.append(row)
for k in range(0, len(tiles), 6):
    cv2.imwrite(str(out / f"sheet{k // 6}.jpg"), cv2.cvtColor(np.vstack(tiles[k:k + 6]), cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 85])
