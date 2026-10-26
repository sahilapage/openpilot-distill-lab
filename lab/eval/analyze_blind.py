"""Can the world-model eval tell driving from not driving?

`straight` (zero steering, constant speed, never looks at the camera) is the blind baseline. Split the paired rollouts by how
much the HUMAN turned over the rollout (|heading change| of the logged path) and compare every policy to `straight`.
A useful closed-loop eval must punish `straight` on curves. Also splits error into lateral and longitudinal parts and
compares each policy's steering activity with the human's on the same windows.

Cluster bootstrap over segments (rollouts from the same drive are not independent).
  python analyze_blind.py results.jsonl
"""
import json
import sys
from collections import defaultdict

import numpy as np

FPS = 5.0


def heading_change_deg(path: np.ndarray) -> float:
    d = np.diff(path, axis=0)
    h = np.unwrap(np.arctan2(d[:, 1], d[:, 0]))
    return float(np.rad2deg(h[-1] - h[0]))


def human_lat_acc_rms(path: np.ndarray) -> float:
    d = np.diff(path, axis=0)
    h = np.unwrap(np.arctan2(d[:, 1], d[:, 0]))
    v = np.linalg.norm(d, axis=1) * FPS
    yaw_rate = np.diff(h) * FPS
    return float(np.sqrt(np.mean((v[1:] * yaw_rate) ** 2)))


def lat_long_err(row) -> tuple[float, float]:
    """Mean |lateral| and |longitudinal| error, measured in the human path's own frame at each time."""
    p, h = np.asarray(row["policy_path"], float), np.asarray(row["human_path"], float)
    d = np.diff(h, axis=0)
    hd = np.arctan2(d[:, 1], d[:, 0])
    e = (p - h)[1:]
    lat = -np.sin(hd) * e[:, 0] + np.cos(hd) * e[:, 1]
    lon = np.cos(hd) * e[:, 0] + np.sin(hd) * e[:, 1]
    return float(np.abs(lat).mean()), float(np.abs(lon).mean())


def cluster_boot(diffs_by_seg: dict, n=4000, seed=0):
    segs = list(diffs_by_seg)
    rng = np.random.default_rng(seed)
    est = float(np.mean(np.concatenate([diffs_by_seg[s] for s in segs])))
    boots = []
    for _ in range(n):
        pick = rng.integers(len(segs), size=len(segs))
        boots.append(np.mean(np.concatenate([diffs_by_seg[segs[i]] for i in pick])))
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return est, float(lo), float(hi)


def main() -> None:
    rows = [json.loads(line) for line in open(sys.argv[1]) if line.strip()]
    by_key = defaultdict(dict)
    for r in rows:
        by_key[(r["segment"], r["start"])][r["policy"]] = r
    keys = [k for k, v in by_key.items() if "straight" in v]
    policies = sorted({p for v in by_key.values() for p in v} - {"straight"})
    turn = {k: abs(heading_change_deg(np.asarray(by_key[k]["straight"]["human_path"], float))) for k in keys}
    hum_acc = {k: human_lat_acc_rms(np.asarray(by_key[k]["straight"]["human_path"], float)) for k in keys}
    edges = np.percentile(list(turn.values()), [33.3, 66.7])
    bins = {"straight road": lambda t: t < edges[0], "gentle bend": lambda t: edges[0] <= t < edges[1],
            "curve": lambda t: t >= edges[1]}
    print(f"{len(keys)} paired rollouts over {len({k[0] for k in keys})} drives; "
          f"human heading change tertiles at {edges[0]:.1f} and {edges[1]:.1f} deg\n")

    for name, inbin in [("all", lambda t: True)] + list(bins.items()):
        ks = [k for k in keys if inbin(turn[k])]
        tr = [turn[k] for k in ks]
        print(f"== {name}: {len(ks)} rollouts, human turned {np.median(tr):.1f} deg (median), "
              f"human lat-acc rms {np.mean([hum_acc[k] for k in ks]):.3f} m/s^2")
        s_ade = np.mean([by_key[k]["straight"]["ade"] for k in ks])
        s_succ = np.mean([by_key[k]["straight"]["success"] for k in ks])
        print(f"   {'straight':>10}  ade {s_ade:5.2f}   success {s_succ:4.2f}")
        for p in policies:
            kp = [k for k in ks if p in by_key[k]]
            diff = defaultdict(list)
            for k in kp:
                diff[k[0]].append(by_key[k][p]["ade"] - by_key[k]["straight"]["ade"])
            est, lo, hi = cluster_boot({s: np.asarray(v) for s, v in diff.items()})
            ade = np.mean([by_key[k][p]["ade"] for k in kp])
            succ = np.mean([by_key[k][p]["success"] for k in kp])
            lat = np.mean([lat_long_err(by_key[k][p])[0] for k in kp])
            lon = np.mean([lat_long_err(by_key[k][p])[1] for k in kp])
            acc = np.mean([by_key[k][p]["lat_acc_rms"] for k in kp])
            flag = "  <-- worse than not driving" if lo > 0 else ("  <-- beats straight" if hi < 0 else "")
            print(f"   {p:>10}  ade {ade:5.2f}  vs straight {est:+5.2f} [{lo:+5.2f},{hi:+5.2f}]  success {succ:4.2f}  "
                  f"lat {lat:4.2f} lon {lon:4.2f} m  lat-acc rms {acc:.3f}{flag}")
        s_lat = np.mean([lat_long_err(by_key[k]['straight'])[0] for k in ks])
        s_lon = np.mean([lat_long_err(by_key[k]['straight'])[1] for k in ks])
        print(f"   {'straight':>10}  lat {s_lat:4.2f} lon {s_lon:4.2f} m\n")


if __name__ == "__main__":
    main()
