"""Summarise eval_open_loop.py: how much of the human's steering each policy reproduces on the real drive.

Per policy and frame source (raw camera / VAE-reconstructed), over all drives, skipping the first WARM frames and frames
below MIN_SPEED, on LATERAL ACCELERATION v^2 * curvature (curvature itself is dominated by noise at low speed):
  lag     the shift (0..MAX_LAG 5 Hz steps) at which the policy best matches the human; policies plan ahead
  gain    slope through the origin of policy lateral accel on the human's (1 = steers as much as the human)
  corr    correlation of the two (pooled), and the median of per-drive correlations
  curve gain   the same slope on frames where the human's |lateral accel| > 0.5 m/s^2
Cluster bootstrap over drives for the gains.

  python -I analyze_open_loop.py open_loop.jsonl [val_segments.txt]
With a val list, every table is repeated for UNSEEN drives (repo val split) and for drives the supervised students
TRAINED on: a student that steers well only on the trained drives has memorised them.
"""
import json
import sys
from collections import defaultdict

import numpy as np

WARM, MAX_LAG, MIN_SPEED = 10, 8, 5.0


def pairs(row, lag):
    a = np.asarray(row["actions"], float)[:, 0]
    h = np.asarray(row["human_actions"], float)[:, 0]
    v = np.asarray(row["speeds"], float)
    n = min(len(a), len(h), len(v)) - lag
    idx = np.arange(WARM, n)
    idx = idx[v[idx] > MIN_SPEED]
    return v[idx] ** 2 * a[idx], v[idx + lag] ** 2 * h[idx + lag], v[idx + lag]


def slope(x, y):
    return float(np.sum(x * y) / np.sum(x * x)) if np.sum(x * x) > 0 else float("nan")


def main() -> None:
    rows = [json.loads(line) for line in open(sys.argv[1]) if line.strip()]
    report(rows, "ALL drives")
    if len(sys.argv) > 2:
        val = set(open(sys.argv[2]).read().split())
        report([r for r in rows if r["segment"] in val], "UNSEEN drives (repo val split)")
        report([r for r in rows if r["segment"] not in val], "drives the supervised students TRAINED on")


def report(rows, title) -> None:
    print("=" * 90 + f"\n{title}")
    groups = defaultdict(list)
    for r in rows:
        groups[(r["policy"], r["source"])].append(r)
    rng = np.random.default_rng(0)
    print(f"{len({r['segment'] for r in rows})} drives; gain = policy lateral accel / human lateral accel (1 = steers like the human)\n")
    print(f"{'policy':>12} {'source':>5}  lag   gain [95% CI]          corr  drive-med  curve gain [95% CI]")
    for (pol, src), rs in sorted(groups.items()):
        lag = max(range(MAX_LAG + 1), key=lambda L: np.corrcoef(*np.concatenate(
            [np.stack(pairs(r, L)[:2]) for r in rs], axis=1))[0, 1])
        per = [pairs(r, lag) for r in rs]

        def stats(items):
            a = np.concatenate([i[0] for i in items])
            h = np.concatenate([i[1] for i in items])
            v = np.concatenate([i[2] for i in items])
            curve = np.abs(h) > 0.5
            return slope(h, a), slope(h[curve], a[curve]), float(np.corrcoef(a, h)[0, 1])

        g, cg, c = stats(per)
        med = float(np.median([np.corrcoef(i[0], i[1])[0, 1] for i in per if len(i[0]) > 2 and i[0].std() > 0 and i[1].std() > 0] or [np.nan]))
        boots = np.array([stats([per[i] for i in rng.integers(len(per), size=len(per))])[:2] for _ in range(2000)])
        lo, hi = np.nanpercentile(boots, 2.5, axis=0), np.nanpercentile(boots, 97.5, axis=0)
        print(f"{pol:>12} {src:>5}  {lag * 0.2:.1f}s  {g:+.2f} [{lo[0]:+.2f},{hi[0]:+.2f}]   {c:+.2f}   {med:+.2f}     "
              f"{cg:+.2f} [{lo[1]:+.2f},{hi[1]:+.2f}]")


if __name__ == "__main__":
    main()
