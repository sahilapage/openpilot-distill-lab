"""Summarise eval_closed_loop results: per-policy means and paired differences with cluster-bootstrap CIs.

Usage: python3 -I analyze.py results.jsonl [--baseline student]
Paired = same (segment, start) across policies. The bootstrap resamples whole SEGMENTS (starts within a segment are
correlated), 10 000 times, so the intervals reflect segment-to-segment variation.
"""
import argparse
import json
from collections import defaultdict

import numpy as np

METRICS = ["ade", "fde", "max_err", "speed_rmse", "lat_acc_rms", "jerk_lat_rms", "jerk_long_rms", "success"]
LOWER_IS_BETTER = {m: m != "success" for m in METRICS}
COMFORT = {"lat_acc_rms", "jerk_lat_rms", "jerk_long_rms"}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("results")
    p.add_argument("--baseline", default="student")
    p.add_argument("--boots", type=int, default=10000)
    p.add_argument("--val", help="file listing the train/val-split held-out segments; adds a val-only table")
    args = p.parse_args()

    rows = [json.loads(line) for line in open(args.results)]
    by_policy = defaultdict(dict)  # policy -> (segment, start) -> row
    for r in rows:
        by_policy[r["policy"]][(r["segment"], r["start"])] = r
    names = sorted(by_policy)
    all_keys = set.intersection(*(set(v) for v in by_policy.values()))  # only starts every policy completed
    val_segs = set(open(args.val).read().split()) if args.val else set()
    groups = {"ALL held-out segments": all_keys}
    if val_segs:
        groups["supervised-val segments only (never seen by the student)"] = {k for k in all_keys if k[0] in val_segs}
        groups["DAgger-holdout segments only (seen by the student, not by DAgger)"] = {k for k in all_keys if k[0] not in val_segs}
    for title, keys in groups.items():
        if keys:
            print("=" * 100 + f"\n{title}")
            summarize(by_policy, names, keys, args)


def summarize(by_policy, names, keys, args) -> None:
    segs = sorted({k[0] for k in keys})
    print(f"policies={names}  paired starts={len(keys)}  segments={len(segs)}\n")

    def seg_means(policy: str, metric: str) -> np.ndarray:
        per_seg = defaultdict(list)
        for k in keys:
            per_seg[k[0]].append(float(by_policy[policy][k][metric]))
        return np.array([np.mean(per_seg[s]) for s in segs])

    rng = np.random.default_rng(0)
    idx = rng.integers(0, len(segs), size=(args.boots, len(segs)))

    print(f"{'metric':14s}" + "".join(f"{n:>22s}" for n in names))
    for m in METRICS:
        cells = []
        for n in names:
            v = seg_means(n, m)
            lo, hi = np.percentile(v[idx].mean(1), [2.5, 97.5])
            cells.append(f"{v.mean():8.3f} [{lo:6.3f},{hi:6.3f}]")
        print(f"{m:14s}" + "".join(f"{c:>22s}" for c in cells))

    if args.baseline in by_policy:
        print(f"\npaired difference vs {args.baseline} (negative = better for error metrics; success: positive = better)")
        for n in names:
            if n == args.baseline:
                continue
            print(f"  {n}:")
            for m in METRICS:
                d = seg_means(n, m) - seg_means(args.baseline, m)
                lo, hi = np.percentile(d[idx].mean(1), [2.5, 97.5])
                sig = "  <-- CI excludes 0" if lo > 0 or hi < 0 else ""
                if m in COMFORT:  # lower means smoother, not necessarily better driving
                    verdict = "(lower = smoother)"
                else:
                    verdict = "better" if (d.mean() < 0) == LOWER_IS_BETTER[m] else "worse "
                print(f"    {m:14s} {d.mean():+8.3f} [{lo:+7.3f},{hi:+7.3f}] {verdict}{sig}")


if __name__ == "__main__":
    main()
