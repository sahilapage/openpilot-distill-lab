"""Choose which rollouts to feature. Lists (segment, start) pairs that have saved frames for the given policies,
ranked by how much the 'after' policy improved on the 'before' one, plus the random set used for the wall.

  python pick_clips.py results.jsonl frames/ [--before student --after dagger200]
"""
from __future__ import annotations

import argparse
import random
from pathlib import Path

import render


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("results", type=Path)
    p.add_argument("frames", type=Path)
    p.add_argument("--before", default="student")
    p.add_argument("--after", default="dagger200")
    p.add_argument("--n", type=int, default=12)
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args()
    rows = render.load_results(a.results)
    keys = sorted({(s, st) for (s, st, pol) in rows if (a.frames / f"{s}_{st}_{a.before}.npz").exists() and (a.frames / f"{s}_{st}_{a.after}.npz").exists() and (s, st, a.before) in rows and (s, st, a.after) in rows})
    print(f"{len(keys)} clips with frames for both {a.before} and {a.after}")
    scored = sorted(((rows[(s, st, a.before)]["max_err"] - rows[(s, st, a.after)]["max_err"], s, st) for s, st in keys), reverse=True)
    print("\nlargest improvement (hero candidates):")
    for d, s, st in scored[:5]:
        print(f"  {s} {st}: {rows[(s, st, a.before)]['max_err']:.1f} m -> {rows[(s, st, a.after)]['max_err']:.1f} m  (-{d:.1f})")
    mid = scored[len(scored) // 2]
    print(f"\nmedian clip: {mid[1]} {mid[2]}  ({rows[(mid[1], mid[2], a.before)]['max_err']:.1f} -> {rows[(mid[1], mid[2], a.after)]['max_err']:.1f})")
    worse = [x for x in scored if x[0] < 0]
    print(f"clips where '{a.after}' did WORSE than '{a.before}': {len(worse)} of {len(scored)}")
    rnd = random.Random(a.seed).sample(keys, min(a.n, len(keys)))
    print("\nrandom set for the wall (seeded, not cherry-picked):")
    for s, st in rnd:
        print(f"  {s} {st}")


if __name__ == "__main__":
    main()
