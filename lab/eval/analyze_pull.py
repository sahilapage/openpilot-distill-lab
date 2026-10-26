"""Steering-pull test (eval_shove.py --mode pull), scored in the repo frame (run rescore.py first).

A constant steering-rate disturbance (pull_dps) is added to every policy's command for the whole rollout. Reported:
  drift(t)       mean signed lateral offset from the human path, + = toward the pull, per 5 Hz frame
  final drift    the same at the last frame; paired differences vs `straight` (never steers) and vs `student`,
                 cluster bootstrap over drives
  cut            1 - final drift / final drift of `straight` on the same disturbances
  counter-curv   mean curvature the policy adds against the pull vs `straight` (1/m, + = steering back)
  split          unseen drives (repo val split, never trained on) vs drives the supervised student trained on;
                 right vs left pull

  python -I analyze_pull.py results_pull_fixed.jsonl [val_segments.txt]
"""
import json
import sys
from collections import defaultdict

import numpy as np

DT = 0.2


def signed_drift(r):
    p = np.asarray(r["policy_path"], float)
    h = np.asarray(r["human_path"], float)
    n = min(len(p), len(h))
    sgn = 1.0 if r["pull_dps"] > 0 else -1.0  # + pull_dps = pulled right (repo frame, y right)
    return sgn * (p[:n, 1] - h[:n, 1]), sgn


def counter(r, ref, sgn):
    """Curvature the policy adds against the pull, relative to `straight` on the same disturbance (logged actions
    include the pull itself), 1/m, + = steering back."""
    a = np.asarray(r["actions"], float)[:, 0]
    b = np.asarray(ref["actions"], float)[:, 0]
    n = min(len(a), len(b))
    return -sgn * (a[:n] - b[:n])


def boot_diff(by_seg, a, b, key, rng, n=4000):
    segs = [s for s in by_seg if a in by_seg[s] and b in by_seg[s]]
    d = {s: np.mean([x - y for x, y in zip(by_seg[s][a][key], by_seg[s][b][key])]) for s in segs}
    w = {s: len(by_seg[s][a][key]) for s in segs}
    vals = np.array([d[s] for s in segs]); wts = np.array([w[s] for s in segs], float)
    est = np.sum(vals * wts) / wts.sum()
    bs = []
    for _ in range(n):
        i = rng.integers(len(segs), size=len(segs))
        bs.append(np.sum(vals[i] * wts[i]) / wts[i].sum())
    lo, hi = np.percentile(bs, [2.5, 97.5])
    return est, lo, hi


def table(rows, title, rng):
    print("=" * 100)
    print(f"{title}: {len({r['segment'] for r in rows})} drives, {len(rows)} rollouts")
    pols = sorted({r["policy"] for r in rows})
    # pair rollouts by (segment, start, kick) so every policy is compared on the same disturbance
    by_seg = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    keyed = defaultdict(dict)
    for r in rows:
        keyed[(r["segment"], r["start"], r["kick"])][r["policy"]] = r
    full = {k: v for k, v in keyed.items() if set(v) == set(pols)}
    T = min(len(signed_drift(r)[0]) for v in full.values() for r in v.values())
    curves = defaultdict(list)
    for (seg, _, _), v in sorted(full.items()):
        for pol, r in v.items():
            d, sgn = signed_drift(r)
            c = counter(r, v["straight"], sgn)
            ref_final = signed_drift(v["straight"])[0][T - 1]
            curves[pol].append(d[:T])
            by_seg[seg][pol]["ref"].append(ref_final)
            by_seg[seg][pol]["final"].append(d[T - 1])
            by_seg[seg][pol]["cs"].append(float(np.mean(c)))
            by_seg[seg][pol]["rec"].append(float(bool(r.get("recovered"))))
    print(f"paired disturbances: {len(full)}  (frames compared: {T}, {T * DT:.1f} s)\n")
    ts = [int(T * f) for f in (0.2, 0.4, 0.6, 0.8)] + [T - 1]
    print(f"{'policy':>10}  " + "  ".join(f"t={t * DT:>3.1f}s" for t in ts) + "   final [95% CI]        cut vs straight  counter-curv  recovered")
    for pol in pols:
        c = np.array(curves[pol])
        segs = list(by_seg)
        fin = np.array([np.mean(by_seg[s][pol]["final"]) for s in segs])
        wts = np.array([len(by_seg[s][pol]["final"]) for s in segs], float)
        bs = [np.sum(fin[i] * wts[i]) / wts[i].sum() for i in (rng.integers(len(segs), size=len(segs)) for _ in range(4000))]
        canc = 1 - np.sum([x for s in segs for x in by_seg[s][pol]["final"]]) / np.sum([x for s in segs for x in by_seg[s][pol]["ref"]])
        cs = np.mean([x for s in segs for x in by_seg[s][pol]["cs"]])
        rec = np.mean([x for s in segs for x in by_seg[s][pol]["rec"]])
        print(f"{pol:>10}  " + "  ".join(f"{c[:, t].mean():>+7.2f}" for t in ts)
              + f"   {c[:, -1].mean():+6.2f} [{np.percentile(bs, 2.5):+.2f},{np.percentile(bs, 97.5):+.2f}]   {100 * canc:>7.0f}%        {1e3 * cs:+.2f}e-3    {100 * rec:>4.0f}%")
    print("\npaired final-drift difference (m, negative = drifts less), cluster bootstrap over drives")
    for ref in ("straight", "student"):
        if ref not in pols:
            continue
        for pol in pols:
            if pol == ref:
                continue
            est, lo, hi = boot_diff(by_seg, pol, ref, "final", rng)
            flag = "  <-- CI excludes 0" if lo > 0 or hi < 0 else ""
            print(f"  {pol:>10} - {ref:<8} {est:+6.2f} [{lo:+.2f},{hi:+.2f}]{flag}")
    print()


def correction_vs_bias(rows) -> None:
    """Split each policy's extra curvature vs `straight` into correction (opposes the pull: (left-pull - right-pull)/2)
    and lean (same sign whichever way it is pulled: (right + left)/2). A lean helps one side and hurts the other, so a
    one-sided drift gain with a large lean is not recovery skill."""
    keyed = defaultdict(dict)
    for r in rows:
        keyed[(r["segment"], r["start"], r["pull_dps"] > 0)][r["policy"]] = r
    starts = sorted({(s, st) for s, st, _ in keyed if (s, st, True) in keyed and (s, st, False) in keyed})
    act = lambda r: np.asarray(r["actions"], float)[:, 0]
    print("=" * 100)
    print("extra curvature vs straight, 1e-3 1/m (+ = right): correction opposes the pull, lean is the same both ways")
    for pol in sorted({r["policy"] for r in rows}):
        c, b = [], []
        for s, st in starts:
            R, L = keyed[(s, st, True)], keyed[(s, st, False)]
            if pol not in R or pol not in L:
                continue
            dR = float(np.mean(act(R[pol]) - act(R["straight"])))
            dL = float(np.mean(act(L[pol]) - act(L["straight"])))
            c.append((dL - dR) / 2e-3); b.append((dR + dL) / 2e-3)
        print(f"  {pol:>10}  correction {np.mean(c):+.2f}   lean {np.mean(b):+.2f}")
    print()


def main() -> None:
    rows = [json.loads(line) for line in open(sys.argv[1]) if line.strip()]
    rows = [r for r in rows if r.get("mode") == "pull"]
    if any("ref_frame" not in r for r in rows):
        raise SystemExit("rows lack ref_frame: run rescore.py first")
    rng = np.random.default_rng(0)
    table(rows, "ALL", rng)
    if len(sys.argv) > 2:
        val = set(open(sys.argv[2]).read().split())
        table([r for r in rows if r["segment"] in val], "UNSEEN drives (repo val split)", rng)
        table([r for r in rows if r["segment"] not in val], "drives the supervised student TRAINED on", rng)
    for side in ("right", "left"):
        table([r for r in rows if r.get("side") == side], f"pull to the {side.upper()}", rng)
    correction_vs_bias(rows)


if __name__ == "__main__":
    main()
