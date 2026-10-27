"""Post chart: how hard each model turns the wheel on roads the student never trained on, vs the human (= 100%).

Steering gain = slope through the origin of the model's lateral acceleration on the human's, on the real frames of
the never-seen drives (repo val split), at each policy's best lag, as analyze_open_loop.py computes it; whiskers are
the 95% cluster bootstrap over drives. Every number on the image is computed here from the eval output.

  python make_scaling_chart.py results_open_loop_more.jsonl unseen_all.txt OUT.png
"""
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "eval"))
from analyze_open_loop import pairs, slope  # noqa: E402
from render import font  # noqa: E402

# dark chart surface and ink (dataviz reference palette, dark mode); one hue for our students, neutral for the reference
SURFACE = (26, 26, 25)
INK = (240, 239, 236)
INK2 = (195, 194, 183)
MUTED = (140, 139, 132)
GRID = (58, 58, 55)
OURS = (57, 135, 229)  # series-1 blue, dark step #3987e5
REF = (120, 119, 112)  # neutral: comma's model is the reference, not a competitor series

BARS = [("student40k", "my student, 128 drives", OURS), ("morefinal", "my student, 360 drives", OURS),
        ("teacher", "comma's model", REF)]


def gains(rows, segs, pol, rng, n=4000):
    best = max(range(9), key=lambda lag: np.corrcoef(*map(np.concatenate, zip(*[pairs(rows[(pol, s)], lag)[:2] for s in segs])))[0, 1])
    per = {s: pairs(rows[(pol, s)], best)[:2] for s in segs}
    g = lambda smp: slope(np.concatenate([per[s][1] for s in smp]), np.concatenate([per[s][0] for s in smp]))
    boot = [g(list(rng.choice(segs, len(segs)))) for _ in range(n)]
    return g(segs), np.percentile(boot, 2.5), np.percentile(boot, 97.5)


def main() -> None:
    res, val_p, out = sys.argv[1], sys.argv[2], sys.argv[3]
    val = set(open(val_p).read().split())
    rows = {(r["policy"], r["segment"]): r for r in map(json.loads, open(res)) if r["source"] == "raw"}
    segs = sorted(s for (p, s) in rows if p == "teacher" and s in val)
    rng = np.random.default_rng(0)
    stats = {pol: gains(rows, segs, pol, rng) for pol, _, _ in BARS}
    for pol, (g, lo, hi) in stats.items():
        print(f"{pol:11s} gain {g:+.3f} [{lo:+.3f}, {hi:+.3f}]")

    W, H, S = 1600, 900, 2  # drawn at 2x, downsampled for clean edges
    img = Image.new("RGB", (W * S, H * S), SURFACE)
    d = ImageDraw.Draw(img)
    f = lambda size, bold=True: font(size * S, bold)
    L, R = 560, 1440  # plot x range (0% .. 100%)
    X = lambda v: (L + (R - L) * v) * S
    d.text((80 * S, 70 * S), "how hard it steers on roads it has never seen", font=f(54), fill=INK)
    d.text((80 * S, 142 * S), f"vs the human driver = 100%  ·  open loop, real camera frames, {len(segs)} never-seen drives",
           font=f(28, False), fill=INK2)
    for v in (0, 0.25, 0.5, 0.75, 1.0):
        d.line([(X(v), 230 * S), (X(v), 760 * S)], fill=GRID, width=2 * S if v in (0, 1.0) else S)
        d.text((X(v), 776 * S), f"{int(v * 100)}%", font=f(26, False), fill=MUTED, anchor="mt")
    d.text((X(1.0) + 12 * S, 236 * S), "human", font=f(26, False), fill=MUTED, anchor="la")
    bar_h, gap = 92, 70
    for i, (pol, label, color) in enumerate(BARS):
        g, lo, hi = stats[pol]
        y0 = (262 + i * (bar_h + gap)) * S
        y1 = y0 + bar_h * S
        d.text((L * S - 28 * S, (y0 + y1) / 2), label, font=f(34, pol == "morefinal"), fill=INK if pol != "student40k" else INK2, anchor="rm")
        x1 = X(max(g, 0.0))
        d.rounded_rectangle([X(0), y0, x1, y1], radius=4 * S, fill=color)
        d.rectangle([X(0), y0, X(0) + 6 * S, y1], fill=color)  # square at the baseline, rounded at the data end
        yc = (y0 + y1) / 2
        d.line([(X(lo), yc), (X(hi), yc)], fill=INK, width=3 * S)
        for xv in (lo, hi):
            d.line([(X(xv), yc - 14 * S), (X(xv), yc + 14 * S)], fill=INK, width=3 * S)
        d.text((X(hi) + 22 * S, yc), f"{100 * g:.0f}%", font=f(46), fill=INK, anchor="lm")
    d.text((80 * S, 838 * S), "steering gain = slope of the model's lateral acceleration on the human's  ·  whiskers: 95% bootstrap over drives",
           font=f(22, False), fill=MUTED)
    img.resize((W, H), Image.LANCZOS).save(out)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
