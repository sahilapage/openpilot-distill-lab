"""Does the world model render a policy's drift as the car leaving the road, or as the road bending with the car?

Everything is in the repo frame: x forward, y RIGHT, positive curvature = right turn.
Per generated frame, fit the rendered ego-lane centre (teacher lane-line head, device frame y right, x in [X0, X1] m) with
y = a + b x + c x^2:
  heading of the car relative to the rendered road  = -atan(b)   (car turned right -> road ahead goes left, b < 0)
  curvature of the rendered road                    = 2c         (road curving right -> y grows with x^2)
The same fit on the REAL drive (same VAE, same window) is the baseline. Physics (recomputed here from the result rows, so
old measurement files with the mirrored human path are fine) gives the car's heading relative to the human path and its
executed curvature; the human path gives the real road's curvature.

Two gains, through the origin over all frames, bootstrap CI over rollouts:
  heading gain = d(rendered heading - real) / d(physics heading rel. human)        1 = deviation shown as it is, 0 = hidden
  bend gain    = d(rendered curvature - real) / d(executed curvature - human)       +1 = the road bends WITH the car
Frame check first: on REAL frames the rendered road curvature must agree in sign with the human's own curvature.

  python -I analyze_bend.py forgiveness.jsonl calib.jsonl results.jsonl
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from evalkit import human_path  # noqa: E402

X_IDXS = 192.0 * (np.linspace(0.0, 1.0, 33) ** 2)  # ModelConstants.X_IDXS
X0, X1 = 4.0, 30.0
FIT = (X_IDXS >= X0) & (X_IDXS <= X1)


def fit(centre) -> tuple[float, float]:
    c2, b, _ = np.polyfit(X_IDXS[FIT], np.asarray(centre, float)[FIT], 2)
    return float(np.rad2deg(-np.arctan(b))), float(2 * c2)  # heading (deg) and road curvature (1/m), right-positive


def path_curvature(path: np.ndarray) -> np.ndarray:
    """Curvature at each pose index 1..n-1 (len n-1), right-positive for a repo-frame path."""
    d = np.diff(path, axis=0)
    hd = np.unwrap(np.arctan2(d[:, 1], d[:, 0]))
    s = np.linalg.norm(d, axis=1)
    k = np.zeros(len(path) - 1)
    k[1:] = np.diff(hd) / np.maximum(s[1:], 0.1)
    return k


def physics_relative(row: dict) -> np.ndarray:
    """Heading (deg) of the car relative to the human path per pose index, right-positive."""
    pts, hum = np.asarray(row["policy_path"], float), human_path(row)
    acts = np.asarray(row["actions"], float)
    chord = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    heading = np.concatenate([[0.0], np.cumsum(chord * acts[: len(chord), 0])])
    hd = np.diff(hum, axis=0)
    return np.rad2deg(heading - np.concatenate([[0.0], np.arctan2(hd[:, 1], hd[:, 0])]))


def load(path):
    return [json.loads(line) for line in open(path) if line.strip()]


def through_origin(x, y):
    return float(np.sum(x * y) / np.sum(x * x))


def main() -> None:
    meas, calib, results = load(sys.argv[1]), load(sys.argv[2]), load(sys.argv[3])
    rows = {(r["segment"], r["start"], r["policy"], r.get("kick")): r for r in results}
    base = {(c["segment"], c["start"]): c for c in calib}
    assert abs(X_IDXS[-1] - 192.0) < 1e-9

    # frame check on real frames: rendered road curvature vs the human's curvature over the same window
    any_row = {(r["segment"], r["start"]): r for r in results}
    rk, hk = [], []
    for key, c in base.items():
        if key not in any_row:
            continue
        h = path_curvature(human_path(any_row[key]))
        for k, cy in enumerate(c["real_centre_y"]):
            if cy is not None and k + 1 < len(h):
                rk.append(fit(cy)[1])
                hk.append(h[k + 1])
    rk, hk = np.asarray(rk), np.asarray(hk)
    bend = np.abs(hk) > 5e-4
    if bend.sum() >= 10:
        corr = float(np.corrcoef(rk[bend], hk[bend])[0, 1])
        print(f"frame check on REAL frames: rendered road curvature vs human curvature on {int(bend.sum())} bend frames: "
              f"corr {corr:+.2f}")
        if corr < 0:
            raise SystemExit("frame check FAILED: lane-line frame and human path disagree in sign")
    else:
        print(f"frame check on REAL frames: only {int(bend.sum())} bend frames, skipped")

    per = []
    for m in meas:
        c = base[(m["segment"], m["start"])]
        row = rows[(m["segment"], m["start"], m["policy"], m.get("kick"))]
        n = len(m["rendered_centre_y"])
        hd_r, cv_r, hd_b, cv_b = (np.full(n, np.nan) for _ in range(4))
        for k in range(n):
            if m["rendered_centre_y"][k] is None or c["real_centre_y"][k] is None:
                continue
            hd_r[k], cv_r[k] = fit(m["rendered_centre_y"][k])
            hd_b[k], cv_b[k] = fit(c["real_centre_y"][k])
        phys_h = physics_relative(row)[1 : n + 1]  # frame k shows the world after action k, i.e. pose index k+1
        exec_k = np.asarray(row["actions"], float)[:n, 0]
        hum_k = path_curvature(human_path(row))[:n]
        ok = ~np.isnan(hd_r)
        side = row.get("side") or ("right" if (row.get("pull_dps") or row.get("kick_deg") or 0) > 0 else "left")
        per.append(dict(name=f"{m['policy']} {side}", ph=phys_h[ok], rh=(hd_r - hd_b)[ok],
                        ek=(exec_k - hum_k)[ok], rk=(cv_r - cv_b)[ok]))
        p = per[-1]
        print(f"{p['name']:>15}  heading rel. road: physics {p['ph'][-1]:+6.1f}  rendered {p['rh'][-1]:+5.1f} deg   "
              f"curvature x1000: executed-human {1000 * np.mean(p['ek'][-8:]):+6.2f}  rendered-real {1000 * np.mean(p['rk'][-8:]):+6.2f}")

    def gains(items):
        return (through_origin(np.concatenate([i["ph"] for i in items]), np.concatenate([i["rh"] for i in items])),
                through_origin(np.concatenate([i["ek"] for i in items]), np.concatenate([i["rk"] for i in items])))

    hg, bg = gains(per)
    rng = np.random.default_rng(0)
    boots = np.array([gains([per[i] for i in rng.integers(len(per), size=len(per))]) for _ in range(2000)])
    (hl, bl), (hh, bh) = np.percentile(boots, 2.5, axis=0), np.percentile(boots, 97.5, axis=0)
    print(f"\nheading gain {hg:+.2f} [{hl:+.2f}, {hh:+.2f}]   (1 = the car is shown turned off the road as it is, 0 = hidden)")
    print(f"bend gain    {bg:+.2f} [{bl:+.2f}, {bh:+.2f}]   (+1 = the car's extra steering is shown as the road bending with it)")
    print(f"{len(per)} rollouts, fit window {X0:.0f}-{X1:.0f} m")


if __name__ == "__main__":
    main()
