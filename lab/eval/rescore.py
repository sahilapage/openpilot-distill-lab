"""Rewrite results rows with every human-path metric in the repo frame (evalkit.rescore), then check the frame end to end:
each rollout starts on a REAL logged frame, so the teacher's first action must agree in sign with the human's curvature at
that moment. A mirrored reference shows up as a strongly negative correlation (it was -0.89 before the fix).

  python -I rescore.py results.jsonl results_rescored.jsonl
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from evalkit import rescore  # noqa: E402


def path_curvature(path: np.ndarray) -> np.ndarray:
    d = np.diff(path, axis=0)
    h = np.unwrap(np.arctan2(d[:, 1], d[:, 0]))
    return np.diff(h) / np.maximum(np.linalg.norm(d, axis=1)[1:], 0.1)


def check_policy_frame(rows: list[dict], policy: str = "teacher") -> float:
    a, b = [], []
    for r in rows:
        if r["policy"] != policy or (r.get("pull_dps") or r.get("kick_deg")):
            continue  # disturbed rows add an injected turn on top of the policy's action
        k = path_curvature(np.asarray(r["human_path"], float))[:3].mean()
        if abs(k) < 5e-4:
            continue
        a.append(np.asarray(r["actions"], float)[:2, 0].mean())
        b.append(k)
    if len(a) < 10:
        print(f"frame check: only {len(a)} undisturbed {policy} rows on bends, skipped")
        return float("nan")
    corr = float(np.corrcoef(a, b)[0, 1])
    print(f"frame check: {policy} first action vs human curvature on {len(a)} bends: corr {corr:+.2f}, "
          f"sign agreement {np.mean(np.sign(a) == np.sign(b)):.0%}")
    if corr < 0.3:
        raise SystemExit("frame check FAILED: policy actions and the human path disagree in sign")
    return corr


def main() -> None:
    src, dst = Path(sys.argv[1]), Path(sys.argv[2])
    rows = [rescore(json.loads(line)) for line in src.read_text().splitlines() if line.strip()]
    dst.write_text("".join(json.dumps(r) + "\n" for r in rows))
    print(f"rescored {len(rows)} rows -> {dst}")
    check_policy_frame(rows)


if __name__ == "__main__":
    main()
