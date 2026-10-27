"""Load every staged segment once through the training loader (supervised.dataset._load_segment, NVDEC decode), so a
segment that would crash a DataLoader worker never enters data/. Prints "OK <seg> <device_type> <n_samples>" or
"FAIL <seg> <error>" per segment. Run from the repo root with PYTHONPATH=. (needs the GPU's video decoder).
  python check_segments.py STAGING_DIR [--procs 4]
"""
import argparse
import multiprocessing as mp
import os
import sys
from pathlib import Path


def check(seg_dir: str) -> str:
    import fsspec
    import numpy as np
    from safetensors.numpy import load

    from supervised.dataset import DatasetConfig, _load_segment

    seg = os.path.basename(seg_dir.rstrip("/"))
    try:
        fs, _ = fsspec.url_to_fs(seg_dir)
        device = int(load(open(f"{seg_dir}/frame_info.safetensors", "rb").read())["device_type"].item())
        np.random.seed(abs(hash(seg)) % 2**31)
        inputs, targets = _load_segment(fs, seg_dir, DatasetConfig(str(Path(seg_dir).parent.parent)), False, 0)
        n = len(inputs["img"])
        if n == 0:
            return f"FAIL {seg} no valid samples"
        bad = [k for k, v in {**inputs, **targets}.items() if v.dtype.kind == "f" and not np.isfinite(v).all()]
        return f"FAIL {seg} non-finite {bad}" if bad else f"OK {seg} {device} {n}"
    except Exception as e:  # noqa: BLE001 - report every failure, keep going
        return f"FAIL {seg} {type(e).__name__}: {str(e)[:200]}"


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("staging", type=Path)
    p.add_argument("--procs", type=int, default=4)
    args = p.parse_args()
    dirs = sorted(str(d) for d in args.staging.iterdir() if (d / "localizer.safetensors").exists())
    with mp.get_context("spawn").Pool(args.procs) as pool:
        for line in pool.imap_unordered(check, dirs):
            print(line, flush=True)
    sys.exit(0)


if __name__ == "__main__":
    main()
