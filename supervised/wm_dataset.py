"""World-model rollouts (python -m rl.collect) as training batches in the same format as Comma1MDataset.

A sample is the 9-step image history the student sees, ending on a GENERATED frame: step t holds the (previous,
current) 5 Hz frame pair, each upscaled to 512x256 and packed to YUV planes exactly as rl.actor.SupercomboActor does in
closed loop. Only the reconstruction target `imgs` is known for generated states; every other label comes from the
teacher run online on the same images (train.py with WM_DATA). Batches mix rollouts (one sample per rollout), and the
rollout list is re-read every epoch, so rollouts collected while training (DAgger iterations) join automatically.
"""

import glob
import json
import os
import random

import cv2
import numpy as np
from torch.utils.data import IterableDataset, get_worker_info

from .dataset import _pack_image

N_STEPS = 9  # image history steps per sample, as Comma1MDataset (history_idxs) gives
TEMPORAL_LEN = 33  # desire / traffic / action_t history length, as Comma1MDataset gives


def _pack(frame: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    narrow = cv2.resize(np.ascontiguousarray(frame[..., :3]), (512, 256), interpolation=cv2.INTER_LINEAR)
    wide = cv2.resize(np.ascontiguousarray(frame[..., 3:]), (512, 256), interpolation=cv2.INTER_LINEAR)
    return _pack_image(narrow), _pack_image(wide)


def load_rollout(path: str) -> tuple[np.ndarray, dict] | None:
    try:
        with np.load(path) as z:
            frames, meta = z["frames"], json.loads(bytes(z["meta"]).decode())
    except Exception:  # noqa: BLE001 - a file being written or a broken file is skipped, never fatal
        return None
    if meta.get("skipped") or len(frames) < meta["n_history"] + 1 or meta["n_history"] < N_STEPS:
        return None
    return frames, meta


def make_sample(frames: np.ndarray, meta: dict, end: int, action_t: np.ndarray):
    """Sample whose last image is frames[end] (end >= n_history, so it is generated)."""
    packed = [_pack(frames[i]) for i in range(end - N_STEPS, end + 1)]
    img = np.stack([np.concatenate((packed[t][0], packed[t + 1][0])) for t in range(N_STEPS)])
    big_img = np.stack([np.concatenate((packed[t][1], packed[t + 1][1])) for t in range(N_STEPS)])
    traffic = np.zeros((TEMPORAL_LEN, 2), np.float32)
    traffic[:, int(bool(meta["lht"]))] = 1
    inputs = {
        "img": img,
        "big_img": big_img,
        "desire_pulse": np.zeros((TEMPORAL_LEN, 8), np.float32),
        "traffic_convention": traffic,
        "action_t": np.broadcast_to(action_t, (TEMPORAL_LEN, 2)).astype(np.float32).copy(),
    }
    return inputs, {"imgs": np.ascontiguousarray(frames[end])}


class WorldModelDataset(IterableDataset):
    def __init__(self, root: str, batch_size: int = 8, refresh_every: int = 200):
        super().__init__()
        self.root, self.batch_size, self.refresh_every = root, batch_size, refresh_every

    def _paths(self) -> list[str]:
        return sorted(glob.glob(os.path.join(self.root, "**", "r*.npz"), recursive=True))

    def __iter__(self):
        worker = get_worker_info()
        seed = (worker.id if worker else 0) * 7919 + os.getpid()
        rng = random.Random(seed)
        np_rng = np.random.default_rng(seed)
        paths, n = self._paths(), 0
        cache: dict[str, tuple[np.ndarray, dict] | None] = {}
        while True:
            if n % self.refresh_every == 0:
                paths = self._paths()
                if len(cache) > 256:
                    cache.clear()
            n += 1
            if not paths:
                raise RuntimeError(f"no rollouts under {self.root}")
            batch = []
            for path in rng.choices(paths, k=self.batch_size * 2):
                if path not in cache:
                    cache[path] = load_rollout(path)
                if cache[path] is None:
                    continue
                frames, meta = cache[path]
                end = rng.randrange(meta["n_history"], len(frames))
                batch.append(make_sample(frames, meta, end, np_rng.uniform(0.0, 1.0, size=2)))
                if len(batch) == self.batch_size:
                    break
            if not batch:
                continue
            inputs = {k: np.stack([b[0][k] for b in batch]) for k in batch[0][0]}
            targets = {k: np.stack([b[1][k] for b in batch]) for k in batch[0][1]}
            yield inputs, targets
