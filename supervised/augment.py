"""Photometric augmentation for the real-drive loader (AUGMENT=1).

Distillation is function matching: the teacher labels the SAME augmented frames online, so the labels stay
consistent with what the student sees (Beyer et al. 2022, "A good teacher is patient and consistent").
Geometry is untouched, so the human-path targets stay valid too.

One set of parameters per sample and camera, shared by every frame of that sample (a clip keeps one look);
sensor noise is drawn per frame. 20% of samples stay clean.
"""

from __future__ import annotations

import os

import cv2
import numpy as np

CLEAN_PROB = 0.2


def enabled() -> bool:
    return os.environ.get("AUGMENT", "0") == "1"


class Photometric:
    def __init__(self, num_samples: int, num_cameras: int = 2, rng: np.random.Generator | None = None):
        self.rng = rng or np.random.default_rng()
        self.params = [[self._draw() for _ in range(num_cameras)] for _ in range(num_samples)]

    def _draw(self):
        r = self.rng
        if r.uniform() < CLEAN_PROB:
            return None
        x = np.arange(256, dtype=np.float32) / 255.0
        gamma = np.exp(r.uniform(-0.3, 0.3))
        contrast = r.uniform(0.7, 1.3)
        gain = np.exp(r.uniform(-0.35, 0.35))
        bias = r.uniform(-0.06, 0.06)
        white = r.uniform(0.9, 1.1, size=3)
        y = (x ** gamma - 0.5) * contrast + 0.5
        lut = np.stack([np.clip((y * gain + bias) * w, 0, 1) for w in white], -1)
        return {
            "lut": np.round(lut * 255).astype(np.uint8).reshape(256, 1, 3),
            "saturation": r.uniform(0.6, 1.4),
            "blur": r.uniform(0.4, 1.1) if r.uniform() < 0.25 else 0.0,
            "noise": r.uniform(2.0, 7.0) if r.uniform() < 0.3 else 0.0,
        }

    def __call__(self, rgb: np.ndarray, sample: int, camera: int) -> np.ndarray:
        p = self.params[sample][camera]
        if p is None:
            return rgb
        out = cv2.LUT(rgb, p["lut"])
        if abs(p["saturation"] - 1) > 1e-3:
            gray = cv2.cvtColor(cv2.cvtColor(out, cv2.COLOR_RGB2GRAY), cv2.COLOR_GRAY2RGB)
            out = cv2.addWeighted(out, p["saturation"], gray, 1 - p["saturation"], 0)
        if p["blur"]:
            out = cv2.GaussianBlur(out, (0, 0), p["blur"])
        if p["noise"]:
            noise = np.empty(out.shape, np.int16)
            cv2.randn(noise, 0, p["noise"])
            out = cv2.add(out, noise, dtype=cv2.CV_8U)
        return out
