"""Smoke test: every component the pipeline needs, with no training. Run from the repo root."""
import time
from pathlib import Path

import numpy as np

ROOT = Path.cwd()
SEG = "2333e255f1de1fe83ea5975b24a3cc67"
D = ROOT / "data" / SEG

import torch

print("torch", torch.__version__, "cuda", torch.version.cuda, "available", torch.cuda.is_available())
print("device", torch.cuda.get_device_name(0), torch.cuda.get_device_capability(0))
x = torch.randn(2048, 2048, device="cuda", dtype=torch.bfloat16)
print("bf16 matmul ok", float((x @ x).float().abs().mean()) > 0)

import onnxruntime as ort

print("onnxruntime", ort.__version__, ort.get_available_providers())

from safetensors.numpy import load_file

fi = load_file(D / "frame_info.safetensors")
lo = load_file(D / "localizer.safetensors")
print("frame_info keys:", sorted(fi)[:12], "... device_type", fi["device_type"])
print("localizer keys:", sorted(lo), {k: lo[k].shape for k in lo if k in ("states", "frame_states", "rpy")})

from helpers.video_helpers import decode_frames

t = time.time()
idx = fi["fcamera/index"]
frames = decode_frames(D / "fcamera.hevc", idx, {0, 20, 40}, gpu_id=0)
print("NVDEC decoded", {k: v.shape for k, v in frames.items()}, f"{time.time() - t:.1f}s")

from openpilot.selfdrive.modeld.constants import ModelConstants  # noqa: F401
from supervised.model_for_inference import ORTSupercomboForInference

t = time.time()
teacher = ORTSupercomboForInference.from_supercombo(
    str(ROOT / "models/big_driving_supercombo.onnx"),
    providers=[("CUDAExecutionProvider", {"device_id": 0}), "CPUExecutionProvider"],
)
print("teacher loaded", f"{time.time() - t:.1f}s", "providers", teacher.session.get_providers() if hasattr(teacher, "session") else "?")

from supervised.model import Supercombo

student = Supercombo().cuda()
print(f"student params: {sum(p.numel() for p in student.parameters()) / 1e6:.1f}M")
print("peak GPU mem MB:", torch.cuda.max_memory_allocated() // 2**20)
print("SMOKE_OK")
