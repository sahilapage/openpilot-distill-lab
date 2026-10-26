"""Idempotently patch rl/server.py so pre-Blackwell GPUs run the NVFP4 MLP weights as dequantized bf16."""
from pathlib import Path

p = Path("rl/server.py")
s = p.read_text()
if "ADA_NVFP4_DEQUANT" in s:
    print("already patched")
    raise SystemExit(0)

old = """        self.model.load_state_dict(state, strict=True, assign=True)
"""
new = """        # ADA_NVFP4_DEQUANT: NVFP4 matmul needs Blackwell (the package declares minimum_compute_capability).
        # On older GPUs run those MLP weights as plain bf16 instead (dequantized once at load).
        minimum_capability = tuple(importer.load_pickle("meta", "meta.pkl").get("minimum_compute_capability", (0, 0)))
        if tuple(torch.cuda.get_device_capability(self.device)) < minimum_capability:
            from torchao.prototype.mx_formats.nvfp4_tensor import NVFP4Tensor

            state = {
                name: value.dequantize(torch.bfloat16).contiguous() if isinstance(value, NVFP4Tensor) else value
                for name, value in state.items()
            }
        self.model.load_state_dict(state, strict=True, assign=True)
"""
assert s.count(old) == 1, "anchor not found exactly once"
p.write_text(s.replace(old, new))
print("patched")
