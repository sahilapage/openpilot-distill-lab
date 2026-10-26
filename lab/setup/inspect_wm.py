"""CPU-only inspection of the world-model torchpackage: how is NVFP4 used, can it be dequantized?"""
import collections
import io
import re

import torch
from torch.package import PackageImporter

path = "models/model.fp8_nvfp4.torchpackage"
imp = PackageImporter(path)
meta = imp.load_pickle("meta", "meta.pkl")
print("meta keys:", list(meta))
print("model_io:", {k: (v if not isinstance(v, dict) else {kk: vv for kk, vv in list(v.items())[:8]}) for k, v in meta["model_io"].items()})
for k in meta:
    if k != "model_io":
        print(k, str(meta[k])[:300])

files = [f for f in imp.zip_reader.get_all_records() if f.endswith(".py")]
print("python sources in package:", len(files))
for f in files:
    print("  ", f)

for f in files:
    src = imp.zip_reader.get_record(f).decode()
    hits = [(n, l) for n, l in enumerate(src.splitlines(), 1) if re.search(r"nvfp4|NVFP4|float4|fp8|Float8|quantiz", l, re.I)]
    if hits:
        print(f"--- {f}: {len(hits)} quantization lines")
        for n, l in hits[:40]:
            print(f"   {n}: {l.strip()[:150]}")

state = torch.load(io.BytesIO(imp.load_binary("assets", "state_dict.pt")), map_location="cpu", weights_only=False)
kinds = collections.Counter()
examples = {}
for k, v in state.items():
    key = f"{type(v).__name__}/{getattr(v, 'dtype', '?')}"
    kinds[key] += 1
    examples.setdefault(key, k)
print("state_dict tensor kinds:")
for k, c in kinds.most_common():
    print(f"  {c:5d} {k}   e.g. {examples[k]}")
total = sum(v.numel() * v.element_size() for v in state.values() if hasattr(v, "numel") and hasattr(v, "element_size"))
print(f"approx stored bytes: {total / 1e9:.2f} GB")
