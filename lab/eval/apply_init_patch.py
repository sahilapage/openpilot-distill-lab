"""Adds INIT_WEIGHTS to supervised/train.py: a NEW run (no resume file yet) starts from another run's weights."""
import sys
p = sys.argv[1]
s = open(p).read()
if "INIT_PATCH" in s:
    print("already patched"); sys.exit(0)
anchor = "    # LR_PATCH:"
assert s.count(anchor) == 1, "LR_PATCH anchor not found"
init = '''    # INIT_PATCH: start a NEW run (no resume file yet) from another run's weights, fresh optimizer and LR schedule
    elif os.environ.get("INIT_WEIGHTS"):
        raw_model.load_state_dict(torch.load(os.environ["INIT_WEIGHTS"], map_location=device))
        if is_main:
            print(f"initialised weights from {os.environ['INIT_WEIGHTS']}")

'''
s = s.replace(anchor, init + anchor)
compile(s, p, "exec")
open(p, "w").write(s)
print("patched")
