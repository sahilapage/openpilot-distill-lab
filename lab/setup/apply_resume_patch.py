"""Make supervised/train.py resumable: save model+optimizer+step every SAVE_EVERY steps, resume if the file exists.
Reason: the shared 4090 was rebooted by someone else at step 9724/10000 and the final-only save lost 2.4 h of training."""
import sys
from pathlib import Path

p = Path(sys.argv[1])
s = p.read_text()
assert "RESUME_PATCH" not in s

a = "    for step in range(1, args.steps + 1):\n        step_start = time.perf_counter()\n"
b = '''    # RESUME_PATCH: periodic resumable checkpoint (model + optimizer + step), written atomically
    resume_path = args.output.with_suffix(".resume.pt")
    save_every = int(os.environ.get("SAVE_EVERY", 500))
    start_step = 0
    if resume_path.exists():
        state = torch.load(resume_path, map_location=device)
        raw_model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        start_step = state["step"]
        if is_main:
            print(f"resumed from {resume_path} at step {start_step}")

    for step in range(start_step + 1, args.steps + 1):
        step_start = time.perf_counter()
'''
assert s.count(a) == 1
s = s.replace(a, b)

c = "            run.log(metrics, step=step)\n"
d = '''            run.log(metrics, step=step)
            if step % save_every == 0:
                torch.save({"model": raw_model.state_dict(), "optimizer": optimizer.state_dict(), "step": step},
                           str(resume_path) + ".tmp")
                os.replace(str(resume_path) + ".tmp", resume_path)
'''
assert s.count(c) == 1
s = s.replace(c, d)
p.write_text(s)
print("patched", p)
