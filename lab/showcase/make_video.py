"""Render a showcase video from eval output.

  python make_video.py hero --results results.jsonl --frames frames/ --segment S --start N \
      --before student --after dagger200 --out hero.mp4
"""
from __future__ import annotations

import argparse
from pathlib import Path

import render
import scenes


def main() -> None:
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)
    h = sub.add_parser("hero")
    h.add_argument("--results", type=Path, required=True)
    h.add_argument("--frames", type=Path, required=True)
    h.add_argument("--segment", required=True)
    h.add_argument("--start", type=int, required=True)
    h.add_argument("--before", default="student")
    h.add_argument("--after", default="dagger200")
    h.add_argument("--before-label", default="SUPERVISED STUDENT")
    h.add_argument("--after-label", default="AFTER DAgger")
    h.add_argument("--out", type=Path, required=True)
    k_ = sub.add_parser("shove", help="hero layout on shove-test output (results_shove.jsonl + its frames)")
    k_.add_argument("--results", type=Path, required=True)
    k_.add_argument("--frames", type=Path, required=True)
    k_.add_argument("--segment", required=True)
    k_.add_argument("--start", type=int, required=True)
    k_.add_argument("--kick", choices=("L", "R"), default="L")
    k_.add_argument("--before", default="student")
    k_.add_argument("--after", default="dagger1400")
    k_.add_argument("--before-label", default="SUPERVISED STUDENT")
    k_.add_argument("--after-label", default="AFTER DAgger")
    k_.add_argument("--bare", action="store_true", help="no title, verdict or outro cards: just the drive")
    k_.add_argument("--headline", help="override the top headline (two parts separated by |, or empty for none)")
    k_.add_argument("--out", type=Path, required=True)
    s_ = sub.add_parser("strip")
    s_.add_argument("--results", type=Path, required=True)
    s_.add_argument("--frames", type=Path, required=True)
    s_.add_argument("--segment", required=True)
    s_.add_argument("--start", type=int, required=True)
    s_.add_argument("--policies", default="student,dagger10,dagger50,dagger100,dagger200")
    s_.add_argument("--labels", default="SUPERVISED,DAgger 10,DAgger 50,DAgger 100,DAgger 200")
    s_.add_argument("--out", type=Path, required=True)
    w_ = sub.add_parser("wall")
    w_.add_argument("--results", type=Path, required=True)
    w_.add_argument("--frames", type=Path, required=True)
    w_.add_argument("--before", default="student")
    w_.add_argument("--after", default="dagger200")
    w_.add_argument("--n", type=int, default=12)
    w_.add_argument("--seed", type=int, default=0)
    w_.add_argument("--out", type=Path, required=True)
    t_ = sub.add_parser("stats")
    t_.add_argument("--results", type=Path, required=True)
    t_.add_argument("--policies", default="straight,student,dagger10,dagger50,dagger100,dagger200,teacher")
    t_.add_argument("--out", type=Path, required=True)
    args = p.parse_args()

    rows = render.load_results(args.results, kick=getattr(args, "kick", None))
    if args.cmd == "shove":
        a = render.load_rollout(rows, args.frames, args.segment, args.start, args.before, args.before_label)
        b = render.load_rollout(rows, args.frames, args.segment, args.start, args.after, args.after_label)
        if a.row.get("mode") == "pull":
            dps = abs(a.row["pull_dps"])
            texts = (("THE STEERING-PULL TEST", f"the steering pulls {dps:.0f}°/s to one side, the whole drive. Can the model fight it?"),
                     (["I sabotaged the steering", "of a self-driving model."], "can it feel it and fight back?"),
                     (["Same pull. Same drive.", "Same seed."], "student distilled from comma's model  ·  coached by comma's model inside its world model"))
        else:
            texts = (("THE SHOVE TEST", "knock the car 15° off line, then let the model take over"),
                     (["I shoved a self-driving", "model off the road."], "can it find its way back?"),
                     (["Same shove. Same drive.", "Same seed."], "student distilled from comma's model  ·  fine-tuned with DAgger"))
        head = texts[0] if args.headline is None else tuple((args.headline.split("|") + [""])[:2])
        bare = dict(passes=((1.0, "real time"), (1.0, "real time")), corner=None) if args.bare else {}  # plays twice at real speed, no corner text
        scenes.render_hero(a, b, args.out, (a.name, b.name), head, "",
                           None if args.bare else texts[1], None if args.bare else texts[2], **bare)
        print("wrote", args.out)
        return
    if args.cmd == "wall":
        import random
        keys = sorted({(s, st) for (s, st, pol) in rows if (s, st, args.before) in rows and (s, st, args.after) in rows
                       and (args.frames / f"{s}_{st}_{args.before}.npz").exists() and (args.frames / f"{s}_{st}_{args.after}.npz").exists()})
        pick = random.Random(args.seed).sample(keys, min(args.n, len(keys)))
        before = [render.load_rollout(rows, args.frames, s, st, args.before) for s, st in pick]
        after = [render.load_rollout(rows, args.frames, s, st, args.after) for s, st in pick]
        scenes.render_wall(
            before, after, args.out, ("BEFORE: supervised student", "AFTER: DAgger"),
            (["12 random drives.", "Same model, before and after."], "no cherry-picking"),
            (["Same drives. Same starts.", "Same seed."], "student distilled from comma's model  ·  fine-tuned with DAgger"),
        )
        print("wrote", args.out, "clips:", pick)
        return
    if args.cmd == "stats":
        pols = args.policies.split(",")
        names = {"straight": "drive straight", "student": "supervised student", "dagger10": "DAgger 10", "dagger50": "DAgger 50",
                 "dagger100": "DAgger 100", "dagger200": "DAgger 200", "teacher": "comma's teacher"}
        scenes.render_stats(rows, pols, names, args.out, (["Measured, not cherry-picked."], "held-out drives  ·  inside comma's world model  ·  dequantized to run on a 4090"))
        print("wrote", args.out)
        return
    if args.cmd == "strip":
        pols, labs = args.policies.split(","), args.labels.split(",")
        rolls = [render.load_rollout(rows, args.frames, args.segment, args.start, q, l) for q, l in zip(pols, labs)]
        scenes.render_strip(
            rolls, labs, args.out,
            ("PRACTICE MAKES PERFECT", "the same model after 0, 10, 50, 100 and 200 DAgger rollouts"),
            (["Watch it learn", "to stay on the road."], "five checkpoints, one drive"),
            (["Same drive. Same start.", "Same seed."], "student distilled from comma's model  ·  fine-tuned with DAgger"),
        )
        print("wrote", args.out)
        return
    a = render.load_rollout(rows, args.frames, args.segment, args.start, args.before, args.before_label)
    b = render.load_rollout(rows, args.frames, args.segment, args.start, args.after, args.after_label)
    scenes.render_hero(
        a, b, args.out, (a.name, b.name),
        ("BEFORE vs AFTER DAgger", "a tiny driving model, trained inside comma's world model"),
        "",
        (["A tiny self-driving model", "drove off the road."], "so I let it practice in an AI-generated world"),
        (["Same drive. Same start.", "Same seed."], "student distilled from comma's model  ·  fine-tuned with DAgger"),
    )
    print("wrote", args.out)


if __name__ == "__main__":
    main()
