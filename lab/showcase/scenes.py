"""Scenes for the showcase videos. Everything shown is computed from eval output; nothing is typed in by hand."""
from __future__ import annotations

import numpy as np

from render import (AMBER, BG, BLUE, GREEN, INK, MUTED, ORANGE, PANEL, RED, FPS_OUT, FPS_SRC, Canvas, Rollout, Writer,
                    crossfade, draw_bev, draw_overlays, ease, err_color, upscale)

W, H = 1920, 1080
MPH = 2.23694


def title_card(lines, sub=None, t=1.0, size=76, color=INK) -> np.ndarray:
    """Lines fade/slide in one after another; t in seconds."""
    c = Canvas(W, H)
    y0 = H // 2 - (len(lines) * (size + 18)) // 2 - (30 if sub else 0)
    for i, line in enumerate(lines):
        a = ease((t - 0.35 * i) / 0.55)
        if a <= 0:
            continue
        col = tuple(int(BG[j] + (color[j] - BG[j]) * a) for j in range(3))
        c.text((W // 2, y0 + i * (size + 18) + int((1 - a) * 22)), line, size=size, fill=col, anchor="ma")
    if sub:
        a = ease((t - 0.35 * len(lines) - 0.1) / 0.6)
        col = tuple(int(BG[j] + (MUTED[j] - BG[j]) * a) for j in range(3))
        c.text((W // 2, y0 + len(lines) * (size + 18) + 18), sub, size=34, fill=col, bold=False, anchor="ma")
    return c.array()


def panel(c: Canvas, r: Rollout, t: float, x: int, y: int, w: int, label: str, accent, bev_extra=()) -> None:
    """One driving panel: label bar, narrow cam with overlays, wide cam, live HUD, BEV inset."""
    img = r.image(t)
    narrow = draw_overlays(upscale(img[..., :3], w), r, t)
    wide = upscale(img[..., 3:], w)
    bar_h, ch = 54, w // 2
    err = r.err_at(t)
    col = err_color(err)
    c.rrect((x, y, x + w, y + bar_h + 2 * ch + 12), r=18, fill=PANEL)
    c.d.ellipse((x + 22, y + 17, x + 42, y + 37), fill=accent)
    c.text((x + 56, y + 8), label, size=30, fill=INK)
    c.paste(narrow, (x + 6, y + bar_h))
    c.paste(wide, (x + 6, y + bar_h + ch))
    # HUD on the narrow view
    _, _, spd = r.pose(t)
    c.text((x + 26, y + bar_h + 16), f"{spd * MPH:3.0f}", size=54, fill=INK, mono=True)
    c.text((x + 26 + 110, y + bar_h + 42), "mph", size=22, fill=MUTED, bold=False)
    bx = x + w - 6 - 250
    c.rrect((bx, y + bar_h + 14, bx + 232, y + bar_h + 84), r=12, fill=(8, 10, 13, 200))
    c.text((bx + 16, y + bar_h + 18), "DRIFT FROM HUMAN PATH", size=13, fill=MUTED, bold=False)
    c.text((bx + 16, y + bar_h + 38), f"{err:4.1f} m", size=36, fill=col, mono=True)
    kf = r.row.get("kick_frames")
    if r.row.get("mode") == "pull":  # steering-pull test: the disturbance is on for the whole drive, keep a tag up
        side = "RIGHT" if r.row["pull_dps"] > 0 else "LEFT"  # + curvature = right turn
        tag = f"STEERING PULLED {abs(r.row['pull_dps']):.0f}°/s {side}"
        c.text((x + w - 24, y + 14), tag, size=22, fill=ORANGE, anchor="ra")
    elif kf and t < kf + 1.2:  # shove test: flash a banner while the steering is being forced and just after
        side = "RIGHT" if r.row["kick_deg"] > 0 else "LEFT"
        c.rrect((x + w // 2 - 215, y + bar_h + 130, x + w // 2 + 215, y + bar_h + 188), r=12, fill=(150, 90, 10, 235), outline=ORANGE)
        c.text((x + w // 2, y + bar_h + 138), f"SHOVED {abs(r.row['kick_deg']):.0f}° {side}", size=32, fill=INK, anchor="ma")
    if err > 2.0 and not (kf and t < kf + 1.2):
        c.rrect((x + w // 2 - 175, y + bar_h + ch - 70, x + w // 2 + 175, y + bar_h + ch - 22), r=12, fill=(120, 20, 24, 235), outline=RED)
        c.text((x + w // 2, y + bar_h + ch - 60), "OFF THE HUMAN PATH", size=24, fill=INK, anchor="ma")
    # BEV on the wide view's lower-right (dashboard area)
    bev = draw_bev(r, t, 236, bev_extra)
    c.paste(bev, (x + w - 6 - 244, y + bar_h + 2 * ch - 236 - 4))
    c.text((x + w - 6 - 238, y + bar_h + 2 * ch - 236 - 2), "TOP VIEW", size=12, fill=MUTED, bold=False)


def hero_frame(a: Rollout, b: Rollout, t: float, clock: float, labels, headline, footer,
               corner: str | None = "same drive  ·  same start  ·  same seed  ·  held-out") -> np.ndarray:
    c = Canvas(W, H)
    c.text((60, 34), headline[0], size=40, fill=INK)
    c.text((60, 82), headline[1], size=22, fill=MUTED, bold=False)
    if corner:
        c.text((W - 60, 40), corner, size=22, fill=MUTED, bold=False, anchor="ra")
    pw, py = 840, 124
    panel(c, a, t, 60, py, pw, labels[0], RED)
    panel(c, b, t, 1020, py, pw, labels[1], GREEN)
    # legend + progress
    ly = H - 36
    c.d.rectangle((60, ly - 8, 84, ly + 8), fill=(*GREEN, 255))
    c.text((94, ly - 14), "where the car is actually going", size=18, fill=MUTED, bold=False)
    c.d.rectangle((388, ly - 3, 412, ly + 3), fill=(*ORANGE, 255))
    c.text((422, ly - 14), "world-model expert plan", size=18, fill=MUTED, bold=False)
    for k in range(4):
        c.d.ellipse((676 + k * 10, ly - 4, 684 + k * 10, ly + 4), fill=INK)
    c.text((730, ly - 14), "what the human actually drove", size=18, fill=MUTED, bold=False)
    c.text((W - 60, ly - 14), footer, size=18, fill=MUTED, bold=False, anchor="ra")
    prog = np.clip(clock, 0, 1)
    c.d.rectangle((0, H - 4, int(W * prog), H), fill=(*BLUE, 255))
    return c.array()


def verdict_card(a: Rollout, b: Rollout, labels, t=1.0) -> np.ndarray:
    c = Canvas(W, H)
    shove = "tail_err" in a.row
    c.text((W // 2, 150), "distance from the road at the end of the drive" if shove else "max drift from the human path",
           size=40, fill=MUTED, bold=False, anchor="ma")
    for i, (r, lab, col) in enumerate(((a, labels[0], RED), (b, labels[1], GREEN))):
        cx = W // 2 + (-1 if i == 0 else 1) * 430
        v = float(r.row["tail_err"] if shove else r.err.max()) * ease(t / 0.9)
        ok = r.row["recovered"] if shove else r.success
        words = ("got back on the road", "never came back") if shove else ("stayed on the road", "left the road")
        c.text((cx, 330), f"{v:.1f} m", size=190, fill=col, mono=True, anchor="ma")
        c.text((cx, 560), lab, size=36, fill=INK, anchor="ma")
        c.text((cx, 612), words[0] if ok else words[1], size=28, fill=col, bold=False, anchor="ma")
    return c.array()


def render_hero(a: Rollout, b: Rollout, out, labels, headline, footer, intro, outro,
                passes=((1.0, "real time"), (0.33, "slow motion")), corner="same drive  ·  same start  ·  same seed  ·  held-out") -> None:
    w = Writer(out, W, H)
    n = min(a.n, b.n)
    tmax = n - 3  # keep the executed ribbon at least 0.6 s long
    # intro (intro=None: open straight on the drive)
    first = hero_frame(a, b, 0.0, 0.0, labels, headline, footer, corner)
    if intro is None:
        w.hold(first, 0.6)
    else:
        last = None
        for i in range(int(2.6 * FPS_OUT)):
            last = title_card(intro[0], intro[1], i / FPS_OUT)
            w.write(last)
        crossfade(w, last, first, 0.5)
    # real time, then slow motion
    total = 2.0 * tmax / FPS_SRC + 1.0
    done = 0.0
    span = sum(1 / sp for sp, _ in passes)
    for speed, label in passes:
        frames = int(tmax / FPS_SRC / speed * FPS_OUT)
        for i in range(frames):
            t = i / frames * tmax
            done += 1
            fr = hero_frame(a, b, t, done / (FPS_OUT * (tmax / FPS_SRC) * span), labels, headline,
                            f"{label}  ·  5 Hz generated frames, interpolated to 30 fps", corner)
            w.write(fr)
        if len(passes) > 1:
            w.hold(fr, 0.4)
    if outro is None:  # end on the drive itself: no verdict or outro cards
        w.hold(fr, 1.4)
        w.close()
        return
    # verdict
    prev = fr
    vc = verdict_card(a, b, labels, 0.0)
    crossfade(w, prev, vc, 0.4)
    for i in range(int(2.4 * FPS_OUT)):
        w.write(verdict_card(a, b, labels, i / FPS_OUT))
    vlast = verdict_card(a, b, labels, 2.0)
    ol = title_card(outro[0], outro[1], 0.0)
    crossfade(w, vlast, ol, 0.4)
    for i in range(int(2.8 * FPS_OUT)):
        w.write(title_card(outro[0], outro[1], i / FPS_OUT))
    w.close()


# ----------------------------------------------------------------------------------------------- progress strip
SCALE_COLORS = [(255, 84, 84), (255, 132, 64), (255, 190, 64), (176, 220, 80), (72, 224, 128), (96, 170, 255)]


def _bev_all(rolls: list, t: float, w: int, h: int) -> np.ndarray:
    """Big top-down view: road band around the human path, every policy's trail so far."""
    import cv2

    img = np.full((h, w, 3), 14, np.uint8)
    human = rolls[0].human
    allp = np.vstack([human] + [r.pts for r in rolls])
    x0, x1 = allp[:, 0].min() - 4, allp[:, 0].max() + 4
    y0, y1 = allp[:, 1].min() - 4, allp[:, 1].max() + 4
    span = max((y1 - y0) / w, (x1 - x0) / h)  # metres per pixel (fit both)
    cy, cx_ = 0.5 * (y0 + y1), x0

    def px(p):
        p = np.atleast_2d(p)
        return np.stack([w / 2 - (p[:, 1] - cy) / span, h - 24 - (p[:, 0] - cx_) / span], 1).astype(np.int32)

    cv2.polylines(img, [px(human)], False, (52, 55, 62), int(3.6 / span), cv2.LINE_AA)  # road band, ~3.6 m wide
    cv2.polylines(img, [px(human)], False, (190, 194, 204), 2, cv2.LINE_AA)
    for r, col in zip(rolls, SCALE_COLORS):
        i = int(np.clip(np.floor(t + 1), 0, len(r.pts) - 1))
        pos, _, _ = r.pose(t)
        trail = np.vstack([r.pts[: i + 1], pos[None]])
        cv2.polylines(img, [px(trail)], False, col, 4, cv2.LINE_AA)
        cv2.circle(img, tuple(px(pos)[0]), 8, col, -1, cv2.LINE_AA)
    return img


def _drift_chart(c: Canvas, rolls: list, t: float, x: int, y: int, w: int, h: int) -> None:
    ymax = max(2.6, max(float(r.err.max()) for r in rolls) * 1.08)
    tmax = (rolls[0].n) / FPS_SRC
    c.rrect((x, y, x + w, y + h), r=16, fill=PANEL)
    c.text((x + 24, y + 14), "DRIFT FROM THE HUMAN PATH", size=18, fill=MUTED, bold=False)
    gx0, gy0, gx1, gy1 = x + 70, y + 60, x + w - 30, y + h - 54

    def to(tt, e):
        return gx0 + (gx1 - gx0) * tt / tmax, gy1 - (gy1 - gy0) * e / ymax

    ty = to(0, 2.0)[1]
    for k in range(int(gx0), int(gx1), 16):
        c.d.line((k, ty, k + 8, ty), fill=(*RED, 150), width=2)
    c.text((gx1, ty - 24), "2 m  ·  off the road", size=16, fill=RED, bold=False, anchor="ra")
    c.d.line((gx0, gy1, gx1, gy1), fill=(70, 76, 88), width=2)
    c.d.line((gx0, gy0, gx0, gy1), fill=(70, 76, 88), width=2)
    for e in range(0, int(ymax) + 1):
        c.text((gx0 - 12, to(0, e)[1] - 10), f"{e}", size=16, fill=MUTED, bold=False, anchor="ra")
    c.text((gx0 + (gx1 - gx0) // 2, gy1 + 16), "seconds into the rollout", size=16, fill=MUTED, bold=False, anchor="ma")
    for r, col in zip(rolls, SCALE_COLORS):
        ts = np.arange(0, min(t + 1, len(r.err) - 1) + 1e-6, 0.25)
        pts = [to((q) / FPS_SRC, r.err_at(q - 1.0)) for q in ts]
        if len(pts) > 1:
            c.d.line(pts, fill=(*col, 255), width=4, joint="curve")
        if pts:
            c.d.ellipse((pts[-1][0] - 6, pts[-1][1] - 6, pts[-1][0] + 6, pts[-1][1] + 6), fill=col)


def strip_frame(rolls: list, names: list, t: float, clock: float, headline) -> np.ndarray:
    c = Canvas(W, H)
    c.text((60, 34), headline[0], size=40, fill=INK)
    c.text((60, 82), headline[1], size=22, fill=MUTED, bold=False)
    c.text((W - 60, 40), "same drive  ·  same start  ·  same seed  ·  held-out", size=22, fill=MUTED, bold=False, anchor="ra")
    n = len(rolls)
    cw, gap = 340, 30
    x0 = (W - (n * cw + (n - 1) * gap)) // 2
    for i, (r, nm, col) in enumerate(zip(rolls, names, SCALE_COLORS)):
        x = x0 + i * (cw + gap)
        img = r.image(t)
        cam = draw_overlays(upscale(img[..., :3], cw - 8), r, t, show_human=False)
        c.rrect((x, 124, x + cw, 124 + 40 + cam.shape[0] + 70), r=16, fill=PANEL)
        c.d.ellipse((x + 16, 138, x + 32, 154), fill=col)
        c.text((x + 42, 130), nm, size=24, fill=INK)
        c.paste(cam, (x + 4, 168))
        e = r.err_at(t)
        c.text((x + 18, 168 + cam.shape[0] + 8), f"{e:4.1f} m", size=34, fill=err_color(e), mono=True)
        c.text((x + cw - 16, 168 + cam.shape[0] + 22), "off the road" if e > 2.0 else "on the road", size=18,
               fill=RED if e > 2.0 else GREEN, bold=False, anchor="ra")
    top = 124 + 40 + (cw - 8) // 2 + 70 + 24
    bev = _bev_all(rolls, t, 900, H - top - 70)
    c.rrect((60, top, 60 + 920, top + bev.shape[0] + 20), r=16, fill=PANEL)
    c.paste(bev, (70, top + 10))
    c.text((84, top + bev.shape[0] - 16), "TOP VIEW  ·  grey road = what the human drove", size=16, fill=MUTED, bold=False)
    _drift_chart(c, rolls, t, 1000, top, W - 60 - 1000, bev.shape[0] + 20)
    c.d.rectangle((0, H - 4, int(W * np.clip(clock, 0, 1)), H), fill=(*BLUE, 255))
    return c.array()


def render_strip(rolls: list, names: list, out, headline, intro, outro, seconds_per_clip: float = 9.0) -> None:
    w = Writer(out, W, H)
    n = min(r.n for r in rolls)
    tmax = n - 3
    last = None
    for i in range(int(2.6 * FPS_OUT)):
        last = title_card(intro[0], intro[1], i / FPS_OUT)
        w.write(last)
    first = strip_frame(rolls, names, 0.0, 0.0, headline)
    crossfade(w, last, first, 0.5)
    frames = int(seconds_per_clip * FPS_OUT)
    for i in range(frames):
        t = ease(i / frames) * tmax * 0.15 + (i / frames) * tmax * 0.85  # gentle ease-in, ends at tmax
        fr = strip_frame(rolls, names, t, i / frames, headline)
        w.write(fr)
    w.hold(fr, 1.6)
    ol = title_card(outro[0], outro[1], 0.0)
    crossfade(w, fr, ol, 0.4)
    for i in range(int(2.8 * FPS_OUT)):
        w.write(title_card(outro[0], outro[1], i / FPS_OUT))
    w.close()


# ----------------------------------------------------------------------------------------------- wall of drives
def _wall(rolls: list, t: float, title: str, accent, cols=4, rows_=3) -> np.ndarray:
    import cv2

    c = Canvas(W, H)
    cw, ch, gap = 450, 225, 20
    x0, y0 = (W - (cols * cw + (cols - 1) * gap)) // 2, 150
    failed = 0
    for i, r in enumerate(rolls[: cols * rows_]):
        x, y = x0 + (i % cols) * (cw + gap), y0 + (i // cols) * (ch + gap)
        tt = min(t, r.n - 3)
        cam = draw_overlays(upscale(r.image(tt)[..., :3], cw), r, tt, show_human=False)
        run_max = float(max(r.err_at(q) for q in np.arange(0, tt + 1e-6, 0.5)))
        col = err_color(r.err_at(tt)) if run_max <= 2.0 else RED
        failed += run_max > 2.0
        c.paste(cam, (x, y))
        c.d.rectangle((x - 3, y - 3, x + cw + 2, y + ch + 2), outline=(*col, 255), width=4)
        c.d.rounded_rectangle((x + 8, y + ch - 40, x + 118, y + ch - 8), 8, fill=(8, 10, 13, 215))
        c.text((x + 16, y + ch - 38), f"{r.err_at(tt):4.1f} m", size=22, fill=col, mono=True)
    c.text((60, 36), title, size=44, fill=INK)
    c.d.ellipse((60, 100, 80, 120), fill=accent)
    c.text((92, 94), "12 random held-out drives  ·  same starts  ·  same seed", size=24, fill=MUTED, bold=False)
    by = y0 + rows_ * (ch + gap) + 4
    n = len(rolls[: cols * rows_])
    c.text((60, by + 6), f"{failed}", size=96, fill=RED if failed > n // 3 else AMBER if failed else GREEN, mono=True)
    c.text((60 + 150 if failed >= 10 else 60 + 100, by + 40), f"of {n} left the road", size=38, fill=INK)
    bw = 760
    c.d.rounded_rectangle((W - 60 - bw, by + 40, W - 60, by + 74), 17, fill=PANEL)
    if failed:
        c.d.rounded_rectangle((W - 60 - bw, by + 40, W - 60 - bw + max(34, int(bw * failed / n)), by + 74), 17, fill=(*RED, 255))
    c.text((W - 60, by + 84), "off the human path by more than 2 m", size=18, fill=MUTED, bold=False, anchor="ra")
    return c.array()


def render_wall(before: list, after: list, out, titles, intro, outro) -> None:
    import cv2

    w = Writer(out, W, H)
    tmax = min(r.n for r in before + after) - 3
    last = None
    for i in range(int(2.4 * FPS_OUT)):
        last = title_card(intro[0], intro[1], i / FPS_OUT)
        w.write(last)
    crossfade(w, last, _wall(before, 0.0, titles[0], RED), 0.5)
    frames = int(tmax / FPS_SRC / 0.8 * FPS_OUT)
    for i in range(frames):
        fr = _wall(before, i / frames * tmax, titles[0], RED)
        w.write(fr)
    w.hold(fr, 1.1)
    a_last = fr
    wipe_frames = int(tmax / FPS_SRC / 0.8 * FPS_OUT)
    for i in range(wipe_frames):
        b_fr = _wall(after, i / wipe_frames * tmax, titles[1], GREEN)
        f = ease(min(1.0, i / (0.55 * wipe_frames)))
        x_edge = int(W * f)
        mix = b_fr.copy()
        mix[:, x_edge:] = a_last[:, x_edge:]
        if 0 < x_edge < W:
            mix[:, max(0, x_edge - 3):x_edge + 3] = (255, 255, 255)
        w.write(mix)
        fr = b_fr
    w.hold(fr, 2.2)
    ol = title_card(outro[0], outro[1], 0.0)
    crossfade(w, fr, ol, 0.4)
    for i in range(int(2.8 * FPS_OUT)):
        w.write(title_card(outro[0], outro[1], i / FPS_OUT))
    w.close()


# ----------------------------------------------------------------------------------------------- aggregate stats
def aggregate(rows_by_key: dict, policies: list, metric: str, boots: int = 4000) -> dict:
    """Mean and 95% cluster-bootstrap CI over segments (starts within a segment are correlated)."""
    segs = sorted({k[0] for k in rows_by_key})
    out = {}
    rng = np.random.default_rng(0)
    idx = rng.integers(0, len(segs), size=(boots, len(segs)))
    for p in policies:
        per = []
        for s in segs:
            v = [float(r[metric]) for k, r in rows_by_key.items() if k[0] == s and k[2] == p]
            per.append(np.mean(v) if v else np.nan)
        per = np.array(per)
        ok = ~np.isnan(per)
        per, idx_ = per[ok], rng.integers(0, ok.sum(), size=(boots, ok.sum()))
        lo, hi = np.percentile(per[idx_].mean(1), [2.5, 97.5])
        out[p] = (float(per.mean()), float(lo), float(hi))
    return out


def stats_frame(stats: dict, labels: dict, colors: dict, title: str, t: float, unit: str, fmt: str, x_max: float, sub: str) -> np.ndarray:
    c = Canvas(W, H)
    c.text((60, 40), title, size=46, fill=INK)
    c.text((60, 100), sub, size=22, fill=MUTED, bold=False)
    names = list(stats)
    row_h = 110
    y0 = 200
    gx0, gx1 = 440, W - 140
    for i, nm in enumerate(names):
        y = y0 + i * row_h
        a = ease((t - 0.18 * i) / 0.9)
        m, lo, hi = stats[nm]
        c.text((gx0 - 28, y + 22), labels[nm], size=34, fill=INK, anchor="ra")
        c.d.rounded_rectangle((gx0, y + 12, gx1, y + 76), 12, fill=PANEL)
        bw = (gx1 - gx0) * min(m, x_max) / x_max * a
        if bw > 6:
            c.d.rounded_rectangle((gx0, y + 12, gx0 + bw, y + 76), 12, fill=colors[nm])
        if a > 0.95:
            for xx in (lo, hi):
                px_ = gx0 + (gx1 - gx0) * min(xx, x_max) / x_max
                c.d.line((px_, y + 22, px_, y + 66), fill=INK, width=3)
            c.d.line((gx0 + (gx1 - gx0) * min(lo, x_max) / x_max, y + 44, gx0 + (gx1 - gx0) * min(hi, x_max) / x_max, y + 44), fill=INK, width=3)
        c.text((gx1 + 16, y + 20), (fmt % (m * a)) + unit, size=34, fill=colors[nm], mono=True)
    c.text((60, H - 56), "bars = mean over held-out drives  ·  whiskers = 95% confidence interval (bootstrap over drives)", size=20, fill=MUTED, bold=False)
    return c.array()


def render_stats(rows_by_key: dict, policies: list, labels: dict, out, outro) -> None:
    colors = {"straight": MUTED, "student": SCALE_COLORS[0], "dagger10": SCALE_COLORS[1], "dagger50": SCALE_COLORS[2],
              "dagger100": SCALE_COLORS[3], "dagger200": SCALE_COLORS[4], "teacher": SCALE_COLORS[5]}
    colors = {p: colors.get(p, MUTED) for p in policies}
    fail = {p: (1 - m, 1 - hi, 1 - lo) for p, (m, lo, hi) in aggregate(rows_by_key, policies, "success").items()}
    fail = {p: tuple(100 * v for v in vals) for p, vals in fail.items()}
    drift = aggregate(rows_by_key, policies, "max_err")
    w = Writer(out, W, H)
    for title, st, unit, fmt, xmax, sub in (
        ("How often it left the road", fail, "%", "%.0f", 100.0, "a rollout 'leaves the road' if it ever strays more than 2 m from the human path"),
        ("How far it drifted", drift, " m", "%.1f", max(v[0] for v in drift.values()) * 1.15 + 1e-6, "the worst drift from the human path within each rollout"),
    ):
        for i in range(int(3.0 * FPS_OUT)):
            w.write(stats_frame(st, labels, colors, title, i / FPS_OUT, unit, fmt, xmax, sub))
        last = stats_frame(st, labels, colors, title, 3.0, unit, fmt, xmax, sub)
        w.hold(last, 1.6)
        if title.startswith("How often"):
            nxt = stats_frame(drift, labels, colors, "How far it drifted", 0.0, " m", "%.1f", max(v[0] for v in drift.values()) * 1.15, "")
            crossfade(w, last, nxt, 0.4)
    ol = title_card(outro[0], outro[1], 0.0)
    crossfade(w, last, ol, 0.4)
    for i in range(int(2.8 * FPS_OUT)):
        w.write(title_card(outro[0], outro[1], i / FPS_OUT))
    w.close()
