"""
Sheet 1 — plant layout of the planar XY-θz stage. Standalone.

    python plant_sheet.py [prefix]        (needs numpy + matplotlib)
    writes <prefix>a_top_view.png (top view) and <prefix>b_section.png (section A-A)

The drawing is generated from the parameter table and the coil/encoder list below; the same points and directions
are what the plant model is built from (write-up section 6: lines of action -> input matrix B0). Only assumed design values are drawn; everything derived is worked out in the write-up (stage_control_post.md).
Design: 150 x 150 x 25 mm plate, m = 1 kg, four voice coils on lines y = ±a, x = ±a (a = 55 mm), four 2-D
grid-encoder heads at (±a, ±a) in the plate underside over grid plates in the base, four air bearings.
"""
import sys
import os
from pathlib import Path
from dataclasses import dataclass
from types import SimpleNamespace as NS
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyArrowPatch, Circle, Polygon, Arc



PREFIX = sys.argv[1] if len(sys.argv) > 1 else "sheet1"

OUT_DIR = Path("pictures")
OUT_DIR.mkdir(parents=True, exist_ok=True)

OUT_A = OUT_DIR / (PREFIX + "a_top_view_png")
OUT_B = OUT_DIR / (PREFIX + "b_section.png")



# ── parameters: name -> (value [SI], provenance tag) ──────────────────────────────────
PARAMS = {
    "L_side":  (0.150, "[A]"),    # square platform side, sized to the actuators
    "a":       (0.055, "[A]"),    # lever arm: distance O -> each line of action
    "h_plate": (0.025, "[A]"),    # platform thickness
    "z_cog":   (0.0125, "[A]"),   # CoG (= force plane) height above the bearing surface
    "reach":   (0.030, "[A]"),    # coil centre beyond the platform edge
    "head":    (0.055, "[A]"),    # 2-D encoder heads at (+/-head, +/-head): where an X line meets a Y line
    "travel":  (2.5e-3, "[A]"),   # x, y travel of a short-stroke fine stage (a few mm)
    "rot":     (1.5e-3, "[A]"),   # theta_z range: rotation range lithography grid encoders support (~1.5 mrad)
}
P = lambda k: PARAMS[k][0]
T = lambda k: PARAMS[k][1]

# ── actuators and encoders: (name, point in body frame [m], unit direction) ───────────
_a, _h = P("a"), P("L_side") / 2
_r, _e = _h + P("reach"), P("head")
g = NS(
    actuators=[NS(name=n, point=np.array(p), direction=np.array(d, float)) for n, p, d in [
        ("X1", (-_r, +_a), (1, 0)), ("X2", (+_r, -_a), (1, 0)),
        ("Y1", (+_a, +_r), (0, 1)), ("Y2", (-_a, -_r), (0, 1))]],
    # 2-D grid-encoder heads in the platform underside: each reads (x_i, y_i) of its own point against a grid
    # plate set flush into the base below it
    sensors=[NS(name=n, point=np.array(p)) for n, p in [
        ("E1", (+_e, +_e)), ("E2", (-_e, +_e)), ("E3", (-_e, -_e)), ("E4", (+_e, -_e))]],
)

mm = 1e3
L, a = P("L_side") * mm, P("a") * mm
h = L / 2
hp = P("h_plate") * mm
INK, DIM, HID, CL, RED, BLUE, NC, SC, FE, COIL = "#1b1b1b", "#444", "#555", "#8e44ad", "#c0392b", "#1f6fb5", "#e74c3c", "#3498db", "#5d6d7e", "#e59866"
w_m, w_c, gapNS, L_m, l_c = 18.0, 10.0, 4.0, 40.0, 30.0     # actuator footprint [A], see sheet 2
pitch = w_m + gapNS


def arrow(ax, p, q, c=RED, lw=2.2, ms=14, z=8, st="-|>"):
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle=st, mutation_scale=ms, color=c, lw=lw, zorder=z))


def dim(ax, p, q, txt, rot=0, fs=8, ext=None):
    arrow(ax, p, q, c=DIM, lw=0.7, ms=7, z=9, st="<|-|>")
    m = (np.array(p) + np.array(q)) / 2
    ax.text(*m, txt, ha="center", va="center", fontsize=fs, color=DIM, rotation=rot,
            bbox=dict(fc="white", ec="none", pad=0.4), zorder=10)
    for e in (ext or []):
        ax.plot(*zip(*e), color=DIM, lw=0.45, zorder=9)


def rot(pts, ang, c):
    R = np.array([[np.cos(ang), -np.sin(ang)], [np.sin(ang), np.cos(ang)]])
    return np.asarray(pts) @ R.T + c


fig = plt.figure(figsize=(11, 11.5))
ax = fig.add_axes([0.01, 0.01, 0.98, 0.98]); ax.set_aspect("equal"); ax.axis("off")
ax.set_xlim(-170, 170); ax.set_ylim(-165, 182)

# base and centre lines
ax.add_patch(Rectangle((-158, -158), 316, 316, fc="#f4f4f4", ec="#999", lw=1.0, zorder=0))
ax.text(-155, -155, "base (fixed)", fontsize=8, color="#666")
for (xs, ys) in [([-146, 146], [0, 0]), ([0, 0], [-146, 146])]:
    ax.plot(xs, ys, color=CL, lw=0.7, ls=(0, (12, 3, 2, 3)), zorder=1)
for yy in (a, -a):
    ax.plot([-146, 146], [yy, yy], color=RED, lw=0.6, ls=(0, (2, 3)), zorder=1)
for xx in (a, -a):
    ax.plot([xx, xx], [-146, 146], color=RED, lw=0.6, ls=(0, (2, 3)), zorder=1)

# platform
ax.add_patch(Rectangle((-h, -h), L, L, fc="#dfe7ef", ec=INK, lw=2.0, zorder=3))
ax.text(8, -h + 4, "platform", fontsize=8.5, ha="left", va="bottom", zorder=12)
# air bearing pads (hidden, under the platform)
for sx in (-1, 1):
    for sy in (-1, 1):
        ax.add_patch(Circle((sx * 25, sy * 25), 15, fc="none", ec=HID, lw=0.9, ls="--", zorder=4))
ax.text(25, 25, "air\nbearing", fontsize=7, ha="center", va="center", color=HID, zorder=4)

# actuators: yoke beside the platform, coil paddle protruding from the platform side
for act in g.actuators:
    p = act.point * mm
    d = act.direction
    ang = np.arctan2(d[1], d[0])
    out = np.sign(p) * (np.abs(p) > h)          # outward normal of the platform side it sticks out of
    # magnets (N|S split along the force) and yoke outline
    for x, col in zip((-pitch / 2, pitch / 2), (NC, SC)):
        ax.add_patch(Polygon(rot([[x - w_m / 2, -L_m / 2], [x + w_m / 2, -L_m / 2], [x + w_m / 2, L_m / 2], [x - w_m / 2, L_m / 2]], ang, p),
                             fc=col, ec="#333", lw=0.6, alpha=0.9, zorder=2))
    yk = rot([[-pitch / 2 - w_m / 2 - 4, -L_m / 2 - 4], [pitch / 2 + w_m / 2 + 4, -L_m / 2 - 4],
              [pitch / 2 + w_m / 2 + 4, L_m / 2 + 4], [-pitch / 2 - w_m / 2 - 4, L_m / 2 + 4]], ang, p)
    ax.add_patch(Polygon(yk, fc="none", ec=FE, lw=1.6, zorder=2))
    # coil paddle (top view) + tongue connecting it to the platform side
    co = rot([[-pitch / 2 - w_c / 2, -l_c / 2 - 3], [pitch / 2 + w_c / 2, -l_c / 2 - 3],
              [pitch / 2 + w_c / 2, l_c / 2 + 3], [-pitch / 2 - w_c / 2, l_c / 2 + 3]], ang, p)
    ci = rot([[-pitch / 2 + w_c / 2, -l_c / 2 + 3], [pitch / 2 - w_c / 2, -l_c / 2 + 3],
              [pitch / 2 - w_c / 2, l_c / 2 - 3], [-pitch / 2 + w_c / 2, l_c / 2 - 3]], ang, p)
    ax.add_patch(Polygon(co, fc=COIL, ec="#7e5109", lw=1.1, alpha=0.75, zorder=5))
    ax.add_patch(Polygon(ci, fc="#f4f4f4", ec="#7e5109", lw=0.8, zorder=5))
    edge = p - out * (abs(p @ out) - h)          # point on the platform edge
    tongue_w = 14
    perp = np.array([-out[1], out[0]])
    tq = [edge + perp * tongue_w / 2, edge - perp * tongue_w / 2, p - perp * tongue_w / 2 - out * 8, p + perp * tongue_w / 2 - out * 8]
    ax.add_patch(Polygon(tq, fc="#aab7b8", ec="#566573", lw=0.8, zorder=4))
    arrow(ax, p, p + d * 28, lw=2.8, ms=17, z=9)
    po = np.array([0.0, np.sign(p[1])]) if abs(d[0]) > 0.5 else np.array([np.sign(p[0]), 0.0])   # away from the axis
    ax.text(*(p + out * 37), f"$f_{{{act.name}}}$", color=RED, fontsize=13, ha="center", va="center", zorder=10)
    ax.text(*(p + po * (L_m / 2 + 14)), f"VC {act.name}", fontsize=8.5, weight="bold", ha="center", va="center", color="#7e5109", zorder=10)

# encoders: 2-D grid-encoder heads in pockets of the platform underside, looking down at grid plates in the base
GRID = 26.0                                      # grid tile side drawn [mm]
for s_ in g.sensors:
    q = s_.point * mm
    tile = Rectangle(q - GRID / 2, GRID, GRID, fc="none", ec=BLUE, lw=1.0, ls="--", zorder=6)
    ax.add_patch(tile)
    for k in np.linspace(-GRID / 2 + 3, GRID / 2 - 3, 7):        # cross grating (symbolic pitch)
        ax.plot([q[0] - GRID / 2, q[0] + GRID / 2], [q[1] + k, q[1] + k], color=BLUE, lw=0.35, alpha=0.6, zorder=6)
        ax.plot([q[0] + k, q[0] + k], [q[1] - GRID / 2, q[1] + GRID / 2], color=BLUE, lw=0.35, alpha=0.6, zorder=6)
    ax.add_patch(Rectangle(q - 3.5, 7, 7, fc=BLUE, ec="#154360", zorder=9))
    arrow(ax, q, q + (11, 0), c=BLUE, lw=1.3, ms=8, z=10); arrow(ax, q, q + (0, 11), c=BLUE, lw=1.3, ms=8, z=10)
    lab = q + np.sign(q) * np.array([-12, -12]) * (-1)
    ax.text(q[0] - np.sign(q[0]) * 2, q[1] - np.sign(q[1]) * (GRID / 2 + 5), s_.name, fontsize=9, weight="bold",
            color=BLUE, ha="center", va="center", zorder=10)
ax.annotate("grid plate (fixed)", xy=(_e * mm + GRID / 2, -_e * mm - GRID / 2 + 3), xytext=(100, -118), fontsize=8,
            color=BLUE, ha="center", arrowprops=dict(arrowstyle="-", color=BLUE, lw=0.6), zorder=12)
ax.annotate("encoder head (moves)", xy=(-_e * mm - 3.5, _e * mm + 3.5), xytext=(-95, 118), fontsize=8,
            color=BLUE, ha="center", arrowprops=dict(arrowstyle="-", color=BLUE, lw=0.6), zorder=12)

# frame, yaw reference
arrow(ax, (0, 0), (32, 0), c=INK, lw=1.5, ms=11, z=11); arrow(ax, (0, 0), (0, 32), c=INK, lw=1.5, ms=11, z=11)
ax.text(33, -7, "x", fontsize=12, style="italic"); ax.text(-8, 31, "y", fontsize=12, style="italic")
ax.add_patch(Circle((0, 0), 4, fc="white", ec=INK, lw=1.3, zorder=12)); ax.add_patch(Circle((0, 0), 1.2, fc=INK, zorder=13))
ax.add_patch(Arc((0, 0), 34, 34, theta1=20, theta2=110, color="#16a085", lw=1.6, zorder=11))
arrow(ax, (17 * np.cos(1.75), 17 * np.sin(1.75)), (17 * np.cos(1.92), 17 * np.sin(1.92)), c="#16a085", lw=1.3, ms=10, z=11)
ax.text(-26, 2, "θz", color="#16a085", fontsize=11)
ax.text(-5, -6, "O = G", fontsize=8, ha="right", va="top")

# dimensions
dim(ax, (-h, -150), (h, -150), f"L = {L:.0f}", ext=[[(-h, -h - 2), (-h, -154)], [(h, -h - 2), (h, -154)]])
dim(ax, (150, 0), (150, a), f"a = {a:.0f}", rot=90, ext=[[(h + 2, a), (154, a)], [(h + 2, 0), (154, 0)]])
dim(ax, (0, 152), (a, 152), f"a = {a:.0f}", ext=[[(a, 148), (a, 156)], [(0, 146), (0, 156)]])
pX1 = g.actuators[0].point * mm
dim(ax, (-h, a - 36), (pX1[0], a - 36), f"{P('reach')*mm:.0f}", fs=7)
# section marker
for xx in (-163, 163):
    ax.text(xx, a + 6, "A", fontsize=12, weight="bold", ha="center")
    arrow(ax, (xx, a), (xx, a - 14), c=INK, lw=1.4, ms=10)
ax.text(-168, 180, "SHEET 1a — TOP VIEW  (mm)", fontsize=12, weight="bold", va="top")
ax.text(-168, 170, "red: coil forces and their lines of action  ·  blue: encoders  ·  dashed: under the platform", fontsize=8, color="#555")

# ── section A-A ──────────────────────────────────────────────────────────────────
fig.savefig(OUT_A, dpi=160, bbox_inches="tight")
fig2 = plt.figure(figsize=(8.6, 4.2))
bx = fig2.add_axes([0.02, 0.02, 0.96, 0.96]); bx.set_aspect("equal"); bx.axis("off")
gap = 3.0                       # drawn gap (real 5 um)
zc = gap + P("z_cog") * mm     # CoG / force plane height
bx.add_patch(Rectangle((-140, -40), 280, 40, fc="#e8e8e8", ec="#888", zorder=0))
bx.text(-136, -34, "base", fontsize=8, color="#666")
bx.add_patch(Rectangle((-h, gap), L, hp, fc="#dfe7ef", ec=INK, lw=1.8, hatch="////", zorder=3))
bx.text(0, gap + hp + 3, "platform", ha="center", fontsize=8.5)
for sx in (-25, 25):              # air bearings at y = ±25 lie in front of the cut plane: ghosted
    bx.add_patch(Rectangle((sx - 15, 0.0), 30, gap, fc="none", ec="#2e86c1", lw=0.7, ls=(0, (2, 2)), zorder=2))
bx.text(0, -30, "air bearings (behind the cut)", fontsize=7.5, ha="center", va="top", color="#2e86c1")
bx.plot([0, -25], [-28, 1], color="#2e86c1", lw=0.5); bx.plot([0, 25], [-28, 1], color="#2e86c1", lw=0.5)
# X1 actuator at the left in section: C-yoke standing on the base, open toward the platform
xc = pX1[0]
H = 27.0
yk = [(xc - 26, 0), (xc + 22, 0), (xc + 22, 6), (xc - 20, 6), (xc - 20, H - 6), (xc + 22, H - 6), (xc + 22, H), (xc - 26, H)]
yk = [(x, y + zc - H / 2) for x, y in yk]
bx.add_patch(Polygon(yk, fc=FE, ec="#2c3e50", zorder=4))
bx.add_patch(Rectangle((xc - 26, 0), 48, zc - H / 2, fc="#aab7b8", ec="#566573", zorder=4))      # yoke pedestal
for x, cb, ct in ((-pitch / 2, NC, SC), (pitch / 2, SC, NC)):
    bx.add_patch(Rectangle((xc + x - w_m / 2, zc - H / 2 + 6), w_m, 5, fc=cb, ec="#333", zorder=5))
    bx.add_patch(Rectangle((xc + x - w_m / 2, zc + H / 2 - 11), w_m, 5, fc=ct, ec="#333", zorder=5))
bx.add_patch(Rectangle((xc - pitch / 2 - w_c / 2, zc - 1.5), -xc - h + pitch / 2 + w_c / 2 + 2, 3, fc=COIL, ec="#7e5109", zorder=6))
arrow(bx, (xc - 4, zc), (xc + 22, zc), lw=2.6, ms=14, z=9)
bx.text(xc - 4, zc + 17, "$f_{X1}$", color=RED, fontsize=12)
bx.text(xc - 29, zc, "coil X1", fontsize=8, ha="right", va="center", color="#7e5109")
bx.plot([-148, 148], [zc, zc], color=RED, lw=0.7, ls=(0, (6, 3)), zorder=1)
bx.text(128, zc + 1.5, "force at\nCoG height", fontsize=7.5, color=RED, va="bottom", ha="center")
bx.plot(0, zc, "o", color="#d35400", ms=7, zorder=8); bx.text(5, zc + 3, "G", color="#d35400", fontsize=9)
# heads E2 (x = -head) and E1 (x = +head) cut by A-A: each sits in a pocket in the platform underside and looks
# DOWN at a grid plate set flush into the base; gap (0.5 mm [A]) drawn enlarged
gz = 2.0                                            # drawn head-face gap above the base surface [mm]
for xh, nm in ((-_e * mm, "E2"), (_e * mm, "E1")):
    bx.add_patch(Rectangle((xh - GRID / 2, -3.5), GRID, 3.5, fc="#d6eaf8", ec=BLUE, lw=0.9, zorder=4))   # grid plate in base
    for k in np.linspace(-GRID / 2 + 2, GRID / 2 - 2, 9):
        bx.plot([xh + k, xh + k], [-3.5, 0], color=BLUE, lw=0.4, zorder=5)
    bx.add_patch(Rectangle((xh - 7, gap), 14, 12, fc="white", ec=INK, lw=0.8, zorder=5))             # pocket
    bx.add_patch(Rectangle((xh - 5.5, gz + 1.5), 11, 10, fc=BLUE, ec="#154360", zorder=7))           # head
    bx.plot([xh - 2, xh - 2], [gz + 1.5, 0], color=BLUE, lw=0.6, ls="--", zorder=6)
    bx.plot([xh + 2, xh + 2], [gz + 1.5, 0], color=BLUE, lw=0.6, ls="--", zorder=6)
    bx.text(xh, gap + 15, nm, fontsize=8.5, color=BLUE, weight="bold", ha="center", zorder=9,
            bbox=dict(fc="white", ec="none", pad=0.3))
bx.annotate("grid plate (fixed)", xy=(_e * mm + 10, -2), xytext=(100, -16), fontsize=7.5, color=BLUE, ha="center",
            va="center", arrowprops=dict(arrowstyle="-", color=BLUE, lw=0.5))
dim(bx, (h + 14, gap), (h + 14, gap + hp), f"{hp:.0f}", rot=90, fs=7)
dim(bx, (h + 30, gap), (h + 30, zc), f"{zc-gap:.1f}", rot=90, fs=7)
bx.set_xlim(-150, 150); bx.set_ylim(-42, 62)
bx.text(-148, 58, "SHEET 1b — SECTION A–A  (mm)", fontsize=11, weight="bold")

fig2.savefig(OUT_B, dpi=160, bbox_inches="tight")
print("saved", OUT_A, OUT_B)
