"""
stage_animation.py — videos of the experiments, drawn like sheet 1a (top view), from the data stage_control.py saved.

    python stage_animation.py            -> pictures/video/anim_exp1.mp4 ... anim_exp5.mp4
    python stage_animation.py 2 4        -> only experiments 2 and 4
    python stage_animation.py 5 --still  -> no video: only the last frame of experiment 5, pictures/video/anim_exp5_still.png
    each video also gets a poster image, pictures/video/poster/anim_expN_poster.jpg (the frame with the largest current)

Magnets: each coil runs between two magnet pairs; the lower pair is drawn solid under the coil, the upper pair
(above the coil) as thin phantom outlines with a cross-hatch, so the coil stays visible. Colours are the face
toward the coil: the upper magnet over an N (red) is S (blue) and vice versa, so the field crosses the gap.
What moves: the platform with everything fixed to it (coil paddles and their arms, encoder heads, air-bearing
pads, the centre G). What stays: the base, magnets and yokes, grid plates, the reference lines.
The real motion is µm-sized, invisible next to a 150 mm plate, so x, y and θz are drawn magnified (the factor is
printed in the frame); the magnification keeps the drawn coils inside their yokes.
Coils: around each coil, in purple, the magnetic field the coil's own current makes — |B_z| in the coil plane from
the Biot–Savart law for the rectangular winding, scaled by the current. The field is drawn down to a fixed
strength, so it reaches further when the coil pushes harder and shrinks to nothing when it does not; the bands
are steps of equal field ratio. The paddle colour also turns purple with |i| / max|i|; the red arrow is the coil
force, its length and sign follow the current. (The field picture follows the coil's position, not its tiny turn.)
The thin purple line is the path of G (the logged q), drawn over the stage.
"""
import sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter
from matplotlib.patches import Rectangle, FancyArrowPatch, Circle, Polygon, ConnectionPatch
from matplotlib.colors import to_rgb

HERE = Path(__file__).resolve().parent
DATA, OUT = HERE / "data", HERE / "pictures" / "video"
POSTER = OUT / "poster"

# ── drawing values: as sheet 1a (plant_sheet.py), in mm ───────────────────────────────────────────────
INK, HID, CL, RED, BLUE, NC, SC, FE = "#1b1b1b", "#555", "#8e44ad", "#c0392b", "#1f6fb5", "#e74c3c", "#3498db", "#5d6d7e"
COIL, PURPLE = "#e59866", "#8e44ad"
w_m, w_c, gapNS, L_m, l_c = 18.0, 10.0, 4.0, 40.0, 30.0
pitch = w_m + gapNS
GRID = 26.0                                   # grid plate tile drawn [mm]
GRID_PITCH = (GRID - 6) / 6                   # spacing of the drawn grid lines; the encoder head is one cell
LIM = 159.0                                   # frame = the base (±158 mm), nothing outside it
DRAWN_MAX = 3.0                               # mm: largest drawn displacement (coils stay inside their yokes)
ANGLE_DRAWN_MAX = 0.015                       # rad: largest drawn turn (~0.9°: the coils move ~2 mm more)

# per experiment: how much slower than real time the video plays, and frames per second
PLAYBACK = {1: 100, 2: 100, 3: 100, 4: 30, 5: 30}
FPS = 30

# coil field picture: FIELD_REACH [mm] is how far from the coil centre the field is drawn at the run's largest
# current; FIELD_GAIN multiplies the purple opacity (1 = as computed, clipped at fully opaque)
FIELD_REACH, FIELD_GAIN = 40.0, 1.6
TOP_MAGNET_ALPHA = 0.9                        # strength of the upper pair's cross-hatch, drawn over the coil
PHANTOM = (0, (8, 2, 1.5, 2, 1.5, 2))         # drafting "phantom" line: long dash, two short dashes


def rot(pts, ang, c=(0.0, 0.0)):
    R = np.array([[np.cos(ang), -np.sin(ang)], [np.sin(ang), np.cos(ang)]])
    return np.asarray(pts, float) @ R.T + np.asarray(c, float)


def rect(cx, cy, hw, hh):
    return np.array([[cx - hw, cy - hh], [cx + hw, cy - hh], [cx + hw, cy + hh], [cx - hw, cy + hh]])


def blend(c0, c1, s):
    c0, c1 = np.array(to_rgb(c0)), np.array(to_rgb(c1))
    return tuple(c0 + (c1 - c0) * s)


def coil_field(ang, half=48.0, n=181, z=2.5, loops=6):
    """|B_z| per unit current of one coil in its own plane (z mm above it), on an n x n grid of ±half mm around the
    coil centre, turned by ang. The winding fills the paddle between the inner and outer rectangle; it is drawn as
    `loops` nested rectangular turns, each summed with Biot–Savart. Normalised to 1 at the coil centre."""
    xs = np.linspace(-half, half, n)
    X, Y = np.meshgrid(xs, xs)
    c, s = np.cos(-ang), np.sin(-ang)                      # grid points into the coil frame
    Xc, Yc = c * X - s * Y, s * X + c * Y
    Bz = np.zeros_like(X)
    for hx, hy in zip(np.linspace(pitch / 2 - w_c / 2, pitch / 2 + w_c / 2, loops),
                      np.linspace(l_c / 2 - 3, l_c / 2 + 3, loops)):
        corners = np.array([[-hx, -hy], [hx, -hy], [hx, hy], [-hx, hy], [-hx, -hy]])
        for p0, p1 in zip(corners[:-1], corners[1:]):
            for u in np.linspace(0, 1, 60, endpoint=False) + 1 / 120:
                m = p0 + (p1 - p0) * u; dl = (p1 - p0) / 60
                rx, ry = Xc - m[0], Yc - m[1]
                Bz += (dl[0] * ry - dl[1] * rx) / (rx**2 + ry**2 + z**2) ** 1.5
    B = np.abs(Bz)
    return B / B[n // 2, n // 2], half


def animate(n, still=False):
    d = np.load(DATA / f"exp{n}.npz")
    t, q, cur = d["t"], d["q"], d["i"]
    L, a, c = float(d["p_L"]) * 1e3, float(d["p_a"]) * 1e3, float(d["p_c"]) * 1e3
    h = L / 2
    name = str(d["name"]).split("  ", 1)[-1]

    # magnification: a round number that draws the largest displacement at about DRAWN_MAX
    reach = max(np.abs(q[:2]).max() * 1e3, 1e-9)
    def round_down(v):                                     # 1, 2, 3, ..., 10, 20, ... never more than v
        e = 10.0 ** np.floor(np.log10(v)); return float(np.floor(v / e) * e)
    mag = round_down(DRAWN_MAX / reach)
    th_reach = np.abs(q[2]).max()
    mag_th = round_down(ANGLE_DRAWN_MAX / th_reach) if th_reach > 1e-9 else mag   # the angle gets its own factor
    i_max = np.abs(cur).max()

    # actuators in the body frame: point (coil centre), force direction
    acts = [("X1", (-c, +a), (1, 0)), ("X2", (+c, -a), (1, 0)), ("Y1", (+a, +c), (0, 1)), ("Y2", (-a, -c), (0, 1))]
    heads = [("E1", (a, a)), ("E2", (-a, a)), ("E3", (-a, -a)), ("E4", (a, -a))]

    fig = plt.figure(figsize=(7.2, 7.2), dpi=150)
    ax = fig.add_axes([0, 0, 1, 1]); ax.set_aspect("equal"); ax.axis("off")
    ax.set_xlim(-LIM, LIM); ax.set_ylim(-LIM, LIM)

    # ── fixed parts ──
    ax.add_patch(Rectangle((-158, -158), 316, 316, fc="#f4f4f4", ec="#999", lw=1.0, zorder=0))
    for xs, ys in [([-146, 146], [0, 0]), ([0, 0], [-146, 146])]:
        ax.plot(xs, ys, color=CL, lw=0.7, ls=(0, (12, 3, 2, 3)), zorder=1)
    for v in (a, -a):
        ax.plot([-146, 146], [v, v], color=RED, lw=0.6, ls=(0, (2, 3)), zorder=1)
        ax.plot([v, v], [-146, 146], color=RED, lw=0.6, ls=(0, (2, 3)), zorder=1)
    for nm, p, dvec in acts:
        ang = np.arctan2(dvec[1], dvec[0])
        for x, col in zip((-pitch / 2, pitch / 2), (NC, SC)):
            ax.add_patch(Polygon(rot(rect(x, 0, w_m / 2, L_m / 2), ang, p), fc=col, ec="#333", lw=0.6, alpha=0.9, zorder=2))
        ax.add_patch(Polygon(rot(rect(0, 0, pitch / 2 + w_m / 2 + 1, L_m / 2 + 1), ang, p), fc="none", ec=FE, lw=1.6, zorder=2))
        for x, col in zip((-pitch / 2, pitch / 2), (SC, NC)):          # upper pair: above the coil, so drawn over it;
            #                                                             opposite face to the coil (S over N, N over S):
            #                                                             a cross-hatch in its colour, thin phantom outline
            ax.add_patch(Polygon(rot(rect(x, 0, w_m / 2, L_m / 2), ang, p), fc="none", ec=col, hatch="xx",
                                 lw=0.0, alpha=TOP_MAGNET_ALPHA, zorder=5.3))
            ax.add_patch(Polygon(rot(rect(x, 0, w_m / 2, L_m / 2), ang, p), fc="none", ec=col, lw=0.7, ls=PHANTOM,
                                 zorder=5.3))
    for nm, s in heads:
        ax.add_patch(Rectangle(np.array(s) - GRID / 2, GRID, GRID, fc="none", ec=BLUE, lw=1.0, ls="--", zorder=6))
        for k in np.linspace(-GRID / 2 + 3, GRID / 2 - 3, 7):
            ax.plot([s[0] - GRID / 2, s[0] + GRID / 2], [s[1] + k, s[1] + k], color=BLUE, lw=0.35, alpha=0.6, zorder=6)
            ax.plot([s[0] + k, s[0] + k], [s[1] - GRID / 2, s[1] + GRID / 2], color=BLUE, lw=0.35, alpha=0.6, zorder=6)
    FancyArrow = lambda p0, p1, **kw: ax.add_patch(FancyArrowPatch(p0, p1, **kw))
    FancyArrow((0, 0), (32, 0), arrowstyle="-|>", mutation_scale=11, color=INK, lw=1.5, zorder=11)
    FancyArrow((0, 0), (0, 32), arrowstyle="-|>", mutation_scale=11, color=INK, lw=1.5, zorder=11)
    ax.text(33, -7, "x", fontsize=11, style="italic"); ax.text(-8, 31, "y", fontsize=11, style="italic")

    # ── moving parts: (patch, body-frame vertices) ──
    movers = []

    def mover(poly, **kw):
        patch = Polygon(poly, **kw); ax.add_patch(patch); movers.append((patch, np.asarray(poly, float))); return patch

    mover(rect(0, 0, h, h), fc="#dfe7ef", ec=INK, lw=2.0, zorder=3)
    pads = []
    for sx in (-1, 1):
        for sy in (-1, 1):
            cp = Circle((sx * 25, sy * 25), 15, fc="none", ec=HID, lw=0.9, ls="--", zorder=4)
            ax.add_patch(cp); pads.append((cp, np.array([sx * 25.0, sy * 25.0])))
    coils, glows, arrows = [], [], []
    for nm, p, dvec in acts:
        p, dvec = np.array(p, float), np.array(dvec, float)
        ang = np.arctan2(dvec[1], dvec[0])
        out = np.sign(p) * (np.abs(p) > h)
        edge = p - out * (abs(p @ out) - h)
        perp = np.array([-out[1], out[0]])
        mover([edge + perp * 7, edge - perp * 7, p - perp * 7 - out * 8, p + perp * 7 - out * 8],
              fc="#aab7b8", ec="#566573", lw=0.8, zorder=4)
        field, half = coil_field(ang)
        img = ax.imshow(np.zeros(field.shape + (4,)), extent=(p[0] - half, p[0] + half, p[1] - half, p[1] + half),
                        origin="lower", interpolation="bilinear", zorder=4.5)
        glows.append((img, field, half, p))
        coils.append(mover(rot(rect(0, 0, pitch / 2 + w_c / 2, l_c / 2 + 3), ang, p), fc=COIL, ec="#7e5109", lw=1.1, zorder=5))
        mover(rot(rect(0, 0, pitch / 2 - w_c / 2, l_c / 2 - 3), ang, p), fc="#f4f4f4", ec="#7e5109", lw=0.8, zorder=5)
        for hx, hy in zip(np.linspace(pitch / 2 - w_c / 2, pitch / 2 + w_c / 2, 7)[1:-1],
                          np.linspace(l_c / 2 - 3, l_c / 2 + 3, 7)[1:-1]):                  # the turns of the winding
            mover(rot(rect(0, 0, hx, hy), ang, p), fc="none", ec="#7e5109", lw=0.45, alpha=0.8, zorder=5.1)
        ar = FancyArrowPatch(p, p, arrowstyle="-|>", mutation_scale=15, color=RED, lw=2.6, zorder=9); ax.add_patch(ar)
        arrows.append((ar, p, dvec))
        po = np.array([0.0, np.sign(p[1])]) if abs(dvec[0]) > 0.5 else np.array([np.sign(p[0]), 0.0])
        ax.text(*(p + po * (L_m / 2 + 14)), f"VC {nm}", fontsize=8, weight="bold", ha="center", va="center",
                color="#7e5109", zorder=10)
    for nm, s in heads:
        mover(rect(s[0], s[1], GRID_PITCH / 2, GRID_PITCH / 2), fc=BLUE, ec="#154360", lw=0.6, zorder=9)
    g_dot = Circle((0, 0), 2.2, fc=INK, ec="white", lw=0.6, zorder=13); ax.add_patch(g_dot)
    trace, = ax.plot([], [], color=PURPLE, lw=1.1, zorder=14)
    ref, = ax.plot(d["r"][0] * 1e3 * mag, d["r"][1] * 1e3 * mag, color="0.55", lw=0.6, ls=":", zorder=13)

    # inset: the path of G at its own scale (µm), in the free corner of the base
    ins = fig.add_axes([0.045, 0.04, 0.145, 0.145])
    ins.set_aspect("equal", adjustable="box")
    for sp in ins.spines.values():
        sp.set_color("#999"); sp.set_linewidth(0.6)
    ins.tick_params(labelsize=5, length=2, pad=1)
    ins.plot(d["r"][0] * 1e6, d["r"][1] * 1e6, color="0.6", lw=0.6, ls=":")
    ins_trace, = ins.plot([], [], color=PURPLE, lw=0.9)
    ins_dot, = ins.plot([], [], "o", color=INK, ms=2.5)
    # zoom box: one square that holds the whole movement (and the reference), set once for the run; the inset shows
    # exactly this square at true scale, the main view marks where it is (×mag), and two lines join the two
    X = np.r_[q[0], d["r"][0]] * 1e6; Y = np.r_[q[1], d["r"][1]] * 1e6
    cx, cy = (X.min() + X.max()) / 2, (Y.min() + Y.max()) / 2
    half = max(X.max() - X.min(), Y.max() - Y.min(), 1e-3) / 2 * 1.15
    ins.set_xlim(cx - half, cx + half); ins.set_ylim(cy - half, cy + half)
    ins.set_title("path of G (µm)", fontsize=6, pad=2)
    k_mm = 1e-3 * mag                                           # µm of real motion -> mm drawn
    bx0, by0, bw = (cx - half) * k_mm, (cy - half) * k_mm, 2 * half * k_mm
    ax.add_patch(Rectangle((bx0, by0), bw, bw, fc="none", ec="#555", lw=0.7, zorder=15))
    for corner_ins, corner_box in (((1, 1), (bx0 + bw, by0 + bw)), ((1, 0), (bx0 + bw, by0))):
        fig.add_artist(ConnectionPatch(xyA=corner_ins, coordsA=ins.transAxes, xyB=corner_box, coordsB=ax.transData,
                                       color="#555", lw=0.6, ls=(0, (4, 3)), zorder=15))

    info = ax.text(-153, 153, "", fontsize=8.5, va="top", ha="left", family="monospace", zorder=20,
                   bbox=dict(fc="white", ec="#999", lw=0.6, pad=3, alpha=0.9))
    i_txt = f"{i_max:.3g} A" if i_max >= 1 else f"{i_max*1e3:.3g} mA"
    ax.text(153, -153, f"motion drawn ×{mag:g}" + (f", angle ×{mag_th:g}" if mag_th != mag else "") + f"   ·   purple: each coil's own field, |B| ∝ i (largest current {i_txt})",
            fontsize=7.5, ha="right", va="bottom", color="#555", zorder=20)

    # field picture: drawn where the field is above B_MIN; at full current it reaches FIELD_REACH from the centre
    f0 = glows[0][1]; kr = int(round((FIELD_REACH / glows[0][2] + 1) / 2 * (f0.shape[1] - 1)))
    B_MIN = f0[f0.shape[0] // 2, kr]
    rgb = np.array(to_rgb(PURPLE))

    def field_rgba(field, s):
        B = field * s
        lev = np.log(np.maximum(B, 1e-12) / B_MIN)                  # 0 at the edge of the drawn field
        a = np.clip(lev / np.log(3 / B_MIN), 0, 1) ** 0.6           # stronger field, denser purple
        bands = 0.72 + 0.28 * np.cos(2 * np.pi * lev / np.log(2))   # one band per doubling of the field
        alpha = np.where(lev > 0, np.minimum(FIELD_GAIN * 0.85 * a * bands, 1.0), 0.0) if s > 0.01 else np.zeros_like(B)
        out = np.empty(B.shape + (4,)); out[..., :3] = rgb; out[..., 3] = alpha
        return out

    # ── frames: real time slowed down by PLAYBACK[n] ──
    t_video = np.arange(0, t[-1] * PLAYBACK[n], 1 / FPS) / PLAYBACK[n]
    idx = np.minimum(np.searchsorted(t, t_video), len(t) - 1)

    def frame(k):
        j = idx[k]
        off = q[:2, j] * 1e3 * mag
        th = q[2, j] * mag_th
        for patch, verts in movers:
            patch.set_xy(rot(verts, th, off))
        for cp, c0 in pads:
            cp.set_center(rot([c0], th, off)[0])
        for m, ((img, field, half, p), coil, (ar, p0, dvec)) in enumerate(zip(glows, coils, arrows)):
            s = abs(cur[m, j]) / i_max
            centre = rot([p], th, off)[0]
            img.set_data(field_rgba(field, s))
            img.set_extent((centre[0] - half, centre[0] + half, centre[1] - half, centre[1] + half))
            coil.set_facecolor(blend(COIL, PURPLE, s))
            tip = rot([p0 + dvec * 40 * cur[m, j] / i_max], th, off)[0]
            ar.set_positions(centre, tip); ar.set_visible(s > 0.02)
        g_dot.set_center(off)
        ins_trace.set_data(q[0, :j + 1] * 1e6, q[1, :j + 1] * 1e6); ins_dot.set_data([q[0, j] * 1e6], [q[1, j] * 1e6])
        trace.set_data(q[0, :j + 1] * 1e3 * mag, q[1, :j + 1] * 1e3 * mag)
        e = np.abs(d["r"][:2, j] - q[:2, j]).max()
        info.set_text(f"experiment {n}: {name}\n"
                      f"t = {t[j]*1e3:7.2f} ms\n"
                      f"x = {q[0, j]*1e6:8.3f} µm   y = {q[1, j]*1e6:8.3f} µm\n"
                      f"|r − q| = {e*1e9:8.2f} nm")
        return []

    # web-friendly MP4: H.264 in yuv420p (plays in every browser), no audio track, quality-based size (CRF 26),
    # and +faststart so the index sits at the start of the file and the video starts before it has fully loaded
    writer = FFMpegWriter(fps=FPS, codec="libx264",
                          extra_args=["-pix_fmt", "yuv420p", "-preset", "veryfast", "-crf", "26",
                                      "-movflags", "+faststart", "-an"])
    path = OUT / f"anim_exp{n}.mp4"
    if still:                                              # check of the drawing: the last frame, full path
        frame(len(idx) - 1)
        fig.savefig(OUT / f"anim_exp{n}_still.png", dpi=150); plt.close(fig)
        print(f"saved {OUT / f'anim_exp{n}_still.png'}  (motion ×{mag:g})")
        return
    FuncAnimation(fig, frame, frames=len(idx), blit=False).save(path, writer=writer, dpi=150)
    # poster: the frame where the coils work hardest, shown before the video plays
    frame(int(np.argmax(np.abs(cur[:, idx]).max(axis=0))))
    fig.savefig(POSTER / f"anim_exp{n}_poster.jpg", dpi=100, pil_kwargs={"quality": 85})
    plt.close(fig)
    print(f"saved {path}  ({len(idx)} frames, {len(idx)/FPS:.1f} s, motion ×{mag:g})")


if __name__ == "__main__":
    POSTER.mkdir(parents=True, exist_ok=True)
    still = "--still" in sys.argv                    # only the poster image, no video (quick check of the drawing)
    for n in [int(a) for a in sys.argv[1:] if a.isdigit()] or [1, 2, 3, 4, 5]:
        animate(n, still)