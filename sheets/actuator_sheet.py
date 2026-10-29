"""
Sheet 2 — voice-coil actuator of the planar stage (coil X1). Standalone. Writes two images:
    <prefix>a_field.png : detail of section A-A with computed flux lines, and the coil from above
    <prefix>b_gain.png  : field along the coil plane, and K_f over the coil position

    python actuator_sheet.py [prefix]        (default prefix "sheet2"; needs numpy + scipy + matplotlib, ~20 s)

What it does (write-up section 5.4)
  1. Covers the 2-D slice with a 0.2 mm grid (lines up with every part edge); each cell is air, steel (mu_r 2000) or magnet (mu_r 1.05, B_r 1.3 T
     up or down). Solves  -div(nu grad A) = d(nu*B_r)/dx  for the vector potential A (A = 0 at the outer
     edge), with B = (dA/dy, -dA/dx). Flux lines are contours of A at equal steps.
  2. Averages B_y over each coil leg -> B_left, B_right; shifts the coil by dx and repeats -> K_f(dx) = N*l*(B_left - B_right).
All values match the write-up (section 5.1); the same solver, stripped down, is vc_field_solver.py.
"""
import sys
import os
from pathlib import Path
from dataclasses import dataclass
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyArrowPatch, Circle, Polygon

PREFIX = sys.argv[1] if len(sys.argv) > 1 else "sheet2"

OUT_DIR = Path("pictures")
OUT_DIR.mkdir(parents=True, exist_ok=True)

OUT_A = OUT_DIR / (PREFIX + "a_field.png")
OUT_B = OUT_DIR / (PREFIX + "b_gain.png")
MU0 = 4e-7 * np.pi

# ── requirements and winding design (assumed, with rationale) ─────────────────────────
TRAVEL = 2.5                  # mm  [A] short-stroke (fine) stage: a few mm of travel per axis
ROT = 1.5e-3                  # rad [A] in-plane rotation range supported by 2-D grid encoders (sheet 1)
WIRE_D = 0.30                 # mm  [A] enamelled copper wire diameter (common size for small voice coils)
FILL = 0.55                   # [-] [A] copper fill factor of a wound rectangular bundle (typ. 0.5-0.65)
RHO_CU = 1.72e-8              # ohm m, copper resistivity at 20 C (material data)
A_LEVER = 55.0                # mm  [A] lever arm of each actuator (sheet 1)
STROKE = TRAVEL + A_LEVER * ROT   # mm each coil must cover along its force


@dataclass
class Geometry:               # all [A] assumed, mm
    w_m: float = 18.0         # magnet width along the force
    t_m: float = 5.0          # magnet thickness
    gap_ns: float = 4.0       # gap between N and S block
    gap: float = 5.0          # magnet face to face: 3 mm coil + 2 x 1 mm clearance
    t_fe: float = 12.0        # back-iron thickness (keeps steel at ~1.4 T)
    x_fe: float = 30.0        # back-iron half length
    w_c: float = 10.0         # coil leg width
    t_c: float = 3.0          # coil thickness
    l_c: float = 30.0         # active leg length (into the page)
    Br: float = 1.3           # NdFeB remanence [T]
    mu_fe: float = 2000.0     # steel relative permeability (linear, no saturation)

    @property
    def pitch(self):
        return self.w_m + self.gap_ns


G = Geometry()
YB0, YT1 = -G.gap / 2 - G.t_m, G.gap / 2 + G.t_m          # outer faces of bottom / top magnets


def regions(X, Y):
    steel = (np.abs(X) <= G.x_fe) & (((Y >= YB0 - G.t_fe) & (Y < YB0)) | ((Y > YT1) & (Y <= YT1 + G.t_fe)))
    steel |= (X >= -G.x_fe - 6) & (X < -G.x_fe) & (Y >= YB0 - G.t_fe) & (Y <= YT1 + G.t_fe)   # back wall
    mags = []
    for xc, sgn in ((-G.pitch / 2, +1), (G.pitch / 2, -1)):      # left column magnetised up, right down
        inx = np.abs(X - xc) <= G.w_m / 2
        mags.append((inx & (Y >= YB0) & (Y <= -G.gap / 2), sgn))
        mags.append((inx & (Y >= G.gap / 2) & (Y <= YT1), sgn))
    return steel, mags


def solve(h=0.2, xl=(-70, 70), yl=(-45, 45)):
    x = np.arange(xl[0], xl[1] + h / 2, h)
    y = np.arange(yl[0], yl[1] + h / 2, h)
    X, Y = np.meshgrid(x, y)
    nu = np.full(X.shape, 1 / MU0)
    Bry = np.zeros(X.shape)
    steel, mags = regions(X, Y)
    nu[steel] = 1 / (MU0 * G.mu_fe)
    for m, sgn in mags:
        nu[m] = 1 / (MU0 * 1.05)
        Bry[m] = sgn * G.Br
    ny, nx = X.shape
    hm = h * 1e-3
    idx = np.arange(nx * ny).reshape(ny, nx)
    fx = 2 / (1 / nu[:, 1:] + 1 / nu[:, :-1])
    fy = 2 / (1 / nu[1:, :] + 1 / nu[:-1, :])
    rows, cols, vals = [], [], []
    diag = np.zeros(nx * ny)
    for p, q, w in ((idx[:, :-1], idx[:, 1:], fx), (idx[:-1, :], idx[1:, :], fy)):
        p, q, w = p.ravel(), q.ravel(), w.ravel() / hm**2
        rows += [p, q]; cols += [q, p]; vals += [-w, -w]
        np.add.at(diag, p, w); np.add.at(diag, q, w)
    K = sp.csr_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))), shape=(nx * ny,) * 2)
    K = K + sp.diags(diag)
    nB = nu * Bry
    S = np.zeros((ny, nx))
    S[:, 1:-1] = ((nB[:, 2:] + nB[:, 1:-1]) / 2 - (nB[:, 1:-1] + nB[:, :-2]) / 2) / hm
    inner = np.ones((ny, nx), bool); inner[0, :] = inner[-1, :] = inner[:, 0] = inner[:, -1] = False
    inner = inner.ravel()
    A = np.zeros(nx * ny)
    A[inner] = spla.spsolve(K[inner][:, inner].tocsc(), S.ravel()[inner])
    A = A.reshape(ny, nx)
    dAdy, dAdx = np.gradient(A, hm, hm)
    return x, y, A, dAdy, -dAdx, steel


x, y, A, Bx, By, steel = solve(h=0.2)
Bm = np.hypot(Bx, By)


from scipy.interpolate import RegularGridInterpolator as _RGI
_fBy = _RGI((y, x), By)
_u, _v = np.meshgrid(np.linspace(-G.w_c / 2, G.w_c / 2, 81), np.linspace(-G.t_c / 2, G.t_c / 2, 13))


def leg_field(xc):
    """Mean vertical field over a winding bundle centred at xc (bilinear sampling, smooth in xc)."""
    return _fBy(np.column_stack([_v.ravel(), (_u + xc).ravel()])).mean()


dxs = np.linspace(-22, 22, 221)
BL = np.array([leg_field(-G.pitch / 2 + d) for d in dxs])
BR = np.array([leg_field(+G.pitch / 2 + d) for d in dxs])
i0 = np.argmin(abs(dxs))
N = int(FILL * G.w_c * G.t_c / (np.pi * WIRE_D**2 / 4))   # turns that fit in one bundle cross-section
KF_PUB = N * G.l_c * 1e-3 * (BL[i0] - BR[i0])             # K_f,0: centred force constant [N/A] (derived)
MLT = 2 * G.l_c + 2 * G.pitch                              # mean turn length [mm]
R_COIL = RHO_CU * N * MLT * 1e-3 / (np.pi * (WIRE_D * 1e-3)**2 / 4)   # coil resistance [ohm]
KF = N * G.l_c * 1e-3 * (BL - BR)

KF_EDGE = np.interp(STROKE, dxs, KF)
DROP = 100 * (1 - KF_EDGE / KF_PUB)
iy0 = np.argmin(abs(y))
By_mid = By[iy0]                                          # field along the coil mid-plane

# ── drawing ────────────────────────────────────────────────────────────────────────
from scipy.interpolate import RegularGridInterpolator
from matplotlib.patches import FancyBboxPatch, ConnectionPatch

INK, RED, FL, CU, CUD = "#1b1b1b", "#c0392b", "#2c1a4d", "#b9770e", "#ca6f1e"
NCOL, SCOL, FORMER, FE, FE_EDGE = "#f5b7b1", "#aed6f1", "#f6e3c5", "#d5d8dc", "#4d5656"
fBx = RegularGridInterpolator((y, x), Bx, bounds_error=False, fill_value=0.0)
fBy = RegularGridInterpolator((y, x), By, bounds_error=False, fill_value=0.0)
x_free = -G.pitch / 2 - G.w_c / 2 - 2.5           # free end of the coil former


def balloon(ax, n, xy, xytext, fs=11):
    """Engineering call-out: numbered balloon with a leader line."""
    ax.annotate(str(n), xy=xy, xytext=xytext, ha="center", va="center", fontsize=fs, weight="bold", color=INK,
                bbox=dict(boxstyle="circle,pad=0.25", fc="white", ec=INK, lw=1.1),
                arrowprops=dict(arrowstyle="-", color=INK, lw=0.8, shrinkA=0, shrinkB=0), zorder=30)
    ax.plot(*xy, "o", ms=3, color=INK, zorder=30)


def winding(ax, xc, into, alpha=0.8, z=9):
    ax.add_patch(Rectangle((xc - G.w_c / 2, -G.t_c / 2), G.w_c, G.t_c, fc=CUD, ec="#784212", lw=1.2, alpha=alpha, zorder=z))
    for xx in np.linspace(xc - G.w_c / 2 + 0.6, xc + G.w_c / 2 - 0.6, 9):
        for yy in (-0.72, 0.72):
            ax.add_patch(Circle((xx, yy), 0.5, fc="#fdebd0", ec="#784212", lw=0.5, zorder=z + 1))
            ax.plot(xx, yy, marker="x" if into else "o", ms=2.6 if into else 1.0, color="#784212", mew=0.8, zorder=z + 2)


def _unused_title_block(fig, rect, rows):
    tb = fig.add_axes(rect); tb.axis("off")
    n = len(rows)
    for k, (key, val) in enumerate(rows[::-1]):
        y0 = k / n
        tb.add_patch(Rectangle((0, y0), 0.22, 1 / n, transform=tb.transAxes, fc="#f2f2f2", ec=INK, lw=0.8))
        tb.add_patch(Rectangle((0.22, y0), 0.78, 1 / n, transform=tb.transAxes, fc="white", ec=INK, lw=0.8))
        tb.text(0.01, y0 + 0.5 / n, key, transform=tb.transAxes, fontsize=9, weight="bold", va="center")
        tb.text(0.23, y0 + 0.5 / n, val, transform=tb.transAxes, fontsize=9.5, va="center")


# ════════════════════════════ IMAGE A: section A-A + coil top view ════════════════════════════
FW, FH = 24, 12.5
figA = plt.figure(figsize=(FW, FH))
XL, YL = (-46, 62), (-27, 27)
WA = 0.66
HA = WA * FW * (YL[1] - YL[0]) / (XL[1] - XL[0]) / FH
ax = figA.add_axes([0.02, 0.93 - HA, WA, HA]); ax.set_aspect("equal"); ax.axis("off")

# steel core: section hatch (ISO style), magnets, flux lines on top
for (x0, y0, w, hh) in [(-G.x_fe, YB0 - G.t_fe, 2 * G.x_fe, G.t_fe), (-G.x_fe, YT1, 2 * G.x_fe, G.t_fe),
                        (-G.x_fe - 6, YB0 - G.t_fe, 6, YT1 - YB0 + 2 * G.t_fe)]:
    ax.add_patch(Rectangle((x0, y0), w, hh, fc=FE, ec="#aab2bd", lw=0, hatch="////", zorder=1))
    ax.add_patch(Rectangle((x0, y0), w, hh, fc="none", ec=FE_EDGE, lw=1.6, zorder=3))
for xc, lb, lt, cb, ct in ((-G.pitch / 2, "N", "S", NCOL, SCOL), (G.pitch / 2, "S", "N", SCOL, NCOL)):
    for y0, lab, c in ((YB0, lb, cb), (G.gap / 2, lt, ct)):
        ax.add_patch(Rectangle((xc - G.w_m / 2, y0), G.w_m, G.t_m, fc=c, ec="#555", lw=1.1, zorder=2))
        ax.text(xc - G.w_m / 2 + 1.0, y0 + G.t_m / 2, lab, color="#333", fontsize=12, weight="bold", va="center", zorder=12,
                bbox=dict(fc=c, ec="none", pad=0.5))
# coil former + arm + windings (drawn below the flux lines so the field through the coil stays visible)
ax.add_patch(Rectangle((x_free, -G.t_c / 2), 44 - x_free, G.t_c, fc=FORMER, ec=CU, lw=1.2, zorder=4))
winding(ax, -G.pitch / 2, True, z=5)
winding(ax, +G.pitch / 2, False, z=5)
sx, sy = (x >= XL[0]) & (x <= XL[1]), (y >= YL[0]) & (y <= YL[1])
Amax = np.abs(A).max()
cs = ax.contour(x[sx], y[sy], A[np.ix_(sy, sx)], levels=np.linspace(-Amax, Amax, 90)[1:-1],
                colors=FL, linewidths=0.5, linestyles="solid", alpha=0.9, zorder=8)
for li, segs in enumerate(cs.allsegs):                      # direction arrows only in the steel core
    if li % 3:
        continue
    xg = (-16.0, 0.0, 16.0)[(li // 3) % 3]
    for seg in segs:
        for k in range(len(seg) - 1):
            (x0, y0), (x1, y1) = seg[k], seg[k + 1]
            if (x0 - xg) * (x1 - xg) < 0 and abs(y0) > YT1 + 0.4:
                p = np.array([xg, y0])
                b = np.array([fBx((p[1], p[0])), fBy((p[1], p[0]))]).ravel()
                u = b / (np.linalg.norm(b) + 1e-12)
                ax.add_patch(FancyArrowPatch(p - 0.9 * u, p + 0.9 * u, arrowstyle="-|>", mutation_scale=10,
                                             color=FL, lw=0.7, zorder=9))
# platform (hatched, as on sheet 1)
ax.add_patch(Rectangle((42, -13), 18, 26, fc="#dfe7ef", ec="#34495e", lw=1.6, hatch="\\\\\\\\", zorder=7))
ax.add_patch(FancyArrowPatch((-6, 0), (6, 0), arrowstyle="-|>", mutation_scale=20, color=RED, lw=3.2, zorder=20))
ax.text(0, 2.2, "F", color=RED, fontsize=15, weight="bold", ha="center", zorder=20,
        bbox=dict(fc="white", ec="none", pad=0.4, alpha=0.9))
for xc, nm in ((-G.pitch / 2, "B_left"), (G.pitch / 2, "B_right")):
    ax.annotate(nm, xy=(xc, YB0 - 0.3), xytext=(xc, -25), ha="center", fontsize=13, weight="bold", color=FL,
                arrowprops=dict(arrowstyle="-", color=FL, lw=1.0, ls="--"), zorder=16,
                bbox=dict(fc="white", ec=FL, boxstyle="round,pad=0.3"))
# centre line, gap dimension
ax.plot([0, 0], [YB0 - G.t_fe - 2, YT1 + G.t_fe + 2], color="#7d3c98", lw=0.8, ls=(0, (10, 3, 2, 3)), zorder=10)
ax.annotate("", xy=(G.pitch / 2 + G.w_m / 2 - 1.5, -G.gap / 2), xytext=(G.pitch / 2 + G.w_m / 2 - 1.5, G.gap / 2),
            arrowprops=dict(arrowstyle="<|-|>", color=INK, lw=1.0), zorder=21)
ax.text(G.pitch / 2 + G.w_m / 2 + 0.3, G.gap / 2 + 0.4, "g", fontsize=13, style="italic", color=INK, zorder=21,
        bbox=dict(fc="white", ec="none", pad=0.3))
# labels (short, direct)
def lab(ax_, txt, xy, xytext, c=INK, ha="center"):
    ax_.annotate(txt, xy=xy, xytext=xytext, ha=ha, va="center", fontsize=12, color=c, zorder=30,
                 arrowprops=dict(arrowstyle="-", color=c, lw=0.8, shrinkA=2, shrinkB=0))
    ax_.plot(*xy, "o", ms=3, color=c, zorder=30)
lab(ax, "steel core (fixed)", (-20, YT1 + G.t_fe - 1.5), (-36, 24))
lab(ax, "magnets (fixed)", (-G.pitch / 2 + 6, G.gap / 2 + G.t_m - 1), (-36, 15))
lab(ax, "coil (moves)", (-G.pitch / 2 - 3, -G.t_c / 2), (-36, -6))
lab(ax, "arm to the platform", (36, G.t_c / 2), (36, 9))
lab(ax, "platform", (55, 10), (55, 20))
ax.set_xlim(*XL); ax.set_ylim(*YL)
figA.text(0.02, 0.985, "SHEET 2a — VOICE COIL X1, DETAIL OF SECTION A–A  (computed flux lines)",
          fontsize=16, weight="bold", va="top")
figA.text(0.02, 0.955, "denser lines = stronger field  ·  ⊗ current into the page  ·  ⊙ current out of the page",
          fontsize=11.5, va="top")

# coil seen from above
R0 = 0.02 + WA + 0.03
cx = figA.add_axes([R0, 0.30, 0.97 - R0, 0.50]); cx.set_aspect("equal"); cx.axis("off")
for xc, c, lb in ((-G.pitch / 2, NCOL, "N"), (G.pitch / 2, SCOL, "S")):
    cx.add_patch(Rectangle((xc - G.w_m / 2, -20), G.w_m, 40, fc=c, ec="#999", lw=0.8, alpha=0.6, zorder=0))
    cx.text(xc, -21.5, f"{lb} magnet below", ha="center", va="top", fontsize=10, color="#555")
cx.add_patch(FancyBboxPatch((-G.pitch / 2 - G.w_c / 2, -G.l_c / 2 - 4), G.pitch + G.w_c, G.l_c + 8,
                            boxstyle="round,pad=0,rounding_size=7", fc=CUD, ec="#784212", lw=1.2, zorder=2))
cx.add_patch(FancyBboxPatch((-G.pitch / 2 + G.w_c / 2, -G.l_c / 2 + 3), G.pitch - G.w_c, G.l_c - 6,
                            boxstyle="round,pad=0,rounding_size=2", fc=FORMER, ec="#784212", lw=1.0, zorder=3))
cx.add_patch(Rectangle((G.pitch / 2 + G.w_c / 2, -3), 22, 6, fc=FORMER, ec=CU, lw=1.0, zorder=1))
for x0, y0, y1 in ((-G.pitch / 2, -8, 8), (G.pitch / 2, 8, -8)):
    cx.add_patch(FancyArrowPatch((x0, y0), (x0, y1), arrowstyle="-|>", mutation_scale=15, color="white", lw=2.2, zorder=5))
cx.text(-G.pitch / 2 + 2.5, 9, "i", color="white", fontsize=13, style="italic", weight="bold", zorder=6)
cx.plot([-30, 40], [0, 0], color=INK, lw=1.0, ls=(0, (10, 3, 2, 3)), zorder=6)
for xx in (-30, 38):
    cx.text(xx, 1.3, "A", fontsize=13, weight="bold")
cx.annotate("", xy=(-G.pitch / 2 - G.w_c / 2 - 3, G.l_c / 2), xytext=(-G.pitch / 2 - G.w_c / 2 - 3, -G.l_c / 2),
            arrowprops=dict(arrowstyle="<|-|>", color="#555", lw=0.8))
cx.text(-G.pitch / 2 - G.w_c / 2 - 5, 0, "l", fontsize=14, color="#555", ha="right", va="center", style="italic")
cx.text(33, 5, "arm", fontsize=10, ha="center", color="#784212")
cx.set_xlim(-34, 42); cx.set_ylim(-27, 23)
cx.set_title("COIL FROM ABOVE", loc="left", fontsize=13, weight="bold")
figA.savefig(OUT_A, dpi=200, bbox_inches="tight")

# ════════════════════════════ IMAGE B: field along the coil, K_f over the coil position ════════════════════════════
from matplotlib.lines import Line2D
figB = plt.figure(figsize=(16, 11))
XB = (-36, 36)
figB.text(0.06, 0.975, "SHEET 2b — FIELD UNDER THE COIL AND FORCE CONSTANT", fontsize=16, weight="bold", va="top")
px = figB.add_axes([0.06, 0.53, 0.90, 0.38])
Bpk = np.abs(By_mid).max()
for xc, c, lb in ((-G.pitch / 2, NCOL, "N magnets"), (G.pitch / 2, SCOL, "S magnets")):
    px.axvspan(xc - G.w_m / 2, xc + G.w_m / 2, color=c, alpha=0.45)
    px.text(xc, -1.13, lb, ha="center", fontsize=12, color="#555", va="bottom")
for xc in (-G.pitch / 2, G.pitch / 2):
    px.add_patch(Rectangle((xc - G.w_c / 2, -0.98), G.w_c, 1.96, fc=CUD, alpha=0.25, ec="#784212", lw=1.4))
    px.add_patch(Rectangle((xc - G.w_c / 2 + STROKE, -0.98), G.w_c, 1.96, fc="none", ec=RED, lw=1.8, ls="--"))
px.plot(x, By_mid / Bpk, color=FL, lw=3)
px.axhline(0, color="#999", lw=0.8)
px.text(-G.pitch / 2, 0.5, "B_left", ha="center", fontsize=14, weight="bold", color=FL,
        bbox=dict(fc="white", ec=FL, boxstyle="round,pad=0.3"))
px.text(G.pitch / 2, -0.5, "B_right", ha="center", fontsize=14, weight="bold", color=FL,
        bbox=dict(fc="white", ec=FL, boxstyle="round,pad=0.3"))
px.set_yticks([-1, 0, 1]); px.set_yticklabels(["down", "0", "up"], fontsize=12)
px.set_xticks([0]); px.set_xticklabels(["coil centre"], fontsize=12)
px.set_xlim(*XB); px.set_ylim(-1.18, 1.18); px.grid(alpha=0.3)
px.set_ylabel("vertical field B_y", fontsize=13)
px.legend(handles=[Line2D([], [], color=FL, lw=3, label="field along the coil plane"),
                   Rectangle((0, 0), 1, 1, fc=CUD, alpha=0.4, ec="#784212", label="coil legs, centred"),
                   Rectangle((0, 0), 1, 1, fc="none", ec=RED, ls="--", lw=1.8, label="coil legs at the end of the stroke")],
          loc="upper right", fontsize=11, framealpha=0.95)
px.set_title("FIELD ALONG THE COIL PLANE", loc="left", fontsize=13, weight="bold")

kx = figB.add_axes([0.06, 0.07, 0.90, 0.36])
kx.axvspan(-STROKE, STROKE, color="#d5f5e3", zorder=0)
kx.plot(dxs, KF / KF_PUB, color=RED, lw=3)
kx.axhline(1, color="#555", ls=":", lw=1)
kx.text(0, 0.12, "stroke used", ha="center", fontsize=12, color="#1e8449")
_z = dxs[(KF > 0)]; kx.set_xlim(_z[0], _z[-1]); kx.set_ylim(0, 1.12)
kx.set_yticks([0, 1]); kx.set_yticklabels(["0", "K_f,0"], fontsize=12)
kx.set_xticks([-STROKE, 0, STROKE]); kx.set_xticklabels(["−Δx_max", "centred", "+Δx_max"], fontsize=12)
kx.set_xlabel("coil position Δx", fontsize=13)
kx.set_ylabel("force per ampere K_f", fontsize=13)
kx.grid(alpha=0.3)
kx.set_title("FORCE CONSTANT OVER THE COIL POSITION", loc="left", fontsize=13, weight="bold")
figB.savefig(OUT_B, dpi=160, bbox_inches="tight")
print(f"saved {OUT_A}, {OUT_B}:  B_left={BL[i0]:.3f} T  B_right={BR[i0]:.3f} T  N={N}  K_f,0={KF_PUB:.2f} N/A  "
      f"R={R_COIL:.1f} ohm  K_f drop at ±{STROKE:.2f} mm = {DROP:.1f}%")
