"""
stage_control.py — controlling the stage, one step at a time. Uses the plant from stage_model.py.

Four experiments, all built from the same blocks:

    1  step + PIDF           go to a point: the reference jumps to the goal, a PID with a filtered D pulls the plate there
    2  path + PIDF           the reference moves to the goal along a minimum-jerk curve instead of jumping
    3  path + PIDF + FF      the model's feedforward supplies the force the path needs; the PIDF only corrects
    4  spiral, as 3          a sequence of targets: a spiral out from the centre to the r = 1 mm circle, then once
                             around it. The master path (all minimum-jerk segments one after the other) is generated
                             before the run and read by the loop every control cycle.
    5  raster scan, as 3     the plate starts turned by 0.3 mrad and is brought to θz = 0 on the first move; then it
                             scans the box x ∈ [-2, 2] mm, y ∈ [-1, 1] mm in columns, and back in rows: a grid.
                             Each segment's time follows from a current limit (the rule of write-up 7.4).

Outputs: pictures/  block diagrams, one figure per experiment, the 1-2-3 comparison, the minimum-jerk curve,
                    PID vs PIDF with a noisy measurement
         data/      one .npz per experiment with every logged signal, the settings and the stage parameters

Blocks (all run every Ts, like a digital controller):

    reference    step:  r_k = q*                      path:  r_k = q0 + (q* - q0) s(t_k / T),  s(τ) = 10τ³ - 15τ⁴ + 6τ⁵
                 master path: segment j (t in [j T_seg, (j+1) T_seg)) is the path from target j to target j+1
    PIDF         f_fb,k = Kp e_k + Ki w_k + Kd v_k,   e_k = r_k - q_k,
                 w_k = w_{k-1} + Ts e_k                                                   (integral)
                 v_k = α v_{k-1} + (1 - α)(e_k - e_{k-1}) / Ts,   α = exp(-Ts ω_c)        (first-order filtered D)
    feedforward  f_ff,k = M r'' + D r'   (the model, evaluated mid-sample so the held force equals the path's average)
    allocation   i_k = A(θz,k) (f_ff,k + f_fb,k) / K_f0                                  (write-up 6.4 read backwards)
    hold         i(t) = i_k for t_k <= t < t_{k+1}
    measurement  ideal: the controller reads the plant output q(t_k) directly. This is the control-law design
                 stage; the real sensor (encoder resolution, noise, delay) is checked later, at deployment.

Gains: found by trial and error on the step and kept because they work well; they are written per unit mass
(m, or J for θz) so x, y and θz respond alike. D filter at 1 kHz.
With an I term a step always overshoots: at rest no force is needed, so Ki ∫e dt must return to 0 — the error
has to change sign. Another reason to use a path.
"""
from pathlib import Path
import numpy as np
import control as ct
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Circle, FancyArrowPatch, Rectangle
from stage_model import StageParams, stage, in_range

# ── settings ───────────────────────────────────────────────────────────────────────────
Ts = 5e-5                                   # s     sample time, 20 kHz                               [A]
F_0 = 100.0                                 # Hz    sets how fast the loop is, same for x, y, θz
_W, _MASS = 2 * np.pi * F_0, np.array([StageParams().m, StageParams().m, StageParams().J])
KP = 3 * _MASS * _W**2                      # N/m, N/m, N m/rad
KD = 3 * _MASS * _W                         # N s/m, N s/m, N m s/rad
KI = _MASS * _W**3                          # N/(m s), N/(m s), N m/(rad s)
F_C = 10 * F_0                              # Hz    D-filter cut-off: 10x above the poles, 10x below Nyquist
PRECISION = 20e-9                           # m     positioning spec, shown as a band on the plots      [A]
GOAL = np.array([20e-6, 10e-6, 0.0])        # small move: go to (x, y) = (20, 10) µm, keep θz = 0
R_SPIRAL, TURNS_OUT, PTS_PER_TURN = 1e-3, 2, 32   # spiral targets: out to r = 1 mm in 2 turns, then 1 turn on the circle
T_SEG = 10e-3                               # s     time for each segment of the master path
SCAN_X, SCAN_Y, SCAN_STEP = (-2e-3, 2e-3), (-1e-3, 1e-3), 0.25e-3   # raster box [m] and line spacing   [A]
THETA0 = 0.3e-3                             # rad   initial turn of the plate in experiment 5        [A]
I_LIMIT, T_SEG_MIN = 0.5, 5e-3              # A, s  current a segment may ask for; shortest segment     [A]
T_MOVE, T_END = 20e-3, 50e-3                # s     path duration, simulated time
OUT = Path(__file__).resolve().parent / "pictures"


# ── blocks ─────────────────────────────────────────────────────────────────────────────
def step_reference(goal):
    """r(t) = goal for all t; speed and acceleration zero."""
    g = np.asarray(goal, float)
    return lambda t: (g, np.zeros(3), np.zeros(3))


def path_reference(q0, goal, T):
    """Minimum-jerk path from q0 to goal in time T: r, r', r'' at time t."""
    q0, d = np.asarray(q0, float), np.asarray(goal, float) - np.asarray(q0, float)

    def r(t):
        if t >= T:
            return q0 + d, np.zeros(3), np.zeros(3)
        u = max(t, 0.0) / T
        return (q0 + d * (10 * u**3 - 15 * u**4 + 6 * u**5),
                d * (30 * u**2 - 60 * u**3 + 30 * u**4) / T,
                d * (60 * u - 180 * u**2 + 120 * u**3) / T**2)
    return r


class PIDF:
    """PID with a first-order filter on the D term, per axis: f = Kp e + Ki w + Kd v, w = ∫e dt, v = filtered de/dt.
    f_c = np.inf gives α = 0: no filter, the plain PID with v = (e_k - e_{k-1}) / Ts."""

    def __init__(self, kp=KP, ki=KI, kd=KD, f_c=F_C):
        self.kp, self.ki, self.kd, self.alpha = kp, ki, kd, np.exp(-Ts * 2 * np.pi * f_c)
        self.e_prev, self.w, self.v = None, np.zeros(3), np.zeros(3)

    def __call__(self, e):
        if self.e_prev is None:                     # first sample: no difference yet (no kick from the step)
            self.e_prev = e
        self.w = self.w + Ts * e
        self.v = self.alpha * self.v + (1 - self.alpha) * (e - self.e_prev) / Ts
        self.e_prev = e
        return self.kp * e + self.ki * self.w + self.kd * self.v


def spiral_targets(R=R_SPIRAL, turns_out=TURNS_OUT, n=PTS_PER_TURN):
    """q*_j on an Archimedean spiral from the centre to radius R in `turns_out` turns, then one turn at radius R."""
    phi = 2 * np.pi * np.arange((turns_out + 1) * n + 1) / n
    rad = np.minimum(phi / (2 * np.pi * turns_out), 1.0) * R
    return np.column_stack([rad * np.cos(phi), rad * np.sin(phi), np.zeros_like(phi)])


def scan_targets(x=SCAN_X, y=SCAN_Y, step=SCAN_STEP, theta0=THETA0):
    """q*_j of the raster: from the start (0, 0, θ0) to the corner (x0, 0, 0), up to (x0, y1), then columns
    (down, step, up, step, ...) until x1, then rows back (step in y, across, step, across, ...): a grid."""
    xs, ys = np.arange(x[0], x[1] + step / 2, step), np.arange(y[0], y[1] + step / 2, step)
    pts = [(0, 0, theta0), (x[0], 0, 0)]
    for k, xk in enumerate(xs):                                # columns, alternately down and up
        y_from, y_to = (y[1], y[0]) if k % 2 == 0 else (y[0], y[1])
        pts += [(xk, y_from, 0), (xk, y_to, 0)]
    for k, yk in enumerate(ys[::-1] if pts[-1][1] == y[1] else ys):   # rows back, starting at the end we are on
        x_from, x_to = (x[1], x[0]) if k % 2 == 0 else (x[0], x[1])
        pts += [(x_from, yk, 0), (x_to, yk, 0)]
    pts = np.array(pts, float)
    keep = np.r_[True, np.any(np.abs(np.diff(pts, axis=0)) > 1e-12, axis=1)]   # drop repeated points
    return pts[keep]


def segment_times(targets, p: StageParams, i_max=I_LIMIT, t_min=T_SEG_MIN):
    """T_j for each segment from the current limit (write-up 7.4): peak acceleration 5.77 d / T² must not need more
    than i_max in each coil — two coils per axis for x and y, all four (lever a) for θz."""
    d = np.abs(np.diff(targets, axis=0))
    T_xy = np.sqrt(5.77 * d[:, :2].max(axis=1) * p.m / (2 * p.Kf0 * i_max))
    T_th = np.sqrt(5.77 * d[:, 2] * p.J / (4 * p.a * p.Kf0 * i_max))
    return np.maximum(np.maximum(T_xy, T_th), t_min)


def master_path(targets, T_seg, T_end):
    """The master path: minimum-jerk segments from target j to target j+1, one after the other (it stops at each).
    T_seg: one time for all segments, or one per segment. Generated once, before the run, as a table every Ts/2
    (so the feedforward can read mid-sample too); the loop only looks it up."""
    T = np.broadcast_to(np.asarray(T_seg, float), (len(targets) - 1,))
    start = np.r_[0.0, np.cumsum(T)]
    segs = [path_reference(a, b, Tj) for a, b, Tj in zip(targets[:-1], targets[1:], T)]
    h, n = Ts / 2, int(round(T_end / (Ts / 2))) + 2
    table = []
    for k in range(n):
        j = min(int(np.searchsorted(start, k * h, side="right")) - 1, len(segs) - 1)
        table.append(segs[j](k * h - start[j]))
    return lambda t: table[min(int(round(t / h)), n - 1)]


def feedforward(ref, t, p: StageParams):
    """f_ff = M r'' + D r', taken at the middle of the coming sample period."""
    _, dr, ddr = ref(t + Ts / 2)
    return np.array([p.m, p.m, p.J]) * ddr + np.array([p.b, p.b, p.b_th]) * dr


def allocation(th, p: StageParams):
    """A(θz): 4x3, wanted (Fx, Fy, Mz) -> coil forces. Rotate into the plate frame, then split."""
    c, s, k = np.cos(th), np.sin(th), 1 / (4 * p.a)
    split = np.array([[0.5, 0, -k], [0.5, 0, k], [0, 0.5, k], [0, 0.5, -k]])
    return split @ np.array([[c, s, 0], [-s, c, 0], [0, 0, 1]])


# ── the loop ───────────────────────────────────────────────────────────────────────────
def simulate(ref, use_ff, p: StageParams = StageParams(), T_end=T_END, pid=None, noise=0.0, seed=0, z0=None):
    """Run the sampled loop: read q, PIDF (+ feedforward), allocate, hold, let the plant move one Ts.
    noise: standard deviation [m] of random noise added to the measured x, y (and noise / 2a to θz) — 0 by default;
    used only to show why the D term needs its filter."""
    plant, rng = stage(p), np.random.default_rng(seed)
    pid = PIDF() if pid is None else pid
    sigma = noise * np.array([1.0, 1.0, 1 / (2 * p.a)])
    n = int(round(T_end / Ts)); t = np.arange(n + 1) * Ts
    z = np.zeros(6) if z0 is None else np.asarray(z0, float).copy()     # start state (x, y, θz, speeds)
    names = ("q", "qd", "q_meas", "r", "rd", "rdd", "i", "f_ff", "f_fb")
    log = {k: np.zeros((4 if k == "i" else 3, n + 1)) for k in names}
    for k in range(n):
        q = z[:3] + sigma * rng.standard_normal(3)          # measure (ideal unless noise > 0)
        r, rd, rdd = ref(t[k])
        f_fb = pid(r - q)                                   # PIDF
        f_ff = feedforward(ref, t[k], p) if use_ff else np.zeros(3)
        i = allocation(q[2], p) @ (f_ff + f_fb) / p.Kf0     # currents
        for key, val in zip(names, (z[:3], z[3:], q, r, rd, rdd, i, f_ff, f_fb)):
            log[key][:, k] = val
        z = ct.input_output_response(plant, [0, Ts], np.column_stack([i, i]), X0=z, return_x=True,
                                     solve_ivp_kwargs=dict(rtol=1e-10, atol=1e-16)).states[:, -1]   # hold, plant moves
    log["q"][:, -1], log["qd"][:, -1], log["q_meas"][:, -1] = z[:3], z[3:], z[:3]
    log["r"][:, -1], log["rd"][:, -1], log["rdd"][:, -1] = ref(t[-1])
    for key in ("i", "f_ff", "f_fb"):
        log[key][:, -1] = log[key][:, -2]
    return t, log


def save_experiment(path, t, o, p: StageParams, **settings):
    """Everything about one run in one .npz: the signals, the controller settings and the stage parameters.
    Signals (rows x, y, θz unless noted):  q, qd  true pose and speed      q_meas  what the controller read
        r, rd, rdd  reference and its derivatives      e = r - q      i  coil currents (X1, X2, Y1, Y2)
        f_coil = K_f0 i  coil forces      f_ff, f_fb  feedforward and feedback (Fx, Fy, Mz)      P_coil = R i²"""
    stage_p = {f"p_{k}": getattr(p, k) for k in ("L", "a", "c", "m", "J", "b", "b_th", "Kf0", "R", "travel", "rot")}
    np.savez(path, t=t, **o, e=o["r"] - o["q"], f_coil=p.Kf0 * o["i"], P_coil=p.R * o["i"]**2,
             p_heads=p.heads, p_B0=p.B0, Ts=Ts, KP=KP, KI=KI, KD=KD, F_C=F_C, **stage_p, **settings)


# ── block diagrams ─────────────────────────────────────────────────────────────────────
def block_diagram(kind, path):
    """kind: 'step' (A), 'path' (B) or 'path_ff' (C)."""
    fig, ax = plt.subplots(figsize=(11.5, 4.0)); ax.set_xlim(0, 22.2); ax.set_ylim(0, 7.2); ax.axis("off")
    ax.set_aspect("equal")

    def box(x, y, w, h, txt, ls="-"):
        ax.add_patch(FancyBboxPatch((x - w / 2, y - h / 2), w, h, boxstyle="square,pad=0", fc="white", ec="k", lw=0.8, ls=ls))
        ax.text(x, y, txt, ha="center", va="center")

    def summ(x, y, signs):
        ax.add_patch(Circle((x, y), 0.22, fc="white", ec="k", lw=0.8))
        for dx, dy, s in signs:
            ax.text(x + dx, y + dy, s, ha="center", va="center", fontsize=8)

    def arr(pts, lab=None, at=0.5, off=(0, 0.22), unit=None, uoff=0.27):
        for (x0, y0), (x1, y1) in zip(pts[:-2], pts[1:-1]):
            ax.plot([x0, x1], [y0, y1], "k", lw=0.8)
        ax.add_patch(FancyArrowPatch(pts[-2], pts[-1], arrowstyle="-|>", mutation_scale=9, color="k", lw=0.8))
        if lab:
            (x0, y0), (x1, y1) = pts[0], pts[1]
            xl, yl = x0 + (x1 - x0) * at + off[0], y0 + (y1 - y0) * at + off[1]
            ax.text(xl, yl, lab, ha="center", va="center", fontsize=8)
            if unit:
                ax.text(xl, yl - uoff, unit, ha="center", va="center", fontsize=6, color="0.25")

    ax.add_patch(Rectangle((0.1, 0.4), 12.3, 6.3, fc="none", ec="0.45", lw=0.7, ls=(0, (4, 3))))
    ax.text(0.25, 6.45, r"digital controller — runs every $T_s$", fontsize=8, color="0.35")
    ax.add_patch(Rectangle((12.85, 0.4), 9.2, 6.3, fc="none", ec="0.45", lw=0.7, ls=(0, (1, 2))))
    ax.text(13.0, 6.45, "physical system", fontsize=8, color="0.35")

    ref_txt = "goal\n(step)\n$\\vec r = \\vec q^{\\,*}$" if kind == "step" else ("path\n(minimum jerk)\n$\\vec r,\\ \\dot{\\vec r},\\ \\ddot{\\vec r}$" if kind == "path_ff" else "path\n(minimum jerk)\n$\\vec r(t)$")
    box(1.45, 3.0, 2.5, 1.2, ref_txt)
    summ(4.3, 3.0, [(-0.4, -0.3, "+"), (0.3, -0.45, "−")])
    box(6.35, 3.0, 3.0, 1.0, "PIDF\n$\\left(K_p + \\frac{K_i}{s} + \\frac{K_d\\, s}{\\tau s + 1}\\right) \\vec e$")
    arr([(2.7, 3.0), (4.08, 3.0)], "$\\vec r$", at=0.5, off=(0, 0.55), unit="[m, m, rad]")
    arr([(4.52, 3.0), (4.85, 3.0)], "$\\vec e$", off=(0, 0.3))
    if kind == "path_ff":
        box(6.3, 5.3, 3.0, 0.9, "feedforward\n$\\vec f_{ff} = M\\ddot{\\vec r} + D\\dot{\\vec r}$")
        summ(8.6, 3.0, [(-0.45, -0.3, "+"), (0.3, 0.45, "+")])
        arr([(2.45, 3.6), (2.45, 5.3), (4.8, 5.3)], r"$\dot{\vec r},\ \ddot{\vec r}$", at=0.5, off=(-1.3, 0.35), unit="[m/s, m/s, rad/s],\n[m/s², m/s², rad/s²]", uoff=0.42)
        arr([(7.8, 5.3), (8.6, 5.3), (8.6, 3.22)], "$\\vec f_{ff}$", at=0.5, off=(1.25, -0.9), unit="[N, N, N·m]")
        arr([(7.85, 3.0), (8.38, 3.0)], "$\\vec f_{fb}$", at=0.5, off=(0, 0.3))
        arr([(8.82, 3.0), (9.5, 3.0)], "$\\vec f$", off=(0, 0.3))
    else:
        arr([(7.85, 3.0), (9.5, 3.0)], "$\\vec f$", at=0.5, off=(0, 0.55), unit="[N, N, N·m]")
    box(10.8, 3.0, 2.6, 1.0, f"allocation\n$\\vec i = A(\\theta_z)\\,\\vec f\\,/\\,K_{{f,0}}$")
    arr([(12.1, 3.0), (13.2, 3.0)], "$\\vec i_k$", at=0.55, off=(0, 0.55), unit="[A]×4")
    box(14.3, 3.0, 2.2, 1.2, "current\namplifiers\n+ hold")
    arr([(15.4, 3.0), (16.15, 3.0)], "$\\vec i(t)$", at=0.5, off=(0, 0.95), unit="[A]×4")
    box(17.85, 3.0, 3.4, 1.2, "PLANT")
    ax.plot([19.55, 21.4, 21.4], [3.0, 3.0, 1.35], "k", lw=0.8)
    ax.text(20.5, 3.55, "$\\vec q$", ha="center", va="center", fontsize=8)
    ax.text(20.5, 3.28, "[m, m, rad]", ha="center", va="center", fontsize=6, color="0.25")
    arr([(21.4, 1.35), (4.3, 1.35), (4.3, 2.78)], r"$\vec q$, sampled every $T_s$", at=0.3, off=(0, 0.55), unit="[m, m, rad]")
    arr([(10.8, 1.35), (10.8, 2.5)], r"$\theta_z$", at=0.5, off=(-0.45, 0.13), unit="[rad]")
    ax.plot(10.8, 1.35, "ko", ms=2.5)
    title = {"step": "Experiment 1 — step + PIDF", "path": "Experiment 2 — path + PIDF",
             "path_ff": "Experiment 3 — path + PIDF + feedforward"}[kind]
    ax.text(0.1, 6.9, title, fontsize=10, weight="bold")
    fig.savefig(path, dpi=200, bbox_inches="tight"); plt.close(fig)


# ── plotting ───────────────────────────────────────────────────────────────────────────
def style():
    plt.rcParams.update({
        "font.family": "serif", "font.serif": ["DejaVu Serif"], "mathtext.fontset": "dejavuserif",
        "font.size": 9, "axes.labelsize": 9, "axes.titlesize": 9, "legend.fontsize": 8,
        "axes.spines.top": False, "axes.spines.right": False, "axes.linewidth": 0.6,
        "xtick.direction": "in", "ytick.direction": "in", "xtick.major.width": 0.6, "ytick.major.width": 0.6,
        "lines.linewidth": 1.0, "axes.grid": True, "grid.linewidth": 0.3, "grid.alpha": 0.5,
        "legend.frameon": False, "savefig.bbox": "tight",
    })


def angle_panel(a, tm, o):
    """θz in nrad (µrad when it moves by more than 1 µrad); the y-scale follows the data, at least ±1 nrad so an
    exact 0 shows as a flat line, not as noise. Returns the unit."""
    unit, k = ("µrad", 1e6) if np.abs(o["q"][2]).max() > 1e-6 else ("nrad", 1e9)
    a.plot(tm, o["r"][2] * k, color="0.6", lw=0.7)
    a.plot(tm, o["q"][2] * k, color="k", lw=0.8, label=r"$\theta_z$")
    lim = max(1.2 * np.abs(o["q"][2]).max() * k, 1.0)
    a.set_ylim(-lim, lim); a.legend(loc="upper right")
    return unit


def plot_results(runs, path, pos=("µm", 1e6), cur=("mA", 1e3)):
    fig, ax = plt.subplots(len(runs), 4, figsize=(12.0, 2.2 * len(runs) + 0.4), sharex=True, constrained_layout=True,
                           squeeze=False)
    for row, (name, (t, o)) in enumerate(runs.items()):
        tm, e = t * 1e3, o["r"] - o["q"]
        a = ax[row, 0]
        for k, (lab, ls) in enumerate(zip((r"$x$", r"$y$"), ("-", "--"))):
            a.plot(tm, o["r"][k] * pos[1], ls, color="0.6", lw=0.7)
            a.plot(tm, o["q"][k] * pos[1], ls, color="k", label=lab)
        a.set_ylabel(name); a.legend(loc="lower right")
        a = ax[row, 1]
        ff = "FF" in name                       # C's error is far below 1 µm: shown in nm, without the band
        if not ff:
            a.axhspan(-PRECISION * 1e6, PRECISION * 1e6, color="0.88", lw=0)   # ±20 nm positioning spec
        if "path" in name:
            a.axvspan(0, T_MOVE * 1e3, color="0.94", lw=0, zorder=0)          # while the path moves
        for k, (lab, ls) in enumerate(zip((r"$r_x - x$", r"$r_y - y$"), ("-", "--"))):
            a.plot(tm, e[k] * (1e9 if ff else 1e6), ls, color="k", label=lab + (" (nm)" if ff else ""))
        a.legend(loc="best")
        a = ax[row, 2]
        for k, (lab, ls) in enumerate(zip(("$i_{X1}, i_{X2}$", "$i_{Y1}, i_{Y2}$"), ("-", "--"))):
            a.step(tm, o["i"][2 * k] * cur[1], ls, color="k", where="post", label=lab)
        a.legend(loc="upper right")
        angle_panel(ax[row, 3], tm, o)
    for a in ax[-1]:
        a.set_xlabel("time [ms]")
    ax[0, 0].set_title(f"position ({pos[0]}), grey: reference"); ax[0, 1].set_title("tracking error (µm), band: ±20 nm, shaded: path moving"); ax[0, 2].set_title(f"coil currents ({cur[0]})")
    ax[0, 3].set_title("angle θz (nrad), grey: reference")
    fig.savefig(path, dpi=200); plt.close(fig)


def plot_spiral(runs, targets, path):
    """Experiments 4 and 5: the plate's path in the plane, and error, currents and angle over time."""
    fig, ax = plt.subplots(len(runs), 4, figsize=(12.0, 2.9 * len(runs)), constrained_layout=True, squeeze=False)
    for row, (name, (t, o)) in enumerate(runs.items()):
        tm, e = t * 1e3, o["r"] - o["q"]
        ff = "FF" in name                       # with feedforward the error is far below 1 µm: shown in nm
        a = ax[row, 0]
        a.plot(o["r"][0] * 1e3, o["r"][1] * 1e3, color="0.6", lw=2.0)
        a.plot(o["q"][0] * 1e3, o["q"][1] * 1e3, color="k", lw=0.6)
        a.plot(targets[:, 0] * 1e3, targets[:, 1] * 1e3, "o", ms=1.4, mfc="white", mec="k", mew=0.4)
        a.set_aspect("equal"); a.set_xlabel("x [mm]"); a.set_ylabel(name)
        a = ax[row, 1]
        for k, ls in enumerate(("-", "--")):
            a.plot(tm, e[k] * (1e9 if ff else 1e6), ls, color="k", lw=0.6)
        a.set_title(f"tracking error ({'nm' if ff else 'µm'}): — x, - - y")
        a = ax[row, 2]
        for k, ls in enumerate(("-", "--")):
            a.step(tm, o["i"][2 * k], ls, color="k", lw=0.6, where="post")
        unit = angle_panel(ax[row, 3], tm, o)
        for a in ax[row, 1:]:
            a.set_xlabel("time [ms]")
    ax[0, 0].set_title("plane (mm), grey: master path, o: targets")
    ax[0, 2].set_title("coil currents (A): — X coils, - - Y coils")
    ax[0, 3].set_title(f"angle θz ({unit}), grey: reference")
    fig.savefig(path, dpi=200); plt.close(fig)


def plot_min_jerk(path):
    """s(τ) = 10τ³ - 15τ⁴ + 6τ⁵ and its first three derivatives, over the normalised time τ = t / T."""
    u = np.linspace(0, 1, 401)
    curves = ((10 * u**3 - 15 * u**4 + 6 * u**5, "position  s(τ)"),
              (30 * u**2 - 60 * u**3 + 30 * u**4, "speed  ds/dτ"),
              (60 * u - 180 * u**2 + 120 * u**3, "acceleration  d²s/dτ²"),
              (60 - 360 * u + 360 * u**2, "jerk  d³s/dτ³"))
    fig, ax = plt.subplots(1, 4, figsize=(12.0, 2.3), constrained_layout=True)
    for a, (c, title) in zip(ax, curves):
        a.plot(u, c, "k"); a.set_xlabel(r"$\tau = t / T$"); a.set_title(title)
    fig.savefig(path, dpi=200); plt.close(fig)


def plot_pid_vs_pidf(runs, path, window=(25e-3, 50e-3)):
    """The same step with a noisy measurement: plain PID (D unfiltered) against PIDF."""
    fig, ax = plt.subplots(1, 2, figsize=(10.0, 2.6), constrained_layout=True)
    for (name, (t, o)), col in zip(runs.items(), ("0.6", "k")):
        ax[0].plot(t * 1e3, o["q"][0] * 1e6, color=col, lw=0.8, label=name)
        w = (t >= window[0]) & (t <= window[1])
        ax[1].step(t[w] * 1e3, o["i"][0, w] * 1e3, color=col, lw=0.6, where="post", label=name)
    ax[0].set_title("position x (µm)"); ax[0].set_xlabel("time [ms]"); ax[0].legend(loc="lower right")
    ax[1].set_title(r"current $i_{X1}$ at rest (mA)"); ax[1].set_xlabel("time [ms]"); ax[1].legend(loc="upper right")
    fig.savefig(path, dpi=200); plt.close(fig)


if __name__ == "__main__":
    p = StageParams()
    style(); OUT.mkdir(exist_ok=True)
    DATA = OUT.parent / "data"; DATA.mkdir(exist_ok=True)
    for n, kind in ((1, "step"), (2, "path"), (3, "path_ff")):
        block_diagram(kind, OUT / f"diagram_exp{n}.png")
    plot_min_jerk(OUT / "min_jerk.png")

    def report(name, t, o):
        e = np.abs(o["r"][:2] - o["q"][:2])
        g = o["r"][:2, -1]
        k = np.where(e.max(axis=0) >= PRECISION)[0]          # last time an axis was 20 nm or more away
        single = name[0] in "123"                           # overshoot only means something for one move
        overshoot = f"{(o['q'][0].max() / g[0] - 1) * 100:5.1f} %" if single else "    —  "
        print(f"  {name:28s} max |r - q| {e.max()*1e6:9.4f} µm   overshoot {overshoot}   "
              f"within 20 nm after {(t[min(k[-1] + 1, len(t) - 1)] if len(k) else 0.0)*1e3:6.1f} ms   "
              f"end error {e[:, -1].max()*1e9:5.2f} nm   peak current {np.abs(o['i']).max()*1e3:8.2f} mA   "
              f"peak heat {(p.R * o['i']**2).sum(axis=0).max():8.3f} W   max |θz| {np.abs(o['q'][2]).max():.1e} rad")

    # experiments 1-3: go to (20, 10) µm
    exps = {
        "1  step + PIDF": (step_reference(GOAL), False),
        "2  path + PIDF": (path_reference(np.zeros(3), GOAL, T_MOVE), False),
        "3  path + PIDF + FF": (path_reference(np.zeros(3), GOAL, T_MOVE), True),
    }
    runs = {}
    print("experiments 1-3, move to (20, 10) µm:")
    for n, (name, (ref, ff)) in enumerate(exps.items(), start=1):
        runs[name] = t, o = simulate(ref, use_ff=ff, p=p)
        report(name, t, o)
        plot_results({name: (t, o)}, OUT / f"results_exp{n}.png")
        save_experiment(DATA / f"exp{n}.npz", t, o, p, name=name, goal=GOAL, T_move=T_MOVE, use_ff=ff)
    plot_results(runs, OUT / "results_exp1_2_3.png")

    # PID vs PIDF: experiment 1 with 1 nm of measurement noise
    noisy = {"PID (D unfiltered)": simulate(step_reference(GOAL), False, p, pid=PIDF(f_c=np.inf), noise=1e-9),
             "PIDF (D filtered at 1 kHz)": simulate(step_reference(GOAL), False, p, noise=1e-9)}
    print("experiment 1 with 1 nm measurement noise, current ripple at rest (25-50 ms):")
    for name, (t, o) in noisy.items():
        w = t >= 25e-3
        print(f"  {name:28s} rms {o['i'][0, w].std()*1e3:6.3f} mA   peak {np.abs(o['i'][0, w]).max()*1e3:6.3f} mA   "
              f"position noise rms {o['q'][0, w].std()*1e9:.3f} nm")
    plot_pid_vs_pidf(noisy, OUT / "pid_vs_pidf.png")

    # experiment 4: the spiral with controller 3
    targets = spiral_targets()
    T_spiral = (len(targets) - 1) * T_SEG + 30e-3                       # all segments, then 30 ms to settle
    name = "4  spiral, path + PIDF + FF"
    t, o = simulate(master_path(targets, T_SEG, T_spiral), use_ff=True, p=p, T_end=T_spiral)
    print(f"experiment 4, spiral, {len(targets) - 1} segments of {T_SEG*1e3:.0f} ms:")
    report(name, t, o)
    plot_spiral({name: (t, o)}, targets, OUT / "results_exp4_spiral.png")
    save_experiment(DATA / "exp4.npz", t, o, p, name=name, targets=targets, T_seg=T_SEG, use_ff=True)

    # experiment 5: raster scan of a box, starting turned by THETA0, with controller 3
    targets = scan_targets()
    T_segs = segment_times(targets, p)
    T_scan = T_segs.sum() + 30e-3
    name = "5  raster scan, path + PIDF + FF"
    t, o = simulate(master_path(targets, T_segs, T_scan), use_ff=True, p=p, T_end=T_scan, z0=[0, 0, THETA0, 0, 0, 0])
    print(f"experiment 5, raster scan, {len(targets) - 1} segments, {T_segs.min()*1e3:.1f}-{T_segs.max()*1e3:.1f} ms each, "
          f"{T_scan*1e3:.0f} ms in all:")
    report(name, t, o)
    print(f"  stays inside the travel: {in_range(o['q'], p).all()}   largest |x|, |y|: "
          f"{np.abs(o['q'][0]).max()*1e3:.3f}, {np.abs(o['q'][1]).max()*1e3:.3f} mm")
    plot_spiral({name: (t, o)}, targets, OUT / "results_exp5_raster.png")
    save_experiment(DATA / "exp5.npz", t, o, p, name=name, targets=targets, T_seg=T_segs, use_ff=True, theta0=THETA0)
    print(f"saved pictures to {OUT} and data to {DATA}")
    plt.show()