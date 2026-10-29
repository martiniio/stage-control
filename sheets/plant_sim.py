"""
Plant of the planar XY-θz stage: simulation and sanity tests. Standalone (numpy + scipy).

Model (write-up 6.9), tests as in write-up 6.10:  M q'' + D q' = K_f0 B(θz) i,   output y = q = (x, y, θz) of the centre G.
    q = (x, y, θz)   i = (i_X1, i_X2, i_Y1, i_Y2)
Each test is a scenario whose answer is known in advance; the script checks the simulation against it.
"""
import numpy as np
from scipy.integrate import solve_ivp

# ── parameters ───────────────────────────────────────────────────────────────────────
m     = 1.0        # kg       [A]   moving mass
J     = 0.0052     # kg m^2   [DFA] yaw inertia: plate + coils + heads (write-up 6.7)
b     = 0.0102     # N s/m    [DFA] air-film drag, translation (write-up 6.6)
b_th  = 1.39e-5    # N m s    [DFA] air-film drag, rotation (write-up 6.6)
Kf0   = 11.4       # N/A      [DFA] force constant, coil centred (write-up 5.5)
a     = 0.055      # m        [A]   lever arm of the coil lines

M = np.diag([m, m, J]); D = np.diag([b, b, b_th]); Minv = np.linalg.inv(M)
B0 = np.array([[1, 1, 0, 0], [0, 0, 1, 1], [-a, a, a, -a]], float)


def B(th):
    c, s = np.cos(th), np.sin(th)
    return np.array([[c, c, -s, -s], [s, s, c, c], [-a, a, a, -a]])


def rhs(t, z, current):
    q, qd = z[:3], z[3:]
    i = current(t, z)
    return np.concatenate([qd, Minv @ (Kf0 * B(q[2]) @ i - D @ qd)])


def sim(current, T, z0=None, n=2001):
    z0 = np.zeros(6) if z0 is None else np.asarray(z0, float)
    t = np.linspace(0, T, n)
    r = solve_ivp(rhs, (0, T), z0, t_eval=t, args=(current,), rtol=1e-10, atol=1e-13, max_step=T / 200)
    return r.t, r.y                                   # y rows: x, y, θz, ẋ, ẏ, θ̇z


const = lambda i: (lambda t, z: np.asarray(i, float))
results = []


def check(name, ok, detail):
    results.append((name, ok, detail)); print(f"[{'PASS' if ok else 'FAIL'}] {name}: {detail}")


# 1. nothing in, nothing out
t, z = sim(const([0, 0, 0, 0]), 1.0)
check("at rest, no current", np.abs(z).max() == 0.0, f"max |state| = {np.abs(z).max():.1e}")

# 2. squeeze pattern: forces cancel, moments cancel -> no motion at all
t, z = sim(const([-1, 1, -1, 1]), 0.1)
check("squeeze currents (-1,1,-1,1) A", np.abs(z).max() < 1e-12, f"max |state| = {np.abs(z).max():.1e}")

# 3. X pair pushes together: pure x, a = 2 K_f0 / m (drag negligible over 10 ms)
T = 0.01
t, z = sim(const([1, 1, 0, 0]), T)
x_exp = 0.5 * 2 * Kf0 / m * T**2
check("X1 = X2 = 1 A for 10 ms", abs(z[0, -1] / x_exp - 1) < 1e-4 and np.abs(z[1:3]).max() < 1e-15,
      f"x = {z[0,-1]*1e3:.4f} mm (expected {x_exp*1e3:.4f}), max |y|,|θz| = {np.abs(z[1:3]).max():.1e}")

# 4. Y pair: pure y, same size
t, z = sim(const([0, 0, 1, 1]), T)
check("Y1 = Y2 = 1 A for 10 ms", abs(z[1, -1] / x_exp - 1) < 1e-4 and np.abs(z[[0, 2]]).max() < 1e-15,
      f"y = {z[1,-1]*1e3:.4f} mm, max |x|,|θz| = {np.abs(z[[0, 2]]).max():.1e}")

# 5. sign symmetry: reversing the currents reverses the motion exactly
t, zn = sim(const([-1, -1, 0, 0]), T)
check("X pair reversed", abs(zn[0, -1] + x_exp) / x_exp < 1e-4, f"x = {zn[0,-1]*1e3:.4f} mm")

# 6. pure turning pattern (-1,1,1,-1): θz only, α = 4 a K_f0 / J; x, y stay zero even while it turns
T = 0.005
t, z = sim(const([-1, 1, 1, -1]), T)
th_exp = 0.5 * 4 * a * Kf0 / J * T**2
check("turning currents (-1,1,1,-1) A for 5 ms", abs(z[2, -1] / th_exp - 1) < 1e-4 and np.abs(z[:2]).max() < 1e-15,
      f"θz = {z[2,-1]*1e3:.3f} mrad (expected {th_exp*1e3:.3f}), max |x|,|y| = {np.abs(z[:2]).max():.1e}")

# 7. turned plate: the X pair pushes along the turned direction, ẍ:ÿ = cos θ0 : sin θ0
th0 = 1.5e-3
t, z = sim(const([1, 1, 0, 0]), 0.002, z0=[0, 0, th0, 0, 0, 0])
check("X pair with plate turned 1.5 mrad", abs(z[1, -1] / z[0, -1] - np.tan(th0)) < 1e-9 and abs(z[2, -1] - th0) < 1e-15,
      f"y/x = {z[1,-1]/z[0,-1]:.6e} (tan θ0 = {np.tan(th0):.6e}), θz unchanged")

# 8. free coasting: drag only, velocity decays with time constant m/b (and J/b_θ)
t, z = sim(const([0, 0, 0, 0]), 10.0, z0=[0, 0, 0, 0.01, 0, 1e-3])
v_exp = 0.01 * np.exp(-b / m * 10); w_exp = 1e-3 * np.exp(-b_th / J * 10)
check("coasting 10 s at 10 mm/s and 1 mrad/s", abs(z[3, -1] / v_exp - 1) < 1e-6 and abs(z[5, -1] / w_exp - 1) < 1e-6,
      f"ẋ = {z[3,-1]*1e3:.5f} mm/s (expected {v_exp*1e3:.5f}), θ̇z = {z[5,-1]*1e3:.5f} mrad/s (expected {w_exp*1e3:.5f})")

# 9. energy: work done by the coils = kinetic energy + energy lost to drag
cur = lambda t, z: np.array([np.sin(40 * t), 0.5, np.cos(25 * t), -0.3]) * 0.8
t, z = sim(cur, 0.05, n=20001)
P_in = np.array([z[3:, k] @ (Kf0 * B(z[2, k]) @ cur(t[k], z[:, k])) for k in range(len(t))])
P_dr = np.array([z[3:, k] @ D @ z[3:, k] for k in range(len(t))])
E_k = 0.5 * (m * z[3, -1]**2 + m * z[4, -1]**2 + J * z[5, -1]**2)
W = np.trapezoid(P_in - P_dr, t)
check("energy balance, mixed currents for 50 ms", abs(W / E_k - 1) < 1e-4, f"work − drag loss = {W:.6e} J, kinetic = {E_k:.6e} J")

# 10. validity limits — what the model does NOT know
t, z = sim(const([1, 1, 0, 0]), 0.03)
k = np.argmax(np.abs(z[0]) > 2.5e-3)
check("[limit] constant 1 A on the X pair", True,
      f"leaves the ±2.5 mm travel after {t[k]*1e3:.1f} ms and keeps going (x = {z[0,-1]*1e3:.1f} mm at 30 ms): "
      "the model has no end stops and no K_f fall-off, so it is only valid inside the travel")

print(f"\n{sum(r[1] for r in results)}/{len(results)} checks passed")
