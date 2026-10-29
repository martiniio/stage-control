"""
stage_model.py — the planar XY-θz stage: everything the write-up derives, as one python-control plant.

    plant      M q'' + D q' = K_f0 B(θz) i                                  (write-up 6.9)
    input      i = (iX1, iX2, iY1, iY2)  [A]   coil currents (ideal current amplifiers, 6.8)
    state      (x, y, thz, vx, vy, wz)
    output     q = (x, y, thz)            [m, m, rad]   position and angle of the centre G

Where each number comes from (tags as in the write-up: A assumed, P published, DE derived, DFA derived-from-assumed):
    geometry, mass                      4.2        [A]
    force constant K_f0 = N l ΔB0       5.2–5.5    [DFA]  (ΔB0 from the 2-D field solve, 5.4)
    coil resistance R                   5.5        [DFA]
    inertia J from the mass split       6.7        [DFA]
    air-film drag b, b_th               6.6        [DFA]
    coil forces turn with the plate     6.5        [DE]
"""
from dataclasses import dataclass
import numpy as np
import control as ct


@dataclass(frozen=True)
class StageParams:
    # geometry and mass (4.2)                                         [A]
    L: float = 0.150          # m      plate side
    a: float = 0.055          # m      coil lines y = ±a, x = ±a; encoder heads at (±a, ±a)
    c: float = 0.105          # m      coil centre from G along its line (L/2 + 30 mm)
    m: float = 1.0            # kg     moving mass
    travel: float = 2.5e-3    # m      model valid for |x|, |y| below this
    rot: float = 1.5e-3       # rad    ... and |thz| below this
    # coil and magnets (5.1, 5.5)
    N: int = 233              # turns                                [DFA] 0.3 mm wire, 55 % fill of a 10 x 3 mm leg
    l: float = 0.030          # m      leg length over the magnets   [A]
    dB0: float = 1.636        # T      B_left - B_right, coil centred [DFA] 2-D field solve
    wire_d: float = 0.3e-3    # m      wire diameter                 [A]
    turn_len: float = 0.104   # m      mean turn length 2 (l + 22 mm) [DE]
    rho_cu: float = 1.72e-8   # ohm m  copper resistivity            [P]
    # mass split for J (6.7)
    m_coil: float = 0.030     # kg     one coil with its arm         [A]
    m_head: float = 0.020     # kg     one encoder head              [A]
    # air bearings (6.6)
    mu_air: float = 1.8e-5    # Pa s   air viscosity                 [P]
    h_film: float = 5e-6      # m      film thickness                [A]
    R_pad: float = 0.015      # m      pad radius (Ø30)              [A]
    r_pad: float = 0.0354     # m      pad centre from G             [A]

    # ── derived values ───────────────────────────────────────────────────────────────
    @property
    def Kf0(self):            # N/A   force constant, coil centred (5.5)
        return self.N * self.l * self.dB0

    @property
    def R(self):              # ohm   coil resistance (5.5)
        return self.rho_cu * self.N * self.turn_len / (np.pi * self.wire_d**2 / 4)

    @property
    def J(self):              # kg m^2  plate m_p L^2/6 + coils + heads as point masses (6.7)
        m_plate = self.m - 4 * self.m_coil - 4 * self.m_head
        return (m_plate * self.L**2 / 6 + 4 * self.m_coil * (self.c**2 + self.a**2)
                + 4 * self.m_head * 2 * self.a**2)

    @property
    def b(self):              # N s/m  translational drag of 4 pads (6.6)
        return self.mu_air * 4 * np.pi * self.R_pad**2 / self.h_film

    @property
    def b_th(self):           # N m s  rotational drag of 4 pads (6.6)
        return 4 * self.mu_air / self.h_film * (np.pi * self.R_pad**2 * self.r_pad**2 + np.pi * self.R_pad**4 / 2)

    @property
    def B0(self):             # input matrix with the plate at rest (6.4)
        a = self.a
        return np.array([[1, 1, 0, 0], [0, 0, 1, 1], [-a, a, a, -a]], float)

    @property
    def heads(self):          # encoder head points E1..E4 on the plate (4.1)
        a = self.a
        return np.array([[a, a], [-a, a], [-a, -a], [a, -a]], float)


def B(th, p: StageParams):
    """Input matrix with the plate turned by thz: diag(R(thz), 1) B0 (6.5)."""
    cs, sn = np.cos(th), np.sin(th)
    return np.array([[cs, -sn, 0], [sn, cs, 0], [0, 0, 1]]) @ p.B0


def stage(p: StageParams = StageParams(), name="stage"):
    """The plant as a nonlinear python-control I/O system."""
    Minv = np.diag([1 / p.m, 1 / p.m, 1 / p.J])
    D = np.diag([p.b, p.b, p.b_th])

    def updfcn(t, z, i, params):
        q, qd = z[:3], z[3:]
        F = p.Kf0 * B(q[2], p) @ np.asarray(i, float)
        # Not modelled yet:
        #   K_f(Δx) = K_f0 (1 - 4.5e3 Δx^2), Δx_j = coil j's position along its magnet axis — at most ~3 % (5.6)
        #   end stops: F_stop = -k_s (x - travel) - c_s vx for x > travel (mirror for x < -travel; same for y, thz)
        return np.concatenate([qd, Minv @ (F - D @ qd)])

    def outfcn(t, z, i, params):
        return z[:3]

    return ct.nlsys(updfcn, outfcn, name=name,
                    states=["x", "y", "thz", "vx", "vy", "wz"],
                    inputs=["iX1", "iX2", "iY1", "iY2"], outputs=["x", "y", "thz"])


def encoder_readings(q, p: StageParams = StageParams()):
    """What the four heads read: (x, y) + (R(thz) - I) s_k for each head k (6.9). Returns 8 values."""
    cs, sn = np.cos(q[2]), np.sin(q[2])
    Rm = np.array([[cs, -sn], [sn, cs]])
    return np.concatenate([np.asarray(q[:2]) + (Rm - np.eye(2)) @ s for s in p.heads])


def in_range(q, p: StageParams = StageParams()):
    """True where the plate is inside the range the model is valid for."""
    q = np.atleast_2d(q)
    return (np.abs(q[0]) <= p.travel) & (np.abs(q[1]) <= p.travel) & (np.abs(q[2]) <= p.rot)


if __name__ == "__main__":
    p = StageParams()
    print(f"K_f0 = {p.Kf0:.2f} N/A   R = {p.R:.2f} ohm   J = {p.J:.5f} kg m^2   "
          f"b = {p.b:.4f} N s/m   b_th = {p.b_th:.3e} N m s")
    print(stage(p))