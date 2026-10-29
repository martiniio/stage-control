"""
Standalone 2-D field solver of one voice coil (write-up section 5.4). Needs numpy + scipy, ~5 s.
Prints B_left, B_right, K_f0 and K_f(dx)/K_f0. Same model as actuator_sheet.py, without the drawing.
"""
# 2-D field of one voice coil (slice along the force) and its force constant K_f(dx).
import numpy as np, scipy.sparse as sp, scipy.sparse.linalg as spla

MU0 = 4e-7 * np.pi
h = 0.2e-3                                   # grid step [m]: 0.2 mm, lines up with every part edge
BR, MU_MAG, MU_FE = 1.3, 1.05, 2000.0        # magnet remanence [T], magnet and steel relative permeability
W_M, T_M, GAP_NS, GAP = 18e-3, 5e-3, 4e-3, 5e-3   # magnet width, thickness, N-S gap, gap between magnet faces
T_FE, X_FE = 12e-3, 30e-3                    # steel plate thickness, half length
W_C, T_C, L_LEG, N = 10e-3, 3e-3, 30e-3, 233 # coil leg width, thickness, length over the magnets, turns
PITCH = W_M + GAP_NS                         # distance between the two magnet columns (= between coil legs)

# 1. grid and materials
x = np.arange(-70e-3, 70e-3 + h / 2, h); y = np.arange(-45e-3, 45e-3 + h / 2, h)
X, Y = np.meshgrid(x, y); ny, nx = X.shape
yb, yt = -GAP / 2 - T_M, GAP / 2 + T_M                       # outer faces of the bottom / top magnets
steel = (np.abs(X) <= X_FE) & (((Y >= yb - T_FE) & (Y < yb)) | ((Y > yt) & (Y <= yt + T_FE)))
steel |= (X >= -X_FE - 6e-3) & (X < -X_FE) & (Y >= yb - T_FE) & (Y <= yt + T_FE)        # back wall
nu = np.full(X.shape, 1 / MU0); nu[steel] = 1 / (MU0 * MU_FE)
Br = np.zeros(X.shape)                                       # built-in field of the magnets (+ up, - down)
for xc, sign in ((-PITCH / 2, +1), (PITCH / 2, -1)):
    mag = (np.abs(X - xc) <= W_M / 2) & (((Y >= yb) & (Y <= -GAP / 2)) | ((Y >= GAP / 2) & (Y <= yt)))
    nu[mag] = 1 / (MU0 * MU_MAG); Br[mag] = sign * BR

# 2. Ampère's law around every square: H around its edges = current inside (here: only the magnets' edges)
idx = np.arange(nx * ny).reshape(ny, nx)
rows, cols, vals, diag = [], [], [], np.zeros(nx * ny)
for p, q, w in ((idx[:, :-1], idx[:, 1:], 2 / (1 / nu[:, 1:] + 1 / nu[:, :-1])),      # nu at each face:
                (idx[:-1, :], idx[1:, :], 2 / (1 / nu[1:, :] + 1 / nu[:-1, :]))):     # harmonic mean of the two cells
    p, q, w = p.ravel(), q.ravel(), w.ravel() / h**2
    rows += [p, q]; cols += [q, p]; vals += [-w, -w]
    np.add.at(diag, p, w); np.add.at(diag, q, w)
K = sp.csr_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))), shape=(nx * ny,) * 2) + sp.diags(diag)
nB = nu * Br
src = np.zeros((ny, nx)); src[:, 1:-1] = (nB[:, 2:] - nB[:, :-2]) / (2 * h)  # magnet edges act as current sheets

# 3. A = 0 on the outer edge; solve the rest in one go
inner = np.ones((ny, nx), bool); inner[[0, -1], :] = False; inner[:, [0, -1]] = False; inner = inner.ravel()
A = np.zeros(nx * ny); A[inner] = spla.spsolve(K[inner][:, inner].tocsc(), src.ravel()[inner]); A = A.reshape(ny, nx)
By = -np.gradient(A, h, axis=1)                              # vertical field B_y = -dA/dx

# 4. field under the legs, and K_f as the coil slides by dx
def leg(xc):                                                 # mean B_y over a leg centred at xc
    m = (np.abs(X - xc) <= W_C / 2 + 1e-9) & (np.abs(Y) <= T_C / 2 + 1e-9)
    return By[m].mean()
def Kf(dx):
    return N * L_LEG * (leg(-PITCH / 2 + dx) - leg(PITCH / 2 + dx))

print(f"B_left = {leg(-PITCH/2):.3f} T, B_right = {leg(PITCH/2):.3f} T, K_f0 = {Kf(0):.2f} N/A")
for dx in (1e-3, 2e-3, 2.6e-3, 4e-3):
    print(f"dx = {dx*1e3:.1f} mm: K_f/K_f0 = {Kf(dx)/Kf(0):.4f}")
