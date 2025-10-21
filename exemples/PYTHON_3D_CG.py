import numpy as np

# ============ GRID 3D ============
nx, ny, nz = 65, 65, 65
xmin, xmax = 0.0, 1.0
ymin, ymax = 0.0, 1.0
zmin, zmax = 0.0, 1.0

dx = (xmax - xmin) / (nx - 1)
dy = (ymax - ymin) / (ny - 1)
dz = (zmax - zmin) / (nz - 1)

x = np.linspace(xmin, xmax, nx)
y = np.linspace(ymin, ymax, ny)
z = np.linspace(zmin, zmax, nz)
X, Y, Z = np.meshgrid(x, y, z, indexing="ij")

# ============ RHS ============
# def rhs_3d(X, Y, Z, Lx=1.0, Ly=1.0, Lz=1.0):
#     return 4 * np.pi * np.sin(2 * np.pi * X / Lx) * np.sin(2 * np.pi * Y / Ly) * np.sin(2 * np.pi * Z / Lz)


def rhs_3d(X, Y, Z, x0=0.5, y0=0.5, z0=0.5, r0=0.25, rho0=1.0):
    R = np.sqrt((X - x0) ** 2 + (Y - y0) ** 2 + (Z - z0) ** 2)
    rho = np.zeros_like(R)
    inside = R <= r0
    r_scaled = R[inside] / r0
    rho[inside] = rho0 * (1 - r_scaled**2) ** 2
    rho_bar = (32.0 * np.pi * r0**3) / 105.0
    return 4 * np.pi * (rho - rho_bar)

    return np.cos(2 * np.pi * X) * np.cos(2 * np.pi * Y) * np.cos(2 * np.pi * Z)


b = rhs_3d(X, Y, Z)


# ============ LAPLACIAN ============
def A(v, dx, dy, dz):
    Av = -(
        (v[2:, 1:-1, 1:-1] - 2 * v[1:-1, 1:-1, 1:-1] + v[:-2, 1:-1, 1:-1]) / dx**2
        + (v[1:-1, 2:, 1:-1] - 2 * v[1:-1, 1:-1, 1:-1] + v[1:-1, :-2, 1:-1]) / dy**2
        + (v[1:-1, 1:-1, 2:] - 2 * v[1:-1, 1:-1, 1:-1] + v[1:-1, 1:-1, :-2]) / dz**2
    )
    return Av


# ============ L2 DIFF ============
def l2_diff(f1, f2):
    return np.sqrt(np.sum((f1 - f2) ** 2)) / f1.size


# ============ L1 DIFF ============
def l1_diff(f1, f2):
    return np.sum(np.abs(f1 - f2)) / f1.size


# ============ LINF DIFF ============
def linf_diff(f1, f2):
    return np.max(np.abs(f1 - f2))


# ============ INITIAL CONDITIONS ============
p = np.zeros((nx, ny, nz))  # solution
r = np.zeros_like(p)
Ad = np.zeros_like(p)

# initial residual
r[1:-1, 1:-1, 1:-1] = -b[1:-1, 1:-1, 1:-1] - A(p, dx, dy, dz)
d = r.copy()

# ============ CG ITERATION ============
tolerance = 1e-10
max_it = 1000

it = 0
diff_l1 = 1.0
diff_l2 = 1.0
diff_linf = 1.0
diff = 1.0
diff = np.min(np.array([diff, diff_l1, diff_l2, diff_linf]))

while diff > tolerance:
    if it > max_it:
        print(f"\nCG did not converge in {max_it} iterations. Final diff = {diff:.2e}")
        break

    Ad[1:-1, 1:-1, 1:-1] = A(d, dx, dy, dz)
    alpha = np.sum(r * r) / np.sum(d * Ad)

    pnew = p + alpha * d
    beta_denom = np.sum(r * r)

    r = r - alpha * Ad
    beta = np.sum(r * r) / beta_denom

    d = r + beta * d

    # diff = l2_diff(pnew, p)
    # print(f"it = {it}, diff = {diff:.2e}")

    diff_l2 = l2_diff(pnew, p)
    diff_l1 = l1_diff(pnew, p)
    diff_linf = linf_diff(pnew, p)
    diff = np.min(np.array([diff, diff_l1, diff_l2, diff_linf]))

    # print(f"it = {it:3d}, L2 = {diff_l2:.2e}")
    print(f"it = {it:3d}, L2 = {diff_l2:.2e}, L1 = {diff_l1:.2e}, Linf = {diff_linf:.2e}")
    # print(f"it = {it:3d}, L2 = {diff_l2:.2e}, L1 = {diff_l1:.2e}, Linf = {diff_linf:.2e}")

    if diff_l2 < tolerance and diff_l1 < tolerance and diff_linf < tolerance:
        print(
            f"\nConverged in {it} iterations with final diff L2 = {diff_l2:.2e}, L1 = {diff_l1:.2e}, Linf = {diff_linf:.2e} ."
        )

    # if ( diff_l1 < tolerance ):
    #     print(f"\nConverged in {it} iterations with final diff   L1 = {diff_l1:.2e}.")

    # if ( diff_linf < tolerance):
    #     print(f"\nConverged in {it} iterations with final diff Linf = {diff_linf:.2e}.")

    p[:] = pnew
    it += 1
else:
    print(
        f"\nConverged in {it} iterations with final diff : L2 = {diff_l2:.2e} , L1 = {diff_l1:.2e}, Linf = {diff_linf:.2e}"
    )
