"""
1D vertical dust settling references, adapted from ``dustysettle_limiters.py``
=============================================================================

The two reference solutions of ``dustysettle_limiters.py`` (which cannot be
imported, as it runs the simulations when loaded), parameterised so that they
can be used at any radius:

- ``GasAtRestSettle``: each dust species settles on a gas at rest (the
  "reference" of ``dustysettle_limiters.py``, same Crank-Nicolson scheme).
- ``MultifluidSettle``: full gas + dust multifluid equations (the
  "multifluid (non-TVA)" reference, same staggered scheme).

The numerics are the same as in ``dustysettle_limiters.py``; the matrix
assembly loops are vectorised with numpy instead of jitted with numba.

Dimensionless units: z / R0, v / vK, t * OmegaK, rho / rho_mid.
"""

import numpy as np
from scipy.linalg import solve_banded


def smooth_mask(z, H, enabled):
    """Dust injection mask of ``dustysettle_limiters.py`` (1 if disabled)."""
    if not enabled:
        return np.ones_like(z)
    return 1 / (1 + np.exp((np.abs(z) - 1.75 * H) / (H / 16)))


# ------------------------------------------
# Dust settling on a gas at rest
# ------------------------------------------


def _S_rho_system(rho, v, vp, hz, tau, Nz):
    """Banded Crank-Nicolson system for the continuity equation."""
    npts = Nz + 2
    j = np.arange(1, Nz + 1)

    ab_A = np.zeros((3, npts))
    bd_main = np.zeros(npts)
    bd_lower = np.zeros(npts)
    bd_upper = np.zeros(npts)

    # Interior points
    C1 = -tau / (8 * hz) * (v[j + 1] + vp[j + 1])
    C2 = -tau / (8 * hz) * (v[j - 1] + vp[j - 1])
    ab_A[2, j - 1] = C2
    ab_A[1, j] = 1.0
    ab_A[0, j + 1] = -C1
    bd_lower[j] = -C2
    bd_main[j] = 1.0
    bd_upper[j] = C1

    # Left boundary
    C = -tau / (8 * hz) * (vp[0] + v[0] - vp[1] - v[1])
    ab_A[1, 0] = 0.5 + C
    ab_A[0, 1] = 0.5 + C
    bd_main[0] = 0.5 - C
    bd_upper[0] = 0.5 - C

    # Right boundary
    C = -tau / (8 * hz) * (vp[-1] + v[-1] - vp[-2] - v[-2])
    ab_A[2, npts - 2] = 0.5 - C
    ab_A[1, npts - 1] = 0.5 - C
    bd_lower[npts - 1] = 0.5 + C
    bd_main[npts - 1] = 0.5 + C

    r = bd_main * rho
    r[1:] += bd_lower[1:] * rho[:-1]
    r[:-1] += bd_upper[:-1] * rho[1:]
    return ab_A, r


def S_rho(rho, v, vp, hz, tau, Nz):
    """Step rho forward using Crank-Nicolson on the continuity equation."""
    ab_A, r = _S_rho_system(rho, v, vp, hz, tau, Nz)
    ret = solve_banded((1, 1), ab_A, r)
    ret = np.maximum(ret, 0)

    # If there are zero values, propagate the zero value to the edges
    n = len(ret)
    center = n // 2

    right = ret[center:]
    propagate_right = np.cumsum(right == 0) > 0
    ret[center:] = right * (~propagate_right)

    left = ret[: center + 1][::-1]
    propagate_left = np.cumsum(left == 0) > 0
    ret[: center + 1] = (left * (~propagate_left))[::-1]

    return ret


def _S_v_system(v, vbar, vbarp, hz, zbar, tau, Nz, Stj):
    """Banded Crank-Nicolson system for the velocity equation."""
    npts = Nz + 2
    j = np.arange(1, Nz + 1)

    ab_A = np.zeros((3, npts))
    bd_main = np.zeros(npts)
    bd_lower = np.zeros(npts)
    bd_upper = np.zeros(npts)
    E0 = np.zeros_like(v)

    E0[j] = -zbar[j] / (1 + zbar[j] ** 2) ** 1.5
    E1 = (vbar[j] + vbarp[j]) / (8 * hz)
    E2 = 1.0 / (2.0 * Stj[j])

    ab_A[2, j - 1] = -E1
    ab_A[1, j] = 1.0 / tau + E2
    ab_A[0, j + 1] = E1
    bd_lower[j] = E1
    bd_main[j] = 1.0 / tau - E2
    bd_upper[j] = -E1

    # Boundary conditions
    ab_A[1, 0] = 1
    ab_A[0, 1] = 1
    ab_A[2, npts - 2] = 1
    ab_A[1, npts - 1] = 1

    r = bd_main * v
    r[1:] += bd_lower[1:] * v[:-1]
    r[:-1] += bd_upper[:-1] * v[1:]
    r += E0
    return ab_A, r


def S_v(v, vbar, vbarp, hz, zbar, tau, Nz, Stj):
    """Step the velocity forward using Crank-Nicolson."""
    ab_A, r = _S_v_system(v, vbar, vbarp, hz, zbar, tau, Nz, Stj)
    return solve_banded((1, 1), ab_A, r)


class GasAtRestSettle:
    """
    One dust species settling on a gas at rest with a Gaussian vertical profile.

    H, R0      : scale height and radius (same length unit, z returned in it)
    rho_mid    : midplane gas density [kg m^-3]
    dust_to_gas: initial dust-to-gas ratio (times the mask if enabled)
    rhoeff     : rho_grain sqrt(pi gamma / 8) [kg m^-3]
    s_grain    : grain size [m]
    cs, OmegaK : sound speed [m s^-1] and Keplerian frequency [s^-1]
    """

    def __init__(
        self, Nz, H, R0, rho_mid, dust_to_gas, rhoeff, s_grain, cs, OmegaK, tau, zrange, mask
    ):
        self.H = H
        self.R0 = R0
        self.rho_mid = rho_mid
        self.Nz = Nz
        self.tau = tau

        zout = zrange * H / R0
        self.hz = 2 * zout / Nz
        # exactly Nz + 2 points
        self.zbar = np.linspace(-zout - 0.5 * self.hz, zout + 0.5 * self.hz, Nz + 2)

        z = self.z_arr()
        self.rhog = rho_mid * np.exp(-(z**2) / (2 * H * H))
        self.rho = smooth_mask(z, H, mask) * dust_to_gas * self.rhog / rho_mid
        self.v = np.zeros_like(self.rho)

        self.ts = rhoeff * s_grain / (cs * self.rhog)
        self.Stj = self.ts * OmegaK

        self.tbar = 0.0

    def z_arr(self):
        return self.zbar * self.R0

    def rho_dust(self):
        """Dust density [kg m^-3]."""
        return self.rho * self.rho_mid

    def advance(self, tau):
        rhop = S_rho(self.rho, self.v, self.v, self.hz, tau, self.Nz)
        vp = S_v(self.v, self.v, self.v, self.hz, self.zbar, tau, self.Nz, self.Stj)
        rhop = S_rho(self.rho, self.v, vp, self.hz, tau, self.Nz)
        self.v = S_v(self.v, self.v, vp, self.hz, self.zbar, tau, self.Nz, self.Stj)
        self.rho = rhop
        self.tbar += tau

    def advance_until(self, tfinal):
        while tfinal - self.tbar > 1e-12:
            self.advance(min(self.tau, tfinal - self.tbar))


# ------------------------------------------
# Multifluid (non-TVA) gas + dust
# ------------------------------------------


def vertical_gravity(zbar):
    """Vertical gravity in units of vK * OmegaK (same as in ``S_v``)."""
    return -zbar / (1 + zbar**2) ** 1.5


def van_leer_slope(q):
    """van Leer limited slope (per cell) along the last axis, zero on the first and last cell."""
    slope = np.zeros_like(q)
    prod = (q[..., 1:-1] - q[..., :-2]) * (q[..., 2:] - q[..., 1:-1])
    sum_ = q[..., 2:] - q[..., :-2]
    slope[..., 1:-1] = np.where(prod > 0, 2 * prod / np.where(prod > 0, sum_, 1.0), 0.0)
    return slope


def continuity_rate(rho, v, hz, second_order):
    """-d(rho v)/dz for cell-centred rho, v at faces (zero at the walls), upwind face rho."""
    slope = van_leer_slope(rho) if second_order else 0 * rho
    v_in = v[..., 1:-1]
    rho_l = rho[..., :-1] + 0.5 * slope[..., :-1]
    rho_r = rho[..., 1:] - 0.5 * slope[..., 1:]
    flux = np.zeros_like(v)
    flux[..., 1:-1] = v_in * np.where(v_in > 0, rho_l, rho_r)
    return -(flux[..., 1:] - flux[..., :-1]) / hz


def advection_rate(v, hz, second_order):
    """v dv/dz = d(v^2/2)/dz on the faces (zero on the walls), Godunov Burgers flux."""
    slope = van_leer_slope(v) if second_order else 0 * v
    u_l = v[..., :-1] + 0.5 * slope[..., :-1]
    u_r = v[..., 1:] - 0.5 * slope[..., 1:]
    f_shock = 0.5 * np.maximum(u_l**2, u_r**2)
    f_rarefaction = np.where((u_l < 0) & (u_r > 0), 0.0, 0.5 * np.minimum(u_l**2, u_r**2))
    f = np.where(u_l > u_r, f_shock, f_rarefaction)
    adv = np.zeros_like(v)
    adv[..., 1:-1] = (f[..., 1:] - f[..., :-1]) / hz
    return adv


def to_faces(rho):
    """Cell-centred to face densities (walls take the adjacent cell)."""
    rho_f = np.empty(rho.shape[:-1] + (rho.shape[-1] + 1,))
    rho_f[..., 1:-1] = 0.5 * (rho[..., :-1] + rho[..., 1:])
    rho_f[..., 0] = rho[..., 0]
    rho_f[..., -1] = rho[..., -1]
    return rho_f


def drag_solve(r_g, r_d, rho_g_f, rho_d_f, a):
    """
    Solve v_d = r_d - a (v_d - v_g) for each species, with the total momentum
    rho_g v_g + sum rho_d v_d = rho_g r_g + sum rho_d r_d.
    """
    w = 1 / (1 + a)
    mom = rho_g_f * r_g + np.sum(rho_d_f * r_d, axis=0)
    v_g = (mom - np.sum(rho_d_f * r_d * w, axis=0)) / (rho_g_f + np.sum(rho_d_f * a * w, axis=0))
    v_d = (r_d + a * v_g[None, :]) * w
    return v_g, v_d


class MultifluidSettle:
    """
    Gas + all dust species (continuity + momentum each), isothermal gas, vertical
    gravity, Epstein drag with back-reaction, optional velocity dissipation on the
    barycentric velocity.

    cs_bar      : isothermal sound speed / vK (= H / R0)
    eta_bar     : velocity dissipation rate / OmegaK (0 for none)
    inv_ts_coef : 1/t_j = inv_ts_coef[j] * rho_g (Epstein drag, gas density)
    dust_to_gas : initial dust-to-gas ratio of each species (times the mask if enabled)
    h_bar       : H / R0
    """

    def __init__(
        self, Nz, cs_bar, eta_bar, inv_ts_coef, dust_to_gas, h_bar, zrange, mask, order=2, cfl=0.4
    ):
        if Nz % 2 != 0:
            raise ValueError("Nz must be even (the midplane is a face)")
        if order not in (1, 2):
            raise ValueError("order must be 1 or 2")

        self.cs2 = cs_bar**2
        self.eta = eta_bar
        self.inv_ts_coef = np.asarray(inv_ts_coef)
        self.order = order
        self.cfl = cfl

        zout = zrange * h_bar
        self.hz = 2 * zout / Nz
        self.zf = np.linspace(-zout, zout, Nz + 1)
        self.zc = 0.5 * (self.zf[:-1] + self.zf[1:])
        self.g_f = vertical_gravity(self.zf)
        self.g_f[[0, -1]] = 0

        # discrete hydrostatic equilibrium: cs^2 (ln rho_{i+1} - ln rho_i) / hz = g_{i+1/2}
        ln_rho = np.zeros(Nz)
        ln_rho[1:] = np.cumsum(self.hz * self.g_f[1:-1] / self.cs2)
        ln_rho += -(self.zc[Nz // 2] ** 2) / (2 * self.cs2) - ln_rho[Nz // 2]
        self.rho_g = np.exp(ln_rho)
        self.v_g = np.zeros(Nz + 1)

        mask_c = smooth_mask(self.zc, h_bar, mask)
        self.rho_d = np.array([dtg * mask_c * self.rho_g for dtg in dust_to_gas])
        self.v_d = np.zeros((len(dust_to_gas), Nz + 1))

        self.tbar = 0.0

    def get_dt(self):
        vmax = max(self.cs2**0.5 + np.abs(self.v_g).max(), np.abs(self.v_d).max())
        return self.cfl * self.hz / vmax

    def density_rates(self, rho_g, v_g, rho_d, v_d):
        second_order = self.order == 2
        drho_g = continuity_rate(rho_g, v_g, self.hz, second_order)
        drho_d = continuity_rate(rho_d, v_d, self.hz, second_order)
        return drho_g, drho_d

    def accelerations(self, rho_g, v_g, rho_d, v_d):
        """Accelerations other than the drag."""
        hz = self.hz
        second_order = self.order == 2

        rho_g_f = to_faces(rho_g)
        rho_d_f = to_faces(rho_d)
        mom = rho_g_f * v_g + np.sum(rho_d_f * v_d, axis=0)
        v_bary = mom / (rho_g_f + np.sum(rho_d_f, axis=0))

        grad_P_over_rho = np.zeros_like(v_g)
        grad_P_over_rho[1:-1] = self.cs2 * np.diff(np.log(rho_g)) / hz

        acc_g = -advection_rate(v_g, hz, second_order) - grad_P_over_rho + self.g_f
        acc_d = -advection_rate(v_d, hz, second_order) + self.g_f
        return acc_g - self.eta * v_bary, acc_d - self.eta * v_bary

    def euler_step(self, dt):
        """Semi-implicit Euler step with backward-Euler drag, returns the new state."""
        drho_g, drho_d = self.density_rates(self.rho_g, self.v_g, self.rho_d, self.v_d)
        rho_g = self.rho_g + dt * drho_g
        rho_d = self.rho_d + dt * drho_d
        acc_g, acc_d = self.accelerations(rho_g, self.v_g, rho_d, self.v_d)
        rho_g_f = to_faces(rho_g)
        a = dt * self.inv_ts_coef[:, None] * rho_g_f[None, :]
        v_g, v_d = drag_solve(
            self.v_g + dt * acc_g, self.v_d + dt * acc_d, rho_g_f, to_faces(rho_d), a
        )
        return rho_g, v_g, rho_d, v_d

    def midpoint_step(self, dt):
        """Predictor (half Euler step) - corrector (mid-step rates, Crank-Nicolson drag)."""
        half = self.euler_step(0.5 * dt)
        drho_g, drho_d = self.density_rates(*half)
        acc_g, acc_d = self.accelerations(*half)
        rho_g_h, rho_d_h = half[0], half[2]

        rho_g_f = to_faces(rho_g_h)
        rho_d_f = to_faces(rho_d_h)
        a = 0.5 * dt * self.inv_ts_coef[:, None] * rho_g_f[None, :]

        # explicit half of the drag at t^n
        drag_d = -a * (self.v_d - self.v_g[None, :])
        drag_g = -np.sum(rho_d_f * drag_d, axis=0) / rho_g_f

        v_g, v_d = drag_solve(
            self.v_g + dt * acc_g + drag_g,
            self.v_d + dt * acc_d + drag_d,
            rho_g_f,
            rho_d_f,
            a,
        )
        return self.rho_g + dt * drho_g, v_g, self.rho_d + dt * drho_d, v_d

    def advance(self, dt):
        if self.order == 1:
            rho_g, v_g, rho_d, v_d = self.euler_step(dt)
        else:
            rho_g, v_g, rho_d, v_d = self.midpoint_step(dt)

        v_g[[0, -1]] = 0
        v_d[:, [0, -1]] = 0

        self.rho_g, self.v_g, self.rho_d, self.v_d = rho_g, v_g, rho_d, v_d
        self.tbar += dt

    def advance_until(self, tfinal):
        while tfinal - self.tbar > 1e-12:
            self.advance(min(self.get_dt(), tfinal - self.tbar))
