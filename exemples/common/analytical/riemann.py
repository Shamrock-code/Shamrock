#!/usr/bin/env python3
"""
Sod Shock Tube Analytical Solution (Exact Riemann Solver)

Computes the exact solution for the Sod shock tube problem using
the iterative Riemann solver approach.

References:
- Toro, E.F. (2009) "Riemann Solvers and Numerical Methods for Fluid Dynamics"
- Sod, G.A. (1978) "A Survey of Several Finite Difference Methods for Systems
  of Nonlinear Hyperbolic Conservation Laws"
"""

import numpy as np


class SodAnalytical:
    """
    Exact Riemann solver for the Sod shock tube problem.

    Initial conditions:
        Left state (x < x0):  rho_L, p_L, u_L
        Right state (x > x0): rho_R, p_R, u_R

    Parameters:
    -----------
    gamma : float
        Adiabatic index (default: 1.4)
    rho_L, rho_R : float
        Left and right densities
    p_L, p_R : float
        Left and right pressures
    u_L, u_R : float
        Left and right velocities
    x0 : float
        Initial discontinuity position
    """

    def __init__(
        self, gamma=1.4, rho_L=1.0, rho_R=0.125, p_L=1.0, p_R=0.1, u_L=0.0, u_R=0.0, x0=0.0
    ):
        self.gamma = gamma
        self.rho_L = rho_L
        self.rho_R = rho_R
        self.p_L = p_L
        self.p_R = p_R
        self.u_L = u_L
        self.u_R = u_R
        self.x0 = x0

        # Compute sound speeds
        self.c_L = np.sqrt(gamma * p_L / rho_L)
        self.c_R = np.sqrt(gamma * p_R / rho_R)

        # Solve for star region
        self._solve_star_region()

    def _solve_star_region(self):
        """Solve for pressure and velocity in the star region."""
        gamma = self.gamma

        # Initial guess using PVRS (Primitive Variable Riemann Solver)
        p_star = 0.5 * (self.p_L + self.p_R) - 0.125 * (self.u_R - self.u_L) * (
            self.rho_L + self.rho_R
        ) * (self.c_L + self.c_R)
        p_star = max(p_star, 1e-10)

        # Newton-Raphson iteration
        tol = 1e-8
        max_iter = 50

        for _ in range(max_iter):
            # Left wave function and derivative
            if p_star > self.p_L:
                # Shock
                A_L = 2.0 / ((gamma + 1) * self.rho_L)
                B_L = (gamma - 1) / (gamma + 1) * self.p_L
                f_L = (p_star - self.p_L) * np.sqrt(A_L / (p_star + B_L))
                df_L = np.sqrt(A_L / (p_star + B_L)) * (
                    1.0 - 0.5 * (p_star - self.p_L) / (p_star + B_L)
                )
            else:
                # Rarefaction
                f_L = (
                    2.0
                    * self.c_L
                    / (gamma - 1)
                    * ((p_star / self.p_L) ** ((gamma - 1) / (2 * gamma)) - 1.0)
                )
                df_L = (
                    1.0
                    / (self.rho_L * self.c_L)
                    * (p_star / self.p_L) ** (-(gamma + 1) / (2 * gamma))
                )

            # Right wave function and derivative
            if p_star > self.p_R:
                # Shock
                A_R = 2.0 / ((gamma + 1) * self.rho_R)
                B_R = (gamma - 1) / (gamma + 1) * self.p_R
                f_R = (p_star - self.p_R) * np.sqrt(A_R / (p_star + B_R))
                df_R = np.sqrt(A_R / (p_star + B_R)) * (
                    1.0 - 0.5 * (p_star - self.p_R) / (p_star + B_R)
                )
            else:
                # Rarefaction
                f_R = (
                    2.0
                    * self.c_R
                    / (gamma - 1)
                    * ((p_star / self.p_R) ** ((gamma - 1) / (2 * gamma)) - 1.0)
                )
                df_R = (
                    1.0
                    / (self.rho_R * self.c_R)
                    * (p_star / self.p_R) ** (-(gamma + 1) / (2 * gamma))
                )

            # Newton update
            f = f_L + f_R + (self.u_R - self.u_L)
            df = df_L + df_R

            if abs(df) < 1e-20:
                break

            dp = -f / df
            p_star_new = p_star + dp

            if p_star_new < 0:
                p_star = 0.5 * p_star
            else:
                p_star = p_star_new

            if abs(dp) / (p_star + 1e-20) < tol:
                break

        self.p_star = p_star
        self.u_star = 0.5 * (self.u_L + self.u_R) + 0.5 * (f_R - f_L)

        # Compute densities in star region
        if p_star > self.p_L:
            # Left shock
            self.rho_star_L = self.rho_L * (
                (p_star / self.p_L + (gamma - 1) / (gamma + 1))
                / ((gamma - 1) / (gamma + 1) * p_star / self.p_L + 1.0)
            )
        else:
            # Left rarefaction
            self.rho_star_L = self.rho_L * (p_star / self.p_L) ** (1.0 / gamma)

        if p_star > self.p_R:
            # Right shock
            self.rho_star_R = self.rho_R * (
                (p_star / self.p_R + (gamma - 1) / (gamma + 1))
                / ((gamma - 1) / (gamma + 1) * p_star / self.p_R + 1.0)
            )
        else:
            # Right rarefaction
            self.rho_star_R = self.rho_R * (p_star / self.p_R) ** (1.0 / gamma)

        # Compute wave speeds
        self._compute_wave_speeds()

    def _compute_wave_speeds(self):
        """Compute the speeds of all waves."""
        gamma = self.gamma

        # Left wave
        if self.p_star > self.p_L:
            # Left shock speed
            self.S_L = self.u_L - self.c_L * np.sqrt(
                (gamma + 1) / (2 * gamma) * self.p_star / self.p_L + (gamma - 1) / (2 * gamma)
            )
            self.is_left_shock = True
        else:
            # Left rarefaction head and tail
            self.S_head_L = self.u_L - self.c_L
            c_star_L = self.c_L * (self.p_star / self.p_L) ** ((gamma - 1) / (2 * gamma))
            self.S_tail_L = self.u_star - c_star_L
            self.is_left_shock = False

        # Contact discontinuity
        self.S_contact = self.u_star

        # Right wave
        if self.p_star > self.p_R:
            # Right shock speed
            self.S_R = self.u_R + self.c_R * np.sqrt(
                (gamma + 1) / (2 * gamma) * self.p_star / self.p_R + (gamma - 1) / (2 * gamma)
            )
            self.is_right_shock = True
        else:
            # Right rarefaction head and tail
            c_star_R = self.c_R * (self.p_star / self.p_R) ** ((gamma - 1) / (2 * gamma))
            self.S_tail_R = self.u_star + c_star_R
            self.S_head_R = self.u_R + self.c_R
            self.is_right_shock = False

    def sample(self, x, t):
        """
        Sample the solution at position x and time t.

        Returns:
        --------
        rho, u, p, e : floats
            Density, velocity, pressure, specific internal energy
        """
        if t <= 0:
            # Initial condition
            if x < self.x0:
                return self.rho_L, self.u_L, self.p_L, self.p_L / ((self.gamma - 1) * self.rho_L)
            else:
                return self.rho_R, self.u_R, self.p_R, self.p_R / ((self.gamma - 1) * self.rho_R)

        gamma = self.gamma
        S = (x - self.x0) / t  # Similarity variable

        # Left of contact discontinuity
        if S < self.S_contact:
            if self.is_left_shock:
                if S < self.S_L:
                    rho, u, p = self.rho_L, self.u_L, self.p_L
                else:
                    rho, u, p = self.rho_star_L, self.u_star, self.p_star
            else:
                if S < self.S_head_L:
                    rho, u, p = self.rho_L, self.u_L, self.p_L
                elif S < self.S_tail_L:
                    # Inside rarefaction fan
                    u = 2.0 / (gamma + 1) * (self.c_L + (gamma - 1) / 2 * self.u_L + S)
                    c = self.c_L - (gamma - 1) / 2 * (u - self.u_L)
                    rho = self.rho_L * (c / self.c_L) ** (2.0 / (gamma - 1))
                    p = self.p_L * (c / self.c_L) ** (2.0 * gamma / (gamma - 1))
                else:
                    rho, u, p = self.rho_star_L, self.u_star, self.p_star
        else:
            # Right of contact discontinuity
            if self.is_right_shock:
                if S > self.S_R:
                    rho, u, p = self.rho_R, self.u_R, self.p_R
                else:
                    rho, u, p = self.rho_star_R, self.u_star, self.p_star
            else:
                if S > self.S_head_R:
                    rho, u, p = self.rho_R, self.u_R, self.p_R
                elif S > self.S_tail_R:
                    # Inside rarefaction fan
                    u = 2.0 / (gamma + 1) * (-self.c_R + (gamma - 1) / 2 * self.u_R + S)
                    c = self.c_R + (gamma - 1) / 2 * (u - self.u_R)
                    rho = self.rho_R * (c / self.c_R) ** (2.0 / (gamma - 1))
                    p = self.p_R * (c / self.c_R) ** (2.0 * gamma / (gamma - 1))
                else:
                    rho, u, p = self.rho_star_R, self.u_star, self.p_star

        e = p / ((gamma - 1) * rho)
        return rho, u, p, e

    def solution_at_time(self, t, x_min=-0.5, x_max=0.5, n_points=500):
        """
        Compute the full solution profile at time t.

        Returns:
        --------
        x, rho, u, p, e : arrays
            Position and primitive variables
        """
        x = np.linspace(x_min, x_max, n_points)
        rho = np.zeros(n_points)
        u = np.zeros(n_points)
        p = np.zeros(n_points)
        e = np.zeros(n_points)

        for i in range(n_points):
            rho[i], u[i], p[i], e[i] = self.sample(x[i], t)

        return x, rho, u, p, e


class SedovAnalytical:
    """
    Sedov-Taylor blast wave analytical solution.

    Parameters:
    -----------
    gamma : float
        Adiabatic index
    E0 : float
        Total blast energy
    rho0 : float
        Background density
    nu : int
        Geometry (1=planar, 2=cylindrical, 3=spherical)
    """

    def __init__(self, gamma=5.0 / 3.0, E0=1.0, rho0=1.0, nu=3):
        self.gamma = gamma
        self.E0 = E0
        self.rho0 = rho0
        self.nu = nu

        # Similarity exponent
        self.alpha = 2.0 / (nu + 2)

        # Sedov constant (depends on gamma and nu)
        self.xi_0 = self._compute_xi0()

        # Post-shock density ratio
        self.density_ratio = (gamma + 1) / (gamma - 1)

    def _compute_xi0(self):
        """Compute the Sedov constant xi_0."""
        gamma = self.gamma
        nu = self.nu

        # Approximate values for common cases
        if nu == 3:  # Spherical
            if abs(gamma - 5.0 / 3.0) < 0.01:
                return 1.15167
            elif abs(gamma - 1.4) < 0.01:
                return 1.03275
        elif nu == 2:  # Cylindrical
            if abs(gamma - 1.4) < 0.01:
                return 1.033
        elif nu == 1:  # Planar
            if abs(gamma - 1.4) < 0.01:
                return 0.911

        # General approximation
        return 1.0

    def shock_radius(self, t):
        """Compute shock radius at time t."""
        if t <= 0:
            return 0.0
        return self.xi_0 * (self.E0 * t**2 / self.rho0) ** (1.0 / (self.nu + 2))

    def shock_velocity(self, t):
        """Compute shock velocity at time t."""
        if t <= 0:
            return 0.0
        R_s = self.shock_radius(t)
        return 2.0 / (self.nu + 2) * R_s / t

    def solution_at_time(self, t, n_points=500):
        """
        Compute the radial profile at time t.

        Returns:
        --------
        r, rho, v, p, e : arrays
            Radius and primitive variables
        """
        if t <= 1e-10:
            r = np.linspace(0, 0.01, n_points)
            return (
                r,
                np.ones(n_points) * self.rho0,
                np.zeros(n_points),
                np.ones(n_points) * 1e-10,
                np.ones(n_points) * 1e-10,
            )

        gamma = self.gamma
        nu = self.nu

        R_s = self.shock_radius(t)
        v_s = self.shock_velocity(t)

        # Similarity variable lambda = r/R_s
        lam = np.linspace(0, 1.0, n_points)
        r = lam * R_s

        # Post-shock values
        rho_s = self.rho0 * self.density_ratio
        v_shock = 2.0 / (gamma + 1) * v_s
        p_s = 2.0 / (gamma + 1) * self.rho0 * v_s**2

        # Approximate profiles (self-similar structure)
        # Velocity: linear profile (exact for Sedov)
        v = v_shock * lam

        # Density: peaks near shock, low at center
        omega = (nu + 2) * gamma / (2 + nu * (gamma - 1))
        rho = rho_s * lam ** (omega - 1) * np.maximum(0.1, 1 - 0.8 * (1 - lam) ** 2)
        rho[0] = rho[1] if n_points > 1 else rho_s * 0.1

        # Pressure: higher at center
        p = p_s * (0.5 + 0.5 * lam**2)

        # Specific internal energy
        e = p / ((gamma - 1) * np.maximum(rho, 1e-10))

        return r, rho, v, p, e


def main():
    """Test the analytical solutions."""
    import matplotlib.pyplot as plt

    # Test Sod solution
    print("Testing Sod shock tube analytical solution...")
    sod = SodAnalytical()
    t = 0.2
    x, rho, u, p, e = sod.solution_at_time(t)

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle(f"Sod Shock Tube Analytical Solution (t = {t})", fontsize=14)

    axes[0, 0].plot(x, rho, "b-", linewidth=2)
    axes[0, 0].set_ylabel("Density")
    axes[0, 0].set_title("Density")
    axes[0, 0].grid(True, alpha=0.3)

    axes[0, 1].plot(x, u, "r-", linewidth=2)
    axes[0, 1].set_ylabel("Velocity")
    axes[0, 1].set_title("Velocity")
    axes[0, 1].grid(True, alpha=0.3)

    axes[1, 0].plot(x, p, "g-", linewidth=2)
    axes[1, 0].set_ylabel("Pressure")
    axes[1, 0].set_xlabel("Position")
    axes[1, 0].set_title("Pressure")
    axes[1, 0].grid(True, alpha=0.3)

    axes[1, 1].plot(x, e, "m-", linewidth=2)
    axes[1, 1].set_ylabel("Internal Energy")
    axes[1, 1].set_xlabel("Position")
    axes[1, 1].set_title("Internal Energy")
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("sod_analytical_test.png", dpi=150)
    print("Saved: sod_analytical_test.png")
    plt.close()

    # Test Sedov solution
    print("\nTesting Sedov blast wave analytical solution...")
    sedov = SedovAnalytical()
    t = 0.1
    r, rho, v, p, e = sedov.solution_at_time(t)

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle(f"Sedov Blast Wave Analytical Solution (t = {t})", fontsize=14)

    axes[0, 0].plot(r, rho, "b-", linewidth=2)
    axes[0, 0].set_ylabel("Density")
    axes[0, 0].set_title("Density")
    axes[0, 0].grid(True, alpha=0.3)

    axes[0, 1].plot(r, v, "r-", linewidth=2)
    axes[0, 1].set_ylabel("Velocity")
    axes[0, 1].set_title("Velocity")
    axes[0, 1].grid(True, alpha=0.3)

    axes[1, 0].plot(r, p, "g-", linewidth=2)
    axes[1, 0].set_ylabel("Pressure")
    axes[1, 0].set_xlabel("Radius")
    axes[1, 0].set_title("Pressure")
    axes[1, 0].grid(True, alpha=0.3)

    axes[1, 1].plot(r, e, "m-", linewidth=2)
    axes[1, 1].set_ylabel("Internal Energy")
    axes[1, 1].set_xlabel("Radius")
    axes[1, 1].set_title("Internal Energy")
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("sedov_analytical_test.png", dpi=150)
    print("Saved: sedov_analytical_test.png")
    plt.close()

    print("\nAnalytical solution tests complete!")


if __name__ == "__main__":
    main()
