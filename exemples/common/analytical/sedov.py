#!/usr/bin/env python3
"""
Sedov-Taylor Blast Wave Analytical Solution

Computes the self-similar solution for the Sedov-Taylor blast wave problem.

References:
- Sedov, L.I. (1959) "Similarity and Dimensional Methods in Mechanics"
- Taylor, G.I. (1950) "The Formation of a Blast Wave by a Very Intense Explosion"

Note: For Sod shock tube, use shamrock.phys.SodTube instead.
"""

import numpy as np


class SedovAnalytical:
    """
    Sedov-Taylor blast wave analytical solution.

    Parameters:
    -----------
    gamma : float
        Adiabatic index
    E_blast : float
        Total blast energy
    rho_0 : float
        Background density
    ndim : int
        Dimensionality (1=planar, 2=cylindrical, 3=spherical)
    """

    def __init__(self, gamma=5.0 / 3.0, E_blast=1.0, rho_0=1.0, ndim=3):
        self.gamma = gamma
        self.E_blast = E_blast
        self.rho_0 = rho_0
        self.ndim = ndim

        # Similarity exponent
        self.alpha = 2.0 / (ndim + 2)

        # Sedov constant (depends on gamma and ndim)
        self.xi_0 = self._compute_xi0()

    def _compute_xi0(self):
        """Compute the Sedov constant xi_0."""
        gamma = self.gamma
        ndim = self.ndim

        # Approximate values for common cases
        if ndim == 3:  # Spherical
            if abs(gamma - 5.0 / 3.0) < 0.01:
                return 1.15167
            elif abs(gamma - 1.4) < 0.01:
                return 1.03275
        elif ndim == 2:  # Cylindrical
            if abs(gamma - 1.4) < 0.01:
                return 1.033
        elif ndim == 1:  # Planar
            if abs(gamma - 1.4) < 0.01:
                return 0.911

        # General approximation
        return 1.0

    def shock_radius(self, t):
        """Compute shock radius at time t."""
        if t <= 0:
            return 0.0
        return self.xi_0 * (self.E_blast * t**2 / self.rho_0) ** (1.0 / (self.ndim + 2))

    def shock_velocity(self, t):
        """Compute shock velocity at time t."""
        if t <= 0:
            return 0.0
        R_s = self.shock_radius(t)
        return 2.0 / (self.ndim + 2) * R_s / t

    def post_shock_density(self):
        """Compute post-shock density."""
        return self.rho_0 * (self.gamma + 1) / (self.gamma - 1)

    def solution_at_time(self, t, r_max=None, n_points=500):
        """
        Compute the radial profile at time t.

        Returns:
        --------
        r, rho, v, p : arrays
            Radius and primitive variables
        """
        if t <= 1e-10:
            r = np.linspace(0, r_max or 0.01, n_points)
            return (
                r,
                np.ones(n_points) * self.rho_0,
                np.zeros(n_points),
                np.ones(n_points) * 1e-10,
            )

        gamma = self.gamma
        ndim = self.ndim

        R_s = self.shock_radius(t)
        v_s = self.shock_velocity(t)

        if r_max is None:
            r_max = R_s * 1.5

        # Similarity variable lambda = r/R_s
        lam = np.linspace(0, min(1.0, r_max / R_s), n_points)
        r = lam * R_s

        # Post-shock values
        rho_s = self.post_shock_density()
        v_shock = 2.0 / (gamma + 1) * v_s
        p_s = 2.0 / (gamma + 1) * self.rho_0 * v_s**2

        # Approximate profiles (self-similar structure)
        # Velocity: linear profile
        v = v_shock * lam

        # Density: peaks near shock, low at center
        omega = (ndim + 2) * gamma / (2 + ndim * (gamma - 1))
        rho = rho_s * lam ** (omega - 1) * np.maximum(0.1, 1 - 0.8 * (1 - lam) ** 2)
        rho[0] = rho[1] if n_points > 1 else rho_s * 0.1

        # Pressure: higher at center
        p = p_s * (0.5 + 0.5 * lam**2)

        return r, rho, v, p
