"""
Standard SPH disc setup helpers.
"""

# pylint: disable=invalid-name

from dataclasses import dataclass
from typing import Any, Literal

import shamrock
from shamrock.utils.numba_helper import maybe_njit

RotationMode = Literal["keplerian", "subkeplerian", "subkeplerian_3d"]
_VALID_ROTATIONS = ("keplerian", "subkeplerian", "subkeplerian_3d")


@dataclass
class DiscProfiles:
    """
    Helper class to store the profiles of the disc.
    """

    sigma: Any
    H: Any
    vtheta_kepler: Any
    omega_k: Any
    cs: Any
    vtheta: Any = None
    velocity: Any = None
    cs_field: Any = None


@dataclass
class StandardDisc:
    """
    Locally isothermal LP07 disc profiles and Monte Carlo generator helper.

    All radii and masses are expressed in the provided code unit system.
    """

    units: shamrock.UnitSystem
    center_mass: float
    disc_mass: float
    rin: float
    rout: float
    H_r_0: float = 0.05
    q: float = 0.5
    p: float = 1.5
    r0: float = 1.0
    H_factor: float = 1.0
    rotation: RotationMode = "subkeplerian"
    inner_tapering: bool = False
    outer_tapering: bool = False

    def __post_init__(self) -> None:
        if self.rin >= self.rout:
            raise ValueError("rin must be smaller than rout")
        if self.center_mass <= 0:
            raise ValueError("center_mass must be positive")
        if self.disc_mass <= 0:
            raise ValueError("disc_mass must be positive")
        if self.rotation not in _VALID_ROTATIONS:
            raise ValueError(f"rotation must be one of {_VALID_ROTATIONS}, got {self.rotation!r}")

    def _G(self) -> float:
        return shamrock.Constants(self.units).G()

    def _get_sigma(self) -> Any:
        if self.outer_tapering:
            raise NotImplementedError("outer_tapering is not implemented yet")

        rin = self.rin
        r0 = self.r0
        p = self.p
        inner_tapering = self.inner_tapering

        def sigma(r: float) -> float:
            sigma_0 = 1.0
            if inner_tapering:
                sigma_0 *= 1 - (rin / r) ** 0.5
            return sigma_0 * (r / r0) ** (-p)

        return maybe_njit(sigma)

    def _get_vtheta_kepler(self) -> Any:
        G = self._G()
        center_mass = self.center_mass

        def vtheta_kepler(r: float) -> float:
            return (G * center_mass / r) ** 0.5

        return maybe_njit(vtheta_kepler)

    def _get_omega_k(self, vtheta_kepler: Any) -> Any:
        def omega_k(r: float) -> float:
            return vtheta_kepler(r) / r

        return maybe_njit(omega_k)

    def _get_cs(self, vtheta_kepler: Any) -> Any:
        H_r_0 = self.H_r_0
        q = self.q
        r0 = self.r0
        cs_in = H_r_0 * vtheta_kepler(r0)  # == (H_r_0 * r0) * omega_k(r0)

        def cs(r: float) -> float:
            return ((r / r0) ** (-q)) * cs_in

        return maybe_njit(cs)

    def _get_H(self, cs: Any, omega_k: Any) -> Any:
        H_factor = self.H_factor

        def H(r: float) -> float:
            return H_factor * cs(r) / omega_k(r)

        return maybe_njit(H)

    def _get_vtheta_subkeplerian(self, vtheta_kepler: Any, cs: Any) -> Any:
        p, q = self.p, self.q

        def vtheta(r: float) -> float:
            return ((vtheta_kepler(r) ** 2) - (2 * p + q) * cs(r) ** 2) ** 0.5

        return maybe_njit(vtheta)

    def _get_velocity_vertical(self, vtheta_kepler: Any, cs: Any) -> Any:
        p, q = self.p, self.q

        def vtheta_rz(r: float, z: float) -> float:
            gm_r = vtheta_kepler(r) ** 2
            term2 = -(p + q + 3.0 / 2.0) * cs(r) ** 2
            r2z2_sqrt = (r**2 + z**2) ** 0.5
            term3 = -gm_r * r * (2 * q) * (1 / r - 1 / r2z2_sqrt)
            return (gm_r + term2 + term3) ** 0.5

        vtheta_rz = maybe_njit(vtheta_rz)

        def velocity(pos):
            x, y, z = pos[0], pos[1], pos[2]
            r = (x**2 + y**2) ** 0.5
            v_mag = vtheta_rz(r, z)
            return (v_mag * (-y / r), v_mag * (x / r), 0.0)

        return maybe_njit(velocity)

    def _get_rotation(self, vtheta_kepler: Any, cs: Any) -> tuple[Any, Any]:
        if self.rotation == "keplerian":
            return vtheta_kepler, None
        if self.rotation == "subkeplerian":
            return self._get_vtheta_subkeplerian(vtheta_kepler, cs), None
        return None, self._get_velocity_vertical(vtheta_kepler, cs)

    def get_profiles(self) -> DiscProfiles:
        """
        Get the profiles of the disc.
        """
        sigma = self._get_sigma()
        vtheta_kepler = self._get_vtheta_kepler()
        omega_k = self._get_omega_k(vtheta_kepler)
        cs = self._get_cs(vtheta_kepler)
        H = self._get_H(cs, omega_k)
        vtheta, velocity = self._get_rotation(vtheta_kepler, cs)

        return DiscProfiles(
            sigma=sigma,
            H=H,
            vtheta_kepler=vtheta_kepler,
            omega_k=omega_k,
            cs=cs,
            vtheta=vtheta,
            velocity=velocity,
        )

    def part_mass(self, npart: int) -> float:
        """
        Get the mass of a single particle from the total mass & number of particles.
        """
        return self.disc_mass / npart

    def cs0(self) -> float:
        """
        Get the sound speed at the reference radius.
        """
        return self.get_profiles().cs(self.r0)

    def make_generator(
        self,
        setup: Any,
        npart: int,
        *,
        random_seed: int = 666,
        init_h_factor: float = 0.8,
    ):
        """
        Make a SPH generator for the disc.
        """
        profiles = self.get_profiles()
        kwargs = {
            "part_mass": self.part_mass(npart),
            "disc_mass": self.disc_mass,
            "r_in": self.rin,
            "r_out": self.rout,
            "sigma_profile": profiles.sigma,
            "H_profile": profiles.H,
            "random_seed": random_seed,
            "init_h_factor": init_h_factor,
        }
        if profiles.velocity is not None:
            kwargs["velocity_field"] = profiles.velocity
        else:
            kwargs["rot_profile"] = profiles.vtheta
        if profiles.cs_field is not None:
            kwargs["cs_field"] = profiles.cs_field
        else:
            kwargs["cs_profile"] = profiles.cs
        return setup.make_generator_disc_mc(**kwargs)
