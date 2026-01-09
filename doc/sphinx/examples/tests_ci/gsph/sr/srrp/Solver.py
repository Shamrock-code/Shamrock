import copy

import numpy as np
import scipy.integrate as integrate
import scipy.optimize as opt

from .ContactDiscontinuity import ContactDiscontinuity
from .EquationOfState import EntropyConservation, IdealEquationOfState
from .Rarefaction import Rarefaction
from .Shock import Shock
from .State import State
from .Util import *
from .Wavefan import Wavefan


def getWavefan(
    state1: State,
    state6: State,
    cdSpeed,
    cdPressure,
    eos: IdealEquationOfState,
    waveLType,
    waveRtype,
    reversed=False,
):
    waveL = waveLType.fromStateAheadAndSpeedPressureBehind(
        state1, cdSpeed, cdPressure, eos, sign=-1
    )
    state3 = waveL.stateB
    waveR = waveRtype.fromStateAheadAndSpeedPressureBehind(
        state6, cdSpeed, cdPressure, eos, sign=+1
    )
    state4 = waveR.stateB
    cd = ContactDiscontinuity(cdSpeed, cdPressure)
    states = [state1, state3, state4, state6]
    waves = [waveL, cd, waveR]

    return Wavefan(states, waves, reversed)


class Solver:
    """
    Solve SRHD Riemann Problem with non-zero tangential velocity:
        - Determine wave pattern (Shock-Shock, Rarefaction-Shock, etc.)
        - Compute p_star (pressure of the contact discontinuity)

    see paper
        Rezzolla_Zanotti_2003_Fluid_Mech_479
        DOI: 10.1017/S0022112002003506
    """

    def get_du_SS(self, p_star):
        ux3 = Shock.computeVxb(self.state1, p_star, self.eos, sign=-1)
        ux4 = Shock.computeVxb(self.state6, p_star, self.eos, sign=+1)
        v13 = relativeSpeed(self.state1.vx, ux3)
        v64 = relativeSpeed(self.state6.vx, ux4)

        return relativeSpeed(v13, v64)

    def get_du_RS(self, p_star):
        isentrope1 = EntropyConservation.fromState(self.state1, self.eos)
        ux3 = Rarefaction.computeVxb(self.state1, p_star, isentrope1, sign=-1)
        ux4 = Shock.computeVxb(self.state6, p_star, self.eos, sign=+1)
        v13 = relativeSpeed(self.state1.vx, ux3)
        v64 = relativeSpeed(self.state6.vx, ux4)
        return relativeSpeed(v13, v64)

    def get_du_RR(self, p_star):
        isentrope1 = EntropyConservation.fromState(self.state1, self.eos)
        isentrope6 = EntropyConservation.fromState(self.state6, self.eos)
        ux3 = Rarefaction.computeVxb(self.state1, p_star, isentrope1, sign=-1)
        ux4 = Rarefaction.computeVxb(self.state6, p_star, isentrope6, sign=+1)
        v13 = relativeSpeed(self.state1.vx, ux3)
        v64 = relativeSpeed(self.state6.vx, ux4)
        return relativeSpeed(v13, v64)

    def get_Vs_lim(self):
        """
        see Appendix C in Paper
        """
        h3p = Shock.computeTaubAdiabat(self.state6, self.state1.pressure, self.eos)
        h6 = self.eos.computeEnthalpy(self.state6.pressure, self.state6.rho)
        J23p_sqr = Shock.computeJ_sqr(self.state6.pressure, self.state1.pressure, h6, h3p, self.eos)
        J23p = np.sqrt(np.abs(J23p_sqr))
        return Shock.computeShockSpeed(self.state6, J23p, sign=+1)

    def get_du_limit_SS(self):
        W6_sqr = self.state6.lorentz() ** 2
        h6 = self.eos.computeEnthalpy(self.state6.pressure, self.state6.rho)
        Vs = self.get_Vs_lim()
        return ((self.state1.pressure - self.state6.pressure) * (1 - self.state6.vx * Vs)) / (
            (Vs - self.state6.vx)
            * (
                h6 * self.state6.rho * W6_sqr * (1 - self.state6.vx**2)
                + self.state1.pressure
                - self.state6.pressure
            )
        )

    def get_du_limit_RS(self):
        A1_sqr = computeA(self.state1, self.eos) ** 2
        isentrope1 = EntropyConservation.fromState(self.state1, self.eos)

        def integrand(_pressure):
            rho = isentrope1.computeRho(_pressure)
            h = isentrope1.computeEnthalpy(_pressure, rho=rho)
            cs = isentrope1.computeSpeedOfSound(_pressure, rho=rho, h=h)
            h_sqr = h**2
            cs_sqr = cs**2
            return np.sqrt(h_sqr + A1_sqr * (1 - cs_sqr)) / ((h_sqr + A1_sqr) * rho * cs)

        return np.tanh(integrate.quad(integrand, self.state1.pressure, self.state6.pressure)[0])

    def get_du_limit_RR(self):
        A1_sqr = computeA(self.state1, self.eos) ** 2
        A6_sqr = computeA(self.state6, self.eos) ** 2
        isentrope1 = EntropyConservation.fromState(self.state1, self.eos)
        isentrope6 = EntropyConservation.fromState(self.state6, self.eos)

        def integrand1(_pressure):
            rho = isentrope1.computeRho(_pressure)
            h = isentrope1.computeEnthalpy(_pressure, rho=rho)
            cs = isentrope1.computeSpeedOfSound(_pressure, rho=rho, h=h)
            h_sqr = h**2
            cs_sqr = cs**2
            return np.sqrt(h_sqr + A1_sqr * (1 - cs_sqr)) / ((h_sqr + A1_sqr) * rho * cs)

        def integrand6(_pressure):
            rho = isentrope6.computeRho(_pressure)
            h = isentrope6.computeEnthalpy(_pressure, rho=rho)
            cs = isentrope6.computeSpeedOfSound(_pressure, rho=rho, h=h)
            h_sqr = h**2
            cs_sqr = cs**2
            return np.sqrt(h_sqr + A6_sqr * (1 - cs_sqr)) / ((h_sqr + A6_sqr) * rho * cs)

        v1c = np.tanh(integrate.quad(integrand1, self.state1.pressure, 0))[0]
        v6c = np.tanh(integrate.quad(integrand6, 0, self.state6.pressure))[0]
        return relativeSpeed(v1c, v6c)

    def solve(self, stateL, stateR, gamma):
        if stateL.pressure >= stateR.pressure:
            self.reversed = False
            self.state1 = copy.copy(stateL)
            self.state6 = copy.copy(stateR)
        else:
            self.reversed = True
            self.state1 = copy.copy(stateR)
            self.state6 = copy.copy(stateL)
            self.state1.vx *= -1
            self.state6.vx *= -1

        self.eos = IdealEquationOfState(gamma)

        return self.determine_wave_pattern()

    def determine_wave_pattern(self):
        du_0 = relativeSpeed(self.state1.vx, self.state6.vx)
        eps = 1e-15

        if du_0 <= self.get_du_limit_RR():
            self.solution_type = "RR*"
            p_star = 0
            ux_star = 0
            solution = getWavefan(
                self.state1,
                self.state6,
                ux_star,
                p_star,
                Rarefaction,
                Rarefaction,
                reversed=self.reversed,
            )

        elif du_0 <= self.get_du_limit_RS():
            self.solution_type = "RR"
            p_min = (self.state6.pressure + eps) * eps
            p_max = self.state1.pressure
            assert p_min < p_max
            p_star = opt.brentq(lambda p: self.get_du_RR(p) - du_0, p_min, p_max)
            isentrope = EntropyConservation.fromState(self.state6, self.eos)
            ux_star = Rarefaction.computeVxb(self.state6, p_star, isentrope, sign=+1)
            solution = getWavefan(
                self.state1,
                self.state6,
                ux_star,
                p_star,
                self.eos,
                Rarefaction,
                Rarefaction,
                reversed=self.reversed,
            )

        elif du_0 <= self.get_du_limit_SS():
            self.solution_type = "RS"
            p_min = self.state6.pressure + eps
            p_max = self.state1.pressure
            assert p_min < p_max
            p_star = opt.brentq(lambda p: self.get_du_RS(p) - du_0, p_min, p_max)
            ux_star = Shock.computeVxb(self.state6, p_star, self.eos, sign=+1)
            solution = getWavefan(
                self.state1,
                self.state6,
                ux_star,
                p_star,
                self.eos,
                Rarefaction,
                Shock,
                reversed=self.reversed,
            )

        else:
            self.solution_type = "SS"
            p_star_guess = 1.1 * self.state1.pressure
            p_star = opt.root(lambda p: self.get_du_SS(p) - du_0, p_star_guess).x[0]
            ux_star = Shock.computeVxb(self.state6, p_star, self.eos, sign=+1)
            solution = getWavefan(
                self.state1,
                self.state6,
                ux_star,
                p_star,
                self.eos,
                Shock,
                Shock,
                reversed=self.reversed,
            )

        return solution
