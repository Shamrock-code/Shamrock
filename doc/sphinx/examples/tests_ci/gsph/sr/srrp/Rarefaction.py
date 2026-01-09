from scipy import integrate, optimize

from .EquationOfState import EntropyConservation, IdealEquationOfState
from .State import State
from .Util import *


class Rarefaction:
    @classmethod
    def fromStateAheadAndSpeedPressureBehind(
        cls, stateA, speedB, pressureB, eos: IdealEquationOfState, sign
    ):
        isentrope = EntropyConservation.fromState(stateA, eos)

        A = computeA(stateA, eos)
        h = isentrope.computeEnthalpy(pressureB)
        stateB = State()
        stateB.pressure = pressureB
        stateB.vx = speedB
        stateB.vt = computeVt(A, stateB.vx, h)
        stateB.rho = isentrope.computeRho(pressureB)
        return cls(stateA, stateB, eos, sign)

    @staticmethod
    def ux(xi, pressure, A, isentrope: EntropyConservation, sign):
        cs = isentrope.computeSpeedOfSound(pressure)
        h = isentrope.computeEnthalpy(pressure)
        a = cs * h
        b = sign * np.sqrt(A**2 * (1 - cs**2) + h**2)
        return (a - b * xi) / (a * xi - b)

    @staticmethod
    def computeVxb(stateA, pressureB, isentrope: EntropyConservation, sign, A=None):
        if A is None:
            A = computeA(stateA, isentrope)
        A_sqr = A**2

        def integrand(_pressure):
            rho = isentrope.computeRho(_pressure)
            h = isentrope.computeEnthalpy(_pressure, rho=rho)
            cs = isentrope.computeSpeedOfSound(_pressure, rho=rho, h=h)
            h_sqr = h**2
            cs_sqr = cs**2
            return np.sqrt(h_sqr + A_sqr * (1 - cs_sqr)) / ((h_sqr + A_sqr) * rho * cs)

        B = (
            0.5 * np.log((1 + stateA.vx) / (1 - stateA.vx))
            + sign * integrate.quad(integrand, stateA.pressure, pressureB)[0]
        )
        return np.tanh(B)

    @staticmethod
    def xi_interface(state, isentrope: EntropyConservation, sign):
        speed_sqr = state.speed() ** 2
        cs = isentrope.computeSpeedOfSound(state.pressure)
        cs_sqr = cs**2
        return (
            state.vx * (1 - cs_sqr)
            + sign
            * cs
            * np.sqrt((1 - speed_sqr) * (1 - speed_sqr * cs_sqr - state.vx**2 * (1 - cs_sqr)))
        ) / (1 - speed_sqr * cs_sqr)

    def __init__(self, stateA, stateB, eos: IdealEquationOfState, sign):
        self.stateA = stateA
        self.stateB = stateB
        self.eos = eos

        self.sign = sign
        self.isentrope = EntropyConservation.fromState(stateA, eos)
        assert (
            np.abs(
                self.isentrope.isentropicConstant
                - EntropyConservation.fromState(stateB, eos).isentropicConstant
            )
            < 1e-10 * self.isentrope.isentropicConstant
        )

        self.speedHead = self.xi_interface(stateA, self.isentrope, sign)
        self.speedTail = self.xi_interface(stateB, self.isentrope, sign)

        self.A = computeA(stateA, eos)

    def computeRarefactionState(self, xi):
        pmin = min(self.stateA.pressure, self.stateB.pressure)
        pmax = max(self.stateA.pressure, self.stateB.pressure)
        state = State()

        state.pressure = optimize.brentq(
            lambda pressure: self.ux(xi, pressure, self.A, self.isentrope, self.sign)
            - self.computeVxb(self.stateA, pressure, self.isentrope, self.sign, A=self.A),
            pmin,
            pmax,
        )

        state.rho = self.isentrope.computeRho(state.pressure)
        state.vx = self.ux(xi, state.pressure, self.A, self.isentrope, self.sign)
        h = self.isentrope.computeEnthalpy(state.pressure, rho=state.rho)
        state.vt = computeVt(self.A, state.vx, h)
        return state

    def __str__(self):
        return f"Rarefaction: vHead={self.speedHead:.3f}, vTail={self.speedTail:.3f}"

    def __repr__(self):
        return str(self)
