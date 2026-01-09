import numpy as np

from .EquationOfState import IdealEquationOfState
from .State import State
from .Util import *


class Shock:
    @classmethod
    def fromStateAheadAndSpeedPressureBehind(
        cls, stateA, speedB, pressureB, eos: IdealEquationOfState, sign
    ):
        hB = Shock.computeTaubAdiabat(stateA, pressureB, eos)
        A = computeA(stateA, eos)

        stateB = State()
        stateB.pressure = pressureB
        stateB.vx = speedB
        stateB.vt = computeVt(A, speedB, hB)
        stateB.rho = eos.computeRho(pressureB, hB)
        return cls(stateA, stateB, eos, sign)

    @staticmethod
    def computeTaubAdiabat(stateA, pressureB, eos: IdealEquationOfState):
        hA = eos.computeEnthalpy(stateA.pressure, stateA.rho)
        c_2 = 1.0 + (stateA.pressure - pressureB) / (pressureB * eos.sigma)
        c_1 = -(stateA.pressure - pressureB) / (pressureB * eos.sigma)
        c_0 = hA * (stateA.pressure - pressureB) / stateA.rho - hA**2
        if c_2 == 0:
            hB = -c_0 / c_1
        else:
            hB = (-c_1 + np.sqrt(c_1 * c_1 - 4 * c_2 * c_0)) / (2 * c_2)
        return hB

    @staticmethod
    def computeJ(stateA, stateB, eos: IdealEquationOfState):
        return np.sqrt(
            (stateB.pressure - stateA.pressure)
            / (
                eos.computeEnthalpy(stateA.pressure, stateA.rho) / stateA.rho
                - eos.computeEnthalpy(stateB.pressure, stateB.rho) / stateB.rho
            )
        )

    @staticmethod
    def computeShockSpeed(stateA, J, sign):
        D = stateA.rho * stateA.lorentz()
        return (D**2 * stateA.vx + sign * J * np.sqrt(J**2 + D**2 * (1 - stateA.vx**2))) / (
            D**2 + J**2
        )

    # @staticmethod
    # def computeJ_sqr(stateA, pressureB, eos: IdealEquationOfState):
    #     hA = eos.computeEnthalpy(stateA.pressure, stateA.rho)
    #     hB = Shock.computeTaubAdiabat(stateA, pressureB, eos)
    #     return - eos.sigma * (stateA.pressure - pressureB) / (
    #             hA * (hA - 1.) / stateA.pressure - hB * (hB - 1.) / pressureB)

    @staticmethod
    def computeJ_sqr(pressureA, pressureB, hA, hB, eos: IdealEquationOfState):
        return (
            -eos.sigma
            * (pressureA - pressureB)
            / (hA * (hA - 1.0) / pressureA - hB * (hB - 1.0) / pressureB)
        )

    @staticmethod
    def computeVxb(stateA, pressureB, eos, sign):
        hA = eos.computeEnthalpy(stateA.pressure, stateA.rho)
        hB = Shock.computeTaubAdiabat(stateA, pressureB, eos)
        J_sqr = Shock.computeJ_sqr(stateA.pressure, pressureB, hA, hB, eos)
        J = np.sqrt(np.abs(J_sqr))
        Vs = Shock.computeShockSpeed(stateA, J, sign)
        Ws = computeLorentz(Vs)
        hA = eos.computeEnthalpy(stateA.pressure, stateA.rho)

        return (
            hA * stateA.lorentz() * stateA.vx + sign * Ws * (pressureB - stateA.pressure) / J
        ) / (
            hA * stateA.lorentz()
            + (pressureB - stateA.pressure)
            * (sign * Ws * stateA.vx / J + 1 / (stateA.rho * stateA.lorentz()))
        )

    def __init__(self, stateA, stateB, eos, sign):
        self.stateA = stateA
        self.stateB = stateB
        self.sign = sign

        self.eos = eos
        j = self.computeJ(stateA, stateB, self.eos)
        self.speed = self.computeShockSpeed(stateA, j, sign)

    def __str__(self):
        return f"Shock: v={self.speed:.3f}"

    def __repr__(self):
        return str(self)
