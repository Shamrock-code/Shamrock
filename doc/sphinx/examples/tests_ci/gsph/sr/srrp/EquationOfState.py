import numpy as np


class IdealEquationOfState:
    def __init__(self, gamma):
        self.gamma = gamma
        self.sigma = gamma / (gamma - 1)

    def computeSpeedOfSound(self, pressure, rho):
        h = self.computeEnthalpy(pressure, rho)
        return np.sqrt(self.gamma * pressure / (h * rho))

    def computeEnthalpy(self, pressure, rho):
        return 1 + self.sigma * pressure / rho

    def computeRho(self, pressure, h):
        return self.sigma / (h - 1) * pressure

    def __eq__(self, other):
        return self.gamma == other.gamma


class EntropyConservation:
    @classmethod
    def fromState(cls, state, eos):
        isentropicConstant = EntropyConservation.computeSpecificEntropy(state, eos)
        return cls(isentropicConstant, eos)

    @staticmethod
    def computeSpecificEntropy(state, eos):
        return state.pressure / np.power(state.rho, eos.gamma)

    def __init__(self, isentropicConstant, eos: IdealEquationOfState):
        self.isentropicConstant = isentropicConstant
        self.eos = eos

    def computeRho(self, pressure):
        return np.power(pressure / self.isentropicConstant, 1.0 / self.eos.gamma)

    def computeSpeedOfSound(self, pressure, rho=None, h=None):
        if rho is None:
            rho = self.computeRho(pressure)
        if h is None:
            h = self.computeEnthalpy(pressure, rho)
        return np.sqrt(self.eos.gamma * pressure / (h * rho))

    def computeEnthalpy(self, pressure, rho=None):
        if rho is None:
            rho = self.computeRho(pressure)
        return 1 + self.eos.sigma * pressure / rho

    def __eq__(self, other):
        return (self.eos == other.eos) and (self.isentropicConstant == other.isentropicConstant)
