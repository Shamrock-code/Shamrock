import numpy as np


def computeLorentz(speed):
    return 1.0 / np.sqrt(1.0 - speed**2)


def relativeSpeed(speedL, speedR):
    return (speedL - speedR) / (1 - speedL * speedR)


def computeVt(invariantA, ux, h):
    return invariantA * np.sqrt((1 - ux**2) / (h**2 + invariantA**2))


def computeA(state, eos):
    h = eos.computeEnthalpy(state.pressure, state.rho)
    return h * state.lorentz() * state.vt
