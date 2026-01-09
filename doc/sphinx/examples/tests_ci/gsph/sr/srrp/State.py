import numpy as np

from .Util import *


class State:
    def __init__(self, rho=0, vx=0, vt=0, pressure=0):
        self.rho = rho
        self.vx = vx
        self.vt = vt
        self.pressure = pressure

    def speed(self):
        return np.sqrt(self.vx**2 + self.vt**2)

    def lorentz(self):
        return computeLorentz(self.speed())

    def __eq__(self, other):
        return (
            (self.rho == other.rho)
            * (self.vx == other.vx)
            * (self.vt == other.vt)
            * (self.pressure == other.pressure)
        )

    def __str__(self):
        return f"State: rho={self.rho:.3f}, vx={self.vx:.3f}, vt={self.vt:.3f}, pressure={self.pressure:.3f}"

    def __repr__(self):
        return str(self)
