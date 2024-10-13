import math
from cmath import *

w = 1.9158960 - 4.410541j
def analytical_values(t, w, x, delta) :
    res = 0.0 + 0.0j
    res = delta * exp(-t*w)*exp(pi * x * (2j))
    return res.real, res.imag
