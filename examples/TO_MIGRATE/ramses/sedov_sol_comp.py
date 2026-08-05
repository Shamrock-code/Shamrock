

import glob
import os
import sys
import argparse
import numpy as np
import re
import matplotlib.pyplot as plt


import shamrock

gamma = 5.0 / 3.0

sedov_sol = shamrock.phys.SedovTaylor()
r_theo = np.linspace(0, 1, 300)
p_theo = []
vr_theo = []
rho_theo = []
for i in range(len(r_theo)):
    _rho_theo, _vr_theo, _p_theo = sedov_sol.get_value(r_theo[i])
    p_theo.append(_p_theo)
    vr_theo.append(_vr_theo)
    rho_theo.append(_rho_theo)

r_theo = np.array(r_theo)
p_theo = np.array(p_theo)
vr_theo = np.array(vr_theo)
rho_theo = np.array(rho_theo)
eint_theo = p_theo / ((gamma - 1.0 )* rho_theo)

output = np.column_stack((np.array(rho_theo), np.array(vr_theo), np.array(p_theo), np.array(eint_theo)))
filename = f"sod_theorical.txt"

np.savetxt(filename,
            output,
            fmt=["%.10f",  "%.10f", "%.10f",  "%.10f"],
            header="rho_r    vr      p_r    eint_r",
            )

