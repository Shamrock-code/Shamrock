import os

import matplotlib.pyplot as plt
import numpy as np

import shamrock

ctx = shamrock.Context()
ctx.pdata_layout_new()

model = shamrock.get_Model_Zeus(context=ctx, vector_type="f64_3", grid_repr="i64_3")

model.init_scheduler(int(1e7), 1)

multx = 1
multy = 1
multz = 1

sz = 1 << 1
base = 64
model.make_base_grid((0, 0, 0), (sz, sz, sz), (base * multx, base * multy, base * multz))


def rho_map(rmin, rmax):
    return 1.0


model.set_field_value_lambda_f64("rho", rho_map)

# model.evolve_once(0,0.1)

for i in range(10):
    model.dump_vtk("test" + str(i) + ".vtk")
    model.evolve_once(0, 0.1)
