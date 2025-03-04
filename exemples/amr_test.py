import os

import matplotlib.pyplot as plt
import numpy as np

import shamrock

ctx = shamrock.Context()
ctx.pdata_layout_new()

ctx.pdata_layout_add_field("cell_min", 1, "u64_3")
ctx.pdata_layout_add_field("cell_max", 1, "u64_3")
ctx.pdata_layout_add_field("sum_field", 1, "u32")

ctx.init_sched(100, 400)

grd = shamrock.AMRGrid(ctx)

sz = 1 << 1

grd.make_base_grid([0, 0, 0], [sz, sz, sz], [128, 128, 128])

# ctx.dump_status()

print("recovered :", len(ctx.collect_data()["cell_min"][::]))

model = shamrock.AMRTestModel(grd)

# model.refine()
# print("recovered :",len(ctx.collect_data()["cell_min"][::]))
#
# model.derefine()
# print("recovered :",len(ctx.collect_data()["cell_min"][::]))

model.step()
dat = ctx.collect_data()
print("recovered :", len(dat["cell_min"][::]))

print(dat["sum_field"])
