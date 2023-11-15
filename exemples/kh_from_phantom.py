"""
Matching the KH to Phantom's
"""

import shamrock
import matplotlib.pyplot as plt
import numpy as np

outputdir = "your_output_directory/"
ph_dir = "directory_of_phantom_dumps/"
ph_file = ph_dir + "dumpname_00000"

ctx = shamrock.Context()
ctx.pdata_layout_new()
model = shamrock.get_SPHModel(context = ctx, vector_type = "f64_3",sph_kernel = "M4")

dump = shamrock.load_phantom_dump(ph_file)

cfg = model.gen_config_from_phantom_dump(dump)
cfg.set_boundary_periodic()
cfg.print_status()

model.set_solver_config(cfg)
pmass = 4.0406753833390151E-006 # valid for basic phantom setup. Otherwise read it in th header of the phantom dump file
model.set_particle_mass(pmass)
model.init_scheduler(int(1e5),1)

model.init_from_phantom_dump(dump)
print(ctx.collect_data())
i = 0
i_dump = 0
t_sum = 0
t_target = 4.
current_dt = 0.002 #ev_dic['dt'][0]
while t_sum < t_target:

    next_dt = model.evolve(t_sum,current_dt, True, outputdir + "dump_"+str(i_dump)+".vtk", True)

    if i % 1 == 0:
        i_dump += 1

    t_sum += current_dt
    current_dt = next_dt

    if (t_target - t_sum) < next_dt:
        current_dt = t_target - t_sum

    i +=1


