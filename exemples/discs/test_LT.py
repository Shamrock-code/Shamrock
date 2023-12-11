import shamrock
import matplotlib.pyplot as plt
import numpy as np

outputdir = "/home/ylapeyre/Shamrock_tests/new_tilt/"
ph_dir = "/home/ylapeyre/phantom_tests/warp_newtilt/"
ph_file = ph_dir + "warp_00000"

ctx = shamrock.Context()
ctx.pdata_layout_new()


model = shamrock.get_SPHModel(context = ctx, vector_type = "f64_3",sph_kernel = "M4")
dump = shamrock.load_phantom_dump(ph_file)

cfg = model.gen_config_from_phantom_dump(dump, bypass_error=True)
cfg.set_eos_locally_isothermal() # since ont the same ID for ios in sham and phan


cfg.print_status()
model.set_solver_config(cfg)

model.init_scheduler(int(1e7),1)

model.init_from_phantom_dump(dump)
print(ctx.collect_data())


print("Run")

t_sum = 0
t_target = 100
current_dt = 1e-7
i = 0
i_dump = 0
while t_sum < t_target:

    print("step : t=",t_sum)

    do_dump = (i % 50 == 0)  
    #next_dt = model.evolve(t_sum,current_dt, do_dump, outputdir + "dump_"+str(i_dump)+".vtk", do_dump)
    next_dt = model.evolve(t_sum,current_dt, False, outputdir + "dump_{:04}.vtk".format(i_dump), False)

    if do_dump:
        dump = model.make_phantom_dump()
        fname = outputdir + "/test_LT_{:04}".format(i_dump)
        dump.save_dump(fname)
        print("saving t=",t_sum+current_dt,fname)

    if i % 50 == 0:
        i_dump += 1

    t_sum += current_dt
    current_dt = next_dt

    if (t_target - t_sum) < next_dt:
        current_dt = t_target - t_sum

    i+= 1
