from statistics import mean, stdev

import shamrock

gamma = 5.0 / 3.0
rho_g = 1
target_tot_u = 1

bmin = (-0.6, -0.6, -0.6)
bmax = (0.6, 0.6, 0.6)

N_target_base = 32e6
compute_multiplier = shamrock.sys.world_size()
# compute_multiplier = 12
scheduler_split_val = int(2e7)
scheduler_merge_val = int(1)


N_target = N_target_base * compute_multiplier
xm, ym, zm = bmin
xM, yM, zM = bmax
vol_b = (xM - xm) * (yM - ym) * (zM - zm)

if shamrock.sys.world_rank() == 0:
    print("N_target_base", N_target_base)
    print("compute_multiplier", compute_multiplier)
    print("scheduler_split_val", scheduler_split_val)
    print("scheduler_merge_val", scheduler_merge_val)
    print("N_target", N_target)
    print("vol_b", vol_b)

part_vol = vol_b / N_target

# lattice volume
part_vol_lattice = 0.74 * part_vol

dr = (part_vol_lattice / ((4.0 / 3.0) * 3.1416)) ** (1.0 / 3.0)

pmass = -1

ctx = shamrock.Context()
ctx.pdata_layout_new()

model = shamrock.get_Model_SPH(context=ctx, vector_type="f64_3", sph_kernel="M4")

cfg = model.gen_default_config()
# cfg.set_artif_viscosity_Constant(alpha_u = 1, alpha_AV = 1, beta_AV = 2)
# cfg.set_artif_viscosity_VaryingMM97(alpha_min = 0.1,alpha_max = 1,sigma_decay = 0.1, alpha_u = 1, beta_AV = 2)
cfg.set_artif_viscosity_VaryingCD10(
    alpha_min=0.0, alpha_max=1, sigma_decay=0.1, alpha_u=1, beta_AV=2
)
cfg.set_boundary_periodic()
cfg.set_eos_adiabatic(gamma)
cfg.set_max_neigh_cache_size(int(100e9))
cfg.print_status()
model.set_solver_config(cfg)
model.init_scheduler(scheduler_split_val, scheduler_merge_val)


bmin, bmax = model.get_ideal_hcp_box(dr, bmin, bmax)
xm, ym, zm = bmin
xM, yM, zM = bmax

model.resize_simulation_box(bmin, bmax)

setup = model.get_setup()
gen = setup.make_generator_lattice_hcp(dr, bmin, bmax)

# On aurora /2 was correct to avoid out of memory
setup.apply_setup(
    gen,
    insert_step=int(scheduler_split_val / 4),
    msg_count_limit=64,
    msg_size_limit=scheduler_split_val /2,
    do_setup_log=True,
)


xc, yc, zc = model.get_closest_part_to((0, 0, 0))

if shamrock.sys.world_rank() == 0:
    print("closest part to (0,0,0) is in :", xc, yc, zc)


vol_b = (xM - xm) * (yM - ym) * (zM - zm)

totmass = rho_g * vol_b
# print("Total mass :", totmass)

pmass = model.total_mass_to_part_mass(totmass)

model.set_value_in_a_box("uint", "f64", 0, bmin, bmax)

rinj = 0.008909042924642563
# rinj = 0.008909042924642563*2*2
# rinj = 0.01718181
u_inj = 1
model.add_kernel_value("uint", "f64", u_inj, (0, 0, 0), rinj)


tot_u = pmass * model.get_sum("uint", "f64")
if shamrock.sys.world_rank() == 0:
    print("total u :", tot_u)


# print("Current part mass :", pmass)

# for it in range(5):
#    setup.update_smoothing_length(ctx)


# print("Current part mass :", pmass)
model.set_particle_mass(pmass)


tot_u = pmass * model.get_sum("uint", "f64")

model.set_cfl_cour(0.1)
model.set_cfl_force(0.1)

model.set_cfl_multipler(1e-4)
model.set_cfl_mult_stiffness(1e6)

# shamrock.dump_profiling("sedov_scale_test_init_" + str(compute_multiplier) + "_")
# shamrock.clear_profiling_data()

res_rates = []
res_cnts = []

for i in range(5):
    model.timestep()

    if i > 0:  # First step make no sense performance wise
        tmp_res_rate, tmp_res_cnt = (
            model.solver_logs_last_rate(),
            model.solver_logs_last_obj_count(),
        )
        res_rates.append(tmp_res_rate)
        res_cnts.append(tmp_res_cnt)

# shamrock.dump_profiling("sedov_scale_test_" + str(compute_multiplier) + "_")
# shamrock.dump_profiling_chrome("sedov_scale_test_chrome_" + str(compute_multiplier) + "_")

res_rate, res_cnt = max(res_rates), res_cnts[0]

if shamrock.sys.world_rank() == 0:
    print(f"res_rates = {res_rates}")
    print(f"res_cnts = {res_cnts}")

    print("--- final score ---")

    print("world size  :", shamrock.sys.world_size())
    print("result rate :", res_rate)
    print("result cnt  :", res_cnt)

    print(
        f"rates infos : max={max(res_rates)}, min={min(res_rates)}, mean={mean(res_rates)}, stddev={stdev(res_rates)}"
    )
