import matplotlib.pyplot as plt
import numpy as np

import shamrock

################### PARAMETERS ###################
outputdir = ""

# simulation parameters
tmax = 0.1
dt_dump = 0.01
C_cour = 0.08
C_force = 0.08

# physics parameters
Rstart = 0.3
gamma = 5.0 / 3.0
bmin = 0.0
bmax = 1.0

# grid parameters
base = 8  # resol = base * 2
multx = 1
multy = 1
multz = 1
sz = 1 << 1  # size of the cell
scale_fact = 1 / (sz * base * multx)

center = (base * scale_fact, base * scale_fact, base * scale_fact)
xc, yc, zc = center

si = shamrock.UnitSystem()
sicte = shamrock.Constants(si)
codeu = shamrock.UnitSystem(
    unit_time=3600 * 24 * 365,
    unit_length=sicte.au(),
    unit_mass=sicte.sol_mass(),
)
ucte = shamrock.Constants(codeu)


################# DUMP HANDLING ##################
def get_last_dump():
    import glob

    res = glob.glob(outputdir + "sedov_" + "*.sham")

    f_max = ""
    num_max = -1

    for f in res:
        try:
            dump_num = int(f[len("sedov_") : -5])
            if dump_num > num_max:
                f_max = f
                num_max = dump_num
        except:
            pass

    if num_max == -1:
        return None
    else:
        return num_max


idump_last_dump = get_last_dump()


##################### SETUP ######################
def rho_map(rmin, rmax):
    return 1.0


def rhoe_map(rmin, rmax):
    x, y, z = rmin
    x = x - xc
    y = y - yc
    z = z - zc
    r = np.sqrt(x * x + y * y + z * z)
    if r < Rstart:
        return 10.0 / (gamma - 1.0)
    else:
        return 0.01 / (gamma - 1.0)


def rhovel_map(rmin, rmax):
    return (0.0, 0.0, 0.0)


ctx = shamrock.Context()
ctx.pdata_layout_new()
model = shamrock.get_Model_Ramses(context=ctx, vector_type="f64_3", grid_repr="i64_3")

if idump_last_dump is not None:
    model.load_from_dump(outputdir + "sedov_" + f"{idump_last_dump:04}" + ".sham")
    idump = idump_last_dump + 1  # avoid overwriting your start dump !
else:
    idump = 0

    cfg = model.gen_default_config()
    cfg.set_scale_factor(scale_fact)
    cfg.set_eos_gamma(gamma)
    cfg.set_riemann_solver_hllc()
    cfg.set_slope_lim_vanleer_sym()
    cfg.set_face_time_interpolation(True)
    cfg.set_Csafe(C_cour)
    model.set_solver_config(cfg)

    model.init_scheduler(int(1e7), 1)
    model.make_base_grid((0, 0, 0), (sz, sz, sz), (base * multx, base * multy, base * multz))

    model.set_field_value_lambda_f64("rho", rho_map)
    model.set_field_value_lambda_f64("rhoetot", rhoe_map)
    model.set_field_value_lambda_f64_3("rhovel", rhovel_map)


################## SIMULATION ####################
t = 0
while t <= tmax:
    # model.dump_vtk(outputdir + "sedov_" + f"{idump:04}" + ".vtk")
    model.dump(outputdir + "sedov_" + f"{idump:04}" + ".sham")
    model.evolve_until(t)
    t += dt_dump
    idump += 1
