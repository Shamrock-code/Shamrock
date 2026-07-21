"""
Testing Dusty TVA Sod tube with SPH
===================================

CI test for Dusty TVA Sod tube with SPH
"""

import json
import os

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import PillowWriter
from shamrock.utils.analysis import AnalysisHelper
from shamrock.utils.plot import show_image_sequence
from shamrock.utils.SimulationRunner import SimulationRunner, callback, simulation_setup

import shamrock

# If we use the shamrock executable to run this script instead of the python interpreter,
# we should not initialize the system as the shamrock executable needs to handle specific MPI logic
if not shamrock.sys.is_initialized():
    shamrock.change_loglevel(1)
    shamrock.sys.init("0:0")

# %%
# Use shamrock documentation style for matplotlib
shamrock.matplotlib.set_shamrock_mpl_style()

# %%

gamma = 1.4

rho_g = 1
rho_d = 0.125

epsilons = [0.5]
ts = [0.001]
ndust = len(epsilons)

eps_all = np.sum(epsilons)
gas_fact = 1 - eps_all

fact = (rho_g / rho_d) ** (1.0 / 3.0)

P_g = 1
P_d = 0.1

u_g = P_g / ((gamma - 1) * rho_g)
u_d = P_d / ((gamma - 1) * rho_d)

resol = int(os.environ.get("RESOL", 128))

sim_folder = f"_to_trash/dustysod_{resol}/"
dump_folder = sim_folder + "dump/"

shamrock.enable_experimental_features()

# %%

ctx = shamrock.Context()
ctx.pdata_layout_new()

model = shamrock.get_Model_SPH(context=ctx, vector_type="f64_3", sph_kernel="M6")

l2_error_analysis = AnalysisHelper(
    analysis_folder=sim_folder,
    analysis_prefix="l2_error",
)

dust_mass_analysis = AnalysisHelper(
    analysis_folder=sim_folder,
    analysis_prefix="dust_mass",
)


# %%


class Simulation(SimulationRunner):
    # Use the global vars defined at the top of the file
    t_end = 0.245
    dump_prefix = dump_folder + "dump_"

    @callback(at_tsim=(0.245 * np.linspace(0.0, 1.0, 10)).tolist())  # Do the analysis every dt_stop
    def analysis_plots(self, ianalysis):
        dic = ctx.collect_data()
        pmass = model.get_particle_mass()

        x = np.array(dic["xyz"][:, 0]) + 0.5
        vx = dic["vxyz"][:, 0]
        uint_tilde = dic["uint"][:]
        sj = dic["s_j"].reshape(-1, ndust)

        hpart = dic["hpart"]
        alpha = dic["alpha_AV"]

        rho = pmass * (model.get_hfact() / hpart) ** 3

        rho_d = sj**2
        rho_g = rho - np.sum(rho_d, axis=1)

        P = (gamma - 1) * rho * uint_tilde  # = rho_g * u

        sod = shamrock.phys.SodTube(gamma=gamma, rho_1=1, P_1=1, rho_5=0.125, P_5=0.1)
        sodanalysis = model.make_analysis_sodtube(
            sod, (1, 0, 0), self.model.get_time(), 0.0, -0.5, 0.5
        )
        l2_rho, l2_v, l2_P = sodanalysis.compute_L2_dist()
        l2_error_analysis.analysis_save(
            ianalysis, {"l2_rho": l2_rho, "l2_v": l2_v, "l2_P": l2_P, "time": self.model.get_time()}
        )

        dmass_a = shamrock.model_sph.analysisDustMass(model=model)
        dmass = dmass_a.get_dust_mass()
        dust_mass_analysis.analysis_save(
            ianalysis, {"dust_mass": dmass, "time": self.model.get_time()}
        )

        plt.plot(x, rho, ".", label="rho_g")
        plt.plot(x, vx, ".", label="v")
        plt.plot(x, P, ".", label="P")
        plt.plot(x, alpha, ".", label="alpha")

        for i in range(ndust):
            plt.plot(x, rho_d[:, i], ".", label=f"rho_d_{i}")

        x = np.linspace(-0.5, 0.5, 1000)

        rho = []
        P = []
        vx = []

        for i in range(len(x)):
            x_ = x[i]

            _rho, _vx, _P = sod.get_value(self.model.get_time(), x_)
            rho.append(_rho)
            vx.append(_vx)
            P.append(_P)

        x += 0.5
        plt.plot(x, rho, color="black", label="analytic")
        plt.plot(x, vx, color="black")
        plt.plot(x, P, color="black")

        plt.legend()
        plt.grid()
        plt.ylim(0, 1.1)
        plt.xlim(0, 1)
        plt.xlabel("x")
        plt.title(f"t={self.model.get_time():.3f}")
        plt.savefig(dump_folder + f"sod_{ianalysis:04d}.png")
        plt.close()

    @callback(walltime_interval=30.0)  # Checkpoint the simulation every 30 seconds
    def checkpoint(self, icheckpoint):
        self.do_checkpoint(icheckpoint, purge_old_dumps=True, keep_first=1, keep_last=3)

    @simulation_setup
    def setup(self):

        cfg = model.gen_default_config()
        cfg.set_artif_viscosity_VaryingCD10(
            alpha_min=0.0, alpha_max=1, sigma_decay=0.1, alpha_u=1, beta_AV=2
        )
        cfg.set_boundary_periodic()
        cfg.set_eos_adiabatic(gamma)
        cfg.set_dust_mode_monofluid_tva(nvar=1)
        cfg.set_dust_drag_constant(ts)
        cfg.set_show_cfl_detail(True)
        cfg.print_status()
        model.set_solver_config(cfg)

        model.init_scheduler(int(1e8), 1)

        (xs, ys, zs) = model.get_box_dim_fcc_3d(1, resol, 24, 24)
        dr = 1 / xs
        (xs, ys, zs) = model.get_box_dim_fcc_3d(dr, resol, 24, 24)

        model.resize_simulation_box((-xs, -ys / 2, -zs / 2), (xs, ys / 2, zs / 2))

        V_g_min = (-xs, -ys / 2, -zs / 2)
        V_g_max = (0, ys / 2, zs / 2)
        V_d_min = (0, -ys / 2, -zs / 2)
        V_d_max = (xs, ys / 2, zs / 2)

        setup = model.get_setup()
        gen1 = setup.make_generator_lattice_hcp(dr, V_g_min, V_g_max)
        gen2 = setup.make_generator_lattice_hcp(dr * fact, V_d_min, V_d_max)
        comb = setup.make_combiner_add(gen1, gen2)
        # print(comb.get_dot())
        setup.apply_setup(comb)

        # model.add_cube_fcc_3d(dr, (-xs,-ys/2,-zs/2),(0,ys/2,zs/2))
        # model.add_cube_fcc_3d(dr*fact, (0,-ys/2,-zs/2),(xs,ys/2,zs/2))

        model.set_value_in_a_box("uint", "f64", u_g, V_g_min, V_g_max)
        model.set_value_in_a_box("uint", "f64", u_d, V_d_min, V_d_max)

        for i in range(ndust):
            epsilon = epsilons[i]

            s_g = np.sqrt(rho_g * epsilon)
            s_d = np.sqrt(rho_d * epsilon)

            model.set_value_in_a_box("s_j", "f64", s_g, V_g_min, V_g_max, ivar=i)
            model.set_value_in_a_box("s_j", "f64", s_d, V_d_min, V_d_max, ivar=i)

        vol_b = xs * ys * zs

        totmass = (rho_d * vol_b) + (rho_g * vol_b)

        print("Total mass :", totmass)

        pmass = model.total_mass_to_part_mass(totmass)
        model.set_particle_mass(pmass)
        print("Current part mass :", pmass)

        model.set_cfl_cour(0.1)
        model.set_cfl_force(0.1)

        model.timestep()


Simulation(model).run()

# %%

glob_str = f"{dump_folder}sod_*.png"
ani = show_image_sequence(glob_str)

writer = PillowWriter(fps=15, metadata=dict(artist="Me"), bitrate=1800)
ani.save("_to_trash/sod.gif", writer=writer)

if shamrock.sys.world_rank() == 0:
    plt.show()

# %%


t_l2, l2_rho, l2_v, l2_P = [], [], [], []
for i in l2_error_analysis.get_list_analysis_id():
    data = l2_error_analysis.load_analysis(i).item()
    t_l2.append(data["time"])
    l2_rho.append(data["l2_rho"])
    l2_v.append(data["l2_v"])
    l2_P.append(data["l2_P"])

l2_v = np.array(l2_v)

plt.plot(t_l2, l2_rho, label="l2_rho")
plt.plot(t_l2, l2_v[:, 0], label="l2_v (vx)")
plt.plot(t_l2, l2_v[:, 1], label="l2_v (vy)")
plt.plot(t_l2, l2_v[:, 2], label="l2_v (vz)")
plt.plot(t_l2, l2_P, label="l2_P")
plt.legend()
plt.xlabel("t")
plt.ylabel("L2 error")
plt.yscale("log")
plt.tight_layout()
plt.savefig(f"{dump_folder}/l2_error.png")
plt.show()


# %%

t_dmass, dmass = [], []
for i in dust_mass_analysis.get_list_analysis_id():
    data = dust_mass_analysis.load_analysis(i).item()
    t_dmass.append(data["time"])
    dmass.append(data["dust_mass"])

dmass = np.array(dmass)[:, 0]


plt.plot(t_dmass, (dmass - dmass[0]) / dmass[0], label="dust_mass")
plt.legend()
plt.xlabel("t")
plt.ylabel("Dust mass deviation")
plt.yscale("symlog", linthresh=1e-8)
plt.tight_layout()
plt.savefig(f"{dump_folder}/dust_mass.png")
plt.show()

# %%

print("##### Test result #####")
result = {
    "t": float(t_l2[-1]),
    "l2_rho": l2_rho[-1],
    "l2_v": l2_v[-1].tolist(),
    "l2_P": l2_P[-1],
    "dust_mass_conservation": (dmass[-1] - dmass[0]) / dmass[0],
    "resol": resol,
}
print(result)

json.dump(result, open(f"{dump_folder}/test_result.json", "w"), indent=4)
