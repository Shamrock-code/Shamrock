import os

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

import shamrock

ctx = shamrock.Context()
ctx.pdata_layout_new()

model = shamrock.get_Model_Ramses(context=ctx, vector_type="f64_3", grid_repr="i64_3")

def run_case(nb_blocks, amr_lev, case="amr"):
    multx = 1
    multy = 1
    multz = 1
    max_amr_lev = amr_lev
    cell_size = 2 << max_amr_lev  # refinement is limited to cell_size = 2
    base = nb_blocks

    cfg = model.gen_default_config()
    scale_fact = 1 / (cell_size * base * multx)
    cfg.set_scale_factor(scale_fact)

    gamma = 1.4
    cfg.set_eos_gamma(gamma)
    cfg.set_Csafe(0.3)
    cfg.set_boundary_condition("x", "reflective")
    cfg.set_boundary_condition("y", "reflective")
    cfg.set_boundary_condition("z", "reflective")
    cfg.set_riemann_solver_hllc()


    cfg.set_slope_lim_minmod()
    cfg.set_face_time_interpolation(True)

    if(case == "amr"):
        err_min = 0.25
        err_max = 0.15
        cfg.set_amr_mode_pseudo_gradient_based(error_min=err_min, error_max=err_max)

    model.set_solver_config(cfg)

    model.init_scheduler(int(1e7), 1)
    model.make_base_grid(
        (0, 0, 0), (cell_size, cell_size, cell_size), (base * multx, base * multy, base * multz)
    )


    def rho_map(rmin, rmax):
        x, y, z = rmin
        if x < 0.5:
            return 1
        else:
            return 0.125


    etot_L = 1.0 / (gamma - 1)
    etot_R = 0.1 / (gamma - 1)


    def rhoetot_map(rmin, rmax):
        x, y, z = rmin
        if x < 0.5:
            return etot_L
        else:
            return etot_R


    def rhovel_map(rmin, rmax):
        return (0, 0, 0)


    model.set_field_value_lambda_f64("rho", rho_map)
    model.set_field_value_lambda_f64("rhoetot", rhoetot_map)
    model.set_field_value_lambda_f64_3("rhovel", rhovel_map)


    def convert_to_cell_coords(dic):

        cmin = dic["cell_min"]
        cmax = dic["cell_max"]

        xmin = []
        ymin = []
        zmin = []
        xmax = []
        ymax = []
        zmax = []

        for i in range(len(cmin)):
            m, M = cmin[i], cmax[i]

            mx, my, mz = m
            Mx, My, Mz = M

            for j in range(8):
                a, b = model.get_cell_coords(((mx, my, mz), (Mx, My, Mz)), j)

                x, y, z = a
                xmin.append(x)
                ymin.append(y)
                zmin.append(z)

                x, y, z = b
                xmax.append(x)
                ymax.append(y)
                zmax.append(z)

        dic["xmin"] = np.array(xmin)
        dic["ymin"] = np.array(ymin)
        dic["zmin"] = np.array(zmin)
        dic["xmax"] = np.array(xmax)
        dic["ymax"] = np.array(ymax)
        dic["zmax"] = np.array(zmax)

        return dic


    t_target = 0.245

    dt = 0
    t = 0
    freq = 100
    dX0 = 0
    for i in range(20000):
        next_dt = model.evolve_once_override_time(t, dt)
        if i == 0:
            dic0 = convert_to_cell_coords(ctx.collect_data())
            dX0 = dic0["xmax"][0] - dic0["xmin"][0]

        t += dt
        dt = next_dt

        if t_target < t + next_dt:
            dt = t_target - t
        if t == t_target:
            break

    xref = 0.5
    xrange = 0.5
    sod = shamrock.phys.SodTube(gamma=gamma, rho_1=1, P_1=1, rho_5=0.125, P_5=0.1)
    sodanalysis = model.make_analysis_sodtube(sod, (1, 0, 0), t_target, xref, 0.0, 1.0)

    #################
    ### Plot
    #################
    # do plot or not
    if True:
        dic = convert_to_cell_coords(ctx.collect_data())

        X = dic["xmin"]
        dX = dic["xmax"] - dic["xmin"]
        rho = dic["rho"]
        rhovel = dic["rhovel"]
        rhovelx = rhovel[:, 0]
        rhovely = rhovel[:, 1]
        rhovelz = rhovel[:, 2]
        rhoetot = dic["rhoetot"]

        u = rhovelx / rho
        v = rhovely / rho
        w = rhovelz / rho
        internal_energy = rhoetot / rho - 0.5 * (u**2 + v**2 + w**2)
        pressure = (rhoetot - 0.5 * rho * (u**2)) * (gamma - 1)

        #### add analytical soluce
        arr_x = np.linspace(xref - xrange, xref + xrange, rho.shape[0])

        arr_rho = []
        arr_P = []
        arr_vx = []

        for i in range(len(arr_x)):
            x_ = arr_x[i] - xref

            _rho, _vx, _P = sod.get_value(t_target, x_)
            arr_rho.append(_rho)
            arr_vx.append(_vx)
            arr_P.append(_P)
        arr_rho = np.array(arr_rho)
        arr_P = np.array(arr_P)
        arr_vx = np.array(arr_vx)

        output = np.column_stack((rho, u, internal_energy, pressure, arr_rho, arr_vx, arr_P))
        np.savetxt(
            f"data_sod_tube_text_{base * 2}.txt",
            output,
            fmt=["%.10f", "%.10f", "%.10f", "%.10f", "%.10f", "%.10f", "%.10f"],
            header="rho u internal_energy press rho_ana, u_ana, press_ana",
        )

        l_0 = np.log2(base * 2)
        l = -np.log2(dX / max(dX0, dX.max())) + l_0

        fig, axs = plt.subplots(2, 2, figsize=(10, 10), dpi=125, constrained_layout=True)

        ## density
        ax00_1 = axs[0, 0]
        if(case == "amr"):
            ax00_2 = ax00_1.twinx()
            ax00_2.set_ylim(np.floor(l.min()), np.ceil(l.max()))  # optional but important
            ax00_2.yaxis.set_major_locator(ticker.MultipleLocator(1))
            ax00_2.yaxis.set_major_formatter(ticker.FormatStrFormatter("%d"))

        ax00_1.scatter(X, rho, rasterized=True, s=12 * np.ones(X.shape), color="red", label="numeric")
        ax00_1.plot(arr_x, arr_rho, ls="--", lw=2.0, color="black", label="exact")
        ax00_1.set_xlabel("$X$")
        ax00_1.set_ylabel("density")
        ax00_1.legend(loc=0)
        ax00_1.grid()

        if(case == "amr"):
            idx = np.argsort(X)
            ax00_2.plot(X[idx], l[idx], color="purple", marker="*", linewidth=1.0, ls="-.")
            ax00_2.set_ylabel("AMR level")
            ax00_2.legend(loc=0)

        ## velocity
        ax01_1 = axs[0, 1]
        if(case=="amr"):
            ax01_2 = ax01_1.twinx()
            ax01_2.set_ylim(np.floor(l.min()), np.ceil(l.max()))  # optional but important
            ax01_2.yaxis.set_major_locator(ticker.MultipleLocator(1))
            ax01_2.yaxis.set_major_formatter(ticker.FormatStrFormatter("%d"))

        ax01_1.scatter(X, u, rasterized=True, s=12 * np.ones(X.shape), color="red", label="numeric")
        ax01_1.plot(arr_x, arr_vx, ls="--", lw=2.0, color="black", label="exact")
        ax01_1.set_xlabel("$X$")
        ax01_1.set_ylabel("velocity")
        ax01_1.legend(loc=0)
        ax01_1.grid()

        if(case=="amr"):
            idx = np.argsort(X)
            ax01_2.plot(X[idx], l[idx], color="purple", marker="*", linewidth=1.0, ls="-.")
            ax01_2.set_ylabel("AMR level")
            ax01_2.legend(loc=0)

        ## pressure
        ax11_1 = axs[1, 1]
        if(case =="amr"):
            ax11_2 = ax11_1.twinx()
            ax11_2.set_ylim(np.floor(l.min()), np.ceil(l.max()))  # optional but important
            ax11_2.yaxis.set_major_locator(ticker.MultipleLocator(1))
            ax11_2.yaxis.set_major_formatter(ticker.FormatStrFormatter("%d"))

        ax11_1.scatter(
            X, pressure, rasterized=True, s=12 * np.ones(X.shape), color="red", label="numeric"
        )
        ax11_1.plot(arr_x, arr_P, ls="--", lw=2.0, color="black", label="exact")
        ax11_1.set_xlabel("$X$")
        ax11_1.set_ylabel("pressure")
        ax11_1.legend(loc=0)
        ax11_1.grid()
        
        if(case == "amr"):
            idx = np.argsort(X)
            ax11_2.plot(X[idx], l[idx], color="purple", marker="*", linewidth=1.0, ls="-.")
            ax11_2.set_ylabel("AMR level")
            ax11_2.legend(loc=0)

        ## internal energy
        ax10_1 = axs[1, 0]
        if(case=="amr"):
            ax10_2 = ax10_1.twinx()
            ax10_2.set_ylim(np.floor(l.min()), np.ceil(l.max()))  # optional but important
            ax10_2.yaxis.set_major_locator(ticker.MultipleLocator(1))
            ax10_2.yaxis.set_major_formatter(ticker.FormatStrFormatter("%d"))

        ax10_1.scatter(
            X, internal_energy, rasterized=True, s=12 * np.ones(X.shape), color="red", label="numeric"
        )
        ax10_1.plot(
            arr_x, arr_P / (arr_rho * (gamma - 1.0)), ls="--", lw=2.0, color="black", label="exact"
        )
        ax10_1.set_xlabel("$X$")
        ax10_1.set_ylabel("internal energy")
        ax10_1.legend(loc=0)
        ax10_1.grid()

        if(case=="amr"):
            idx = np.argsort(X)
            ax10_2.plot(X[idx], l[idx], color="purple", marker="*", linewidth=1.0, ls="-.")
            ax10_2.set_ylabel("AMR level")
            ax10_2.legend(loc=0)

        plt.savefig(f"sod-{case}.pdf")


run_case(16, 3)
