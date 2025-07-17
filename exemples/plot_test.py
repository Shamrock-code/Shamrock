import matplotlib.pyplot as plt
import numpy as np

import shamrock

####################################################
# Setup parameters
####################################################
Npart = 100000
disc_mass = 0.01  # sol mass
center_mass = 1
center_racc = 0.05

rout = 10
rin = 0.1

# alpha_ss ~ alpha_AV * 0.08
alpha_AV = 1e-3 / 0.08
alpha_u = 1
beta_AV = 2

q = 0.5
p = 3.0 / 2.0
r0 = 1
Rcav = 2.5
delta_0 = 1e-5

C_cour = 0.3
C_force = 0.25

H_r_in = 0.05

do_plots = True

dump_prefix = "santa_barbara_1M_"

R_planet_base = 3  # multiplier for the planet orbit radius
planet_list = [
    # {"R": R_planet_base*1, "mass": 1e-3},
    # reso 3:2
    # {"R": R_planet_base*(3/2)**(2/3), "mass": 1e-3},
    # reso 2:1
    # {"R": R_planet_base*(2/1)**(2/3), "mass": 1e-3},
]

racc_overhill = 0.1
for i in range(len(planet_list)):
    planet_list[i]["racc"] = racc_overhill * shamrock.phys.hill_radius(
        R=planet_list[i]["R"], m=planet_list[i]["mass"], M=center_mass
    )

center_object_is_binary = True
m1_over_centermass = 0.5
m2_over_centermass = 0.5
binary_a = 1
binary_e = 0

freq_stop = 1000.0
dt_stop = (1.0 / freq_stop) * 2 * np.pi
norbit = 10
nstop = norbit * freq_stop
t_end = nstop * dt_stop

####################################################
####################################################
####################################################

si = shamrock.UnitSystem()
sicte = shamrock.Constants(si)
codeu = shamrock.UnitSystem(
    unit_time=3600 * 24 * 365,
    unit_length=sicte.au() * (39.42410494106729 ** (1 / 3)),
    unit_mass=sicte.sol_mass(),
)
ucte = shamrock.Constants(codeu)

# Deduced quantities
pmass = disc_mass / Npart
bmin = (-rout * 2, -rout * 2, -rout * 2)
bmax = (rout * 2, rout * 2, rout * 2)
G = ucte.G()

print("GM =", G * center_mass)


def sigma_profile(r):
    sigma_0 = 1
    return sigma_0 * ((1 - delta_0) * np.exp(-((Rcav / r) ** 12)) + delta_0)


def kep_profile(r):
    return (G * center_mass / r) ** 0.5


def omega_k(r):
    return kep_profile(r) / r


def cs_profile(r):
    cs_in = (H_r_in * rin) * omega_k(rin)
    return ((r / rin) ** (-q)) * cs_in


cs0 = cs_profile(rin)


def rot_profile(r):
    # return kep_profile(r)

    # subkeplerian correction
    return ((kep_profile(r) ** 2) - (2 * p + q) * cs_profile(r) ** 2) ** 0.5


def H_profile(r):
    H = cs_profile(r) / omega_k(r)
    # fact = (2.**0.5) * 3.
    fact = 1
    return fact * H  # factor taken from phantom, to fasten thermalizing


def plot_curve_in():
    x = np.linspace(rin, rout)
    sigma = []
    kep = []
    cs = []
    rot = []
    H = []
    H_r = []
    for i in range(len(x)):
        _x = x[i]

        sigma.append(sigma_profile(_x))
        kep.append(kep_profile(_x))
        cs.append(cs_profile(_x))
        rot.append(rot_profile(_x))
        H.append(H_profile(_x))
        H_r.append(H_profile(_x) / _x)

    plt.plot(x, sigma, label="sigma")
    plt.plot(x, kep, label="keplerian speed")
    plt.plot(x, cs, label="cs")
    plt.plot(x, rot, label="rot speed")
    plt.plot(x, H, label="H")
    plt.plot(x, H_r, label="H_r")


if do_plots and False:
    plot_curve_in()
    plt.legend()
    plt.yscale("log")
    # plt.xscale("log")
    plt.show()


####################################################
# Binary coordinate mapping
####################################################
def kepler_to_cartesian_no_rotation(m1, m2, a, e):
    nu = np.radians(0)
    # Total mass and reduced mass
    M = m1 + m2
    mu = m1 * m2 / M

    # Distance between the two stars
    r = a * (1 - e**2) / (1 + e * np.cos(nu))

    # Orbital positions in the orbital plane
    x_orb = r * np.cos(nu)
    y_orb = r * np.sin(nu)

    # Orbital velocities in the orbital plane
    h = np.sqrt(G * M * a * (1 - e**2))
    vx_orb = -G * M / h * np.sin(nu)
    vy_orb = G * M / h * (e + np.cos(nu))

    # Position in 2D orbital plane
    r_orb = np.array([x_orb, y_orb])

    # Velocity in 2D orbital plane
    v_orb = np.array([vx_orb, vy_orb])

    # Center of mass positions
    r1 = -m2 / M * r_orb
    r2 = m1 / M * r_orb

    # Center of mass velocities
    v1 = -m2 / M * v_orb
    v2 = m1 / M * v_orb

    return r1, r2, v1, v2


def split_as_binary(sink, m1, m2, a, e):
    r1, r2, v1, v2 = kepler_to_cartesian_no_rotation(m1, m2, a, e)

    s1 = sink.copy()
    print(s1)
    s1["mass"] = m1
    s1["pos"] = r1 + sink.pos
    s1["vel"] = v1 + sink.vel

    s2 = sink.copy()
    print(s2)
    s2["mass"] = m2
    s2["pos"] = r2 + sink.pos
    s2["vel"] = v2 + sink.vel

    return s1, s2


####################################################
# Dump handling
####################################################
def get_dump_name(idump):
    return dump_prefix + f"{idump:04}" + ".sham"


def get_vtk_dump_name(idump):
    return dump_prefix + f"{idump:04}" + ".vtk"


def get_last_dump():
    import glob

    res = glob.glob(dump_prefix + "*.sham")

    f_max = ""
    num_max = -1

    for f in res:
        try:
            dump_num = int(f[len(dump_prefix) : -5])
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

####################################################
# Plot predicted orbits
####################################################


def plot_sim_orbit(sink_list):

    sink_cnt = len(sink_list)

    X = [[] for i in range(sink_cnt)]
    Y = [[] for i in range(sink_cnt)]
    Z = [[] for i in range(sink_cnt)]
    VX = [[] for i in range(sink_cnt)]
    VY = [[] for i in range(sink_cnt)]
    VZ = [[] for i in range(sink_cnt)]

    for i in range(sink_cnt):
        x, y, z = sink_list[i]["pos"]
        vx, vy, vz = sink_list[i]["vel"]

        X[i].append(x)
        Y[i].append(y)
        Z[i].append(z)
        VX[i].append(vx)
        VY[i].append(vy)
        VZ[i].append(vz)

    dt = 0.00001
    for it in range(int(50000 * 2 * np.pi)):
        t = dt * i
        for i in range(sink_cnt):
            xi, yi, zi = X[i][-1], Y[i][-1], Z[i][-1]
            vxi, vyi, vzi = VX[i][-1], VY[i][-1], VZ[i][-1]
            fx, fy, fz = 0.0, 0.0, 0.0
            for j in range(sink_cnt):
                if i == j:
                    continue

                xj, yj, zj = X[j][-1], Y[j][-1], Z[j][-1]
                vxj, vyj, vzj = VX[j][-1], VY[j][-1], VZ[j][-1]

                d = ((xi - xj) ** 2 + (yi - yj) ** 2 + (zi - zj) ** 2) ** 0.5
                dx = xi - xj
                dy = yi - yj
                dz = zi - zj

                fnorm = -G * sink_list[j]["mass"] / (d**3)
                fx += fnorm * dx
                fy += fnorm * dy
                fz += fnorm * dz
            xi += dt * vxi
            yi += dt * vyi
            zi += dt * vzi

            vxi += dt * fx
            vyi += dt * fy
            vzi += dt * fz

            X[i].append(xi)
            Y[i].append(yi)
            Z[i].append(zi)
            VX[i].append(vxi)
            VY[i].append(vyi)
            VZ[i].append(vzi)

    for i in range(sink_cnt):
        plt.plot(X[i], Y[i])
    plt.axis("equal")
    plt.grid()
    plt.show()


def plot_sim_orbit2(sink_list):

    import matplotlib.pyplot as plt
    import numpy as np
    from scipy.integrate import odeint

    # Define the N-body problem
    def nbody(y, t, N, masses):
        # y: array containing position and velocity of all bodies, shape: (N, 6)
        # t: time
        # N: number of bodies
        # masses: array of masses of the bodies

        # Unpack the positions and velocities from y
        pos = y[: N * 3].reshape((N, 3))  # Positions, shape (N, 3)
        vel = y[N * 3 :].reshape((N, 3))  # Velocities, shape (N, 3)

        # Initialize the derivatives of position and velocity
        dydt = np.zeros_like(y)

        # Compute the forces
        forces = np.zeros_like(pos)
        for i in range(N):
            for j in range(i + 1, N):
                # Vector between body i and body j
                r = pos[j] - pos[i]
                dist = np.linalg.norm(r)
                force_magnitude = G * masses[i] * masses[j] / dist**2
                force = force_magnitude * r / dist
                forces[i] += force
                forces[j] -= force  # Action and reaction

        # Derivative of position is the velocity
        dydt[: N * 3] = vel.flatten()

        # Derivative of velocity is the acceleration
        dydt[N * 3 :] = (forces / masses[:, np.newaxis]).flatten()

        return dydt

    # Initial conditions
    N = len(sink_list)  # Number of bodies
    masses = np.array([s["mass"] for s in sink_list])  # Masses of the bodies (kg)
    positions = np.array([s["pos"] for s in sink_list])  # Initial positions (m)
    velocities = np.array([s["vel"] for s in sink_list])  # Initial velocities (m/s)

    print(f"masses = {masses}")
    print(f"positions = {positions}")
    print(f"velocities = {velocities}")

    # Combine initial conditions into a single array
    y0 = np.hstack((positions.flatten(), velocities.flatten()))

    # Time grid for integration
    t = np.linspace(0, 6.12, 1000)  # Time from 0 to 1e5 seconds

    # Integrate the equations of motion
    solution = odeint(nbody, y0, t, args=(N, masses))

    # Extract positions from the solution
    positions = solution[:, : N * 3].reshape((-1, N, 3))

    # Plot the orbits
    plt.figure(figsize=(8, 8))
    for i in range(N):
        plt.plot(positions[:, i, 0], positions[:, i, 1], label=f"Body {i+1}")

    plt.xlabel("x")
    plt.ylabel("y")
    plt.legend()
    plt.grid(True)
    plt.show()


####################################################
####################################################
####################################################

ctx = shamrock.Context()
ctx.pdata_layout_new()

model = shamrock.get_Model_SPH(context=ctx, vector_type="f64_3", sph_kernel="M4")


def add_callbacks_to_config(cfg):
    cfg.register_callback(name="callback_dump", freq_step=10)
    cfg.register_callback(name="callback_plot", freq_time=dt_stop)


run_post_setup = False

if idump_last_dump is not None:
    model.load_from_dump(get_dump_name(idump_last_dump))
else:
    cfg = model.gen_default_config()
    # cfg.set_artif_viscosity_VaryingMM97(alpha_min = 0.1,alpha_max = 1,sigma_decay = 0.1, alpha_u = 1, beta_AV = 2)
    cfg.set_artif_viscosity_ConstantDisc(alpha_u=alpha_u, alpha_AV=alpha_AV, beta_AV=beta_AV)
    cfg.set_eos_locally_isothermalLP07(cs0=cs0, q=q, r0=r0)
    cfg.print_status()
    cfg.set_units(codeu)
    add_callbacks_to_config(cfg)
    model.set_solver_config(cfg)

    model.init_scheduler(int(8e5), 1)

    model.resize_simulation_box(bmin, bmax)

    sink_list = []

    if center_object_is_binary:
        m1 = m1_over_centermass * center_mass
        m2 = m2_over_centermass * center_mass

        r1, r2, v1, v2 = kepler_to_cartesian_no_rotation(m1, m2, binary_a, binary_e)

        p1 = (r1[0], r1[1], 0)
        v1 = (v1[0], v1[1], 0)

        p2 = (r2[0], r2[1], 0)
        v2 = (v2[0], v2[1], 0)
        sink_list = [
            {"mass": m1, "racc": center_racc, "pos": p1, "vel": v1},
            {"mass": m2, "racc": center_racc, "pos": p2, "vel": v2},
        ]
    else:
        sink_list = [
            {"mass": center_mass, "racc": center_racc, "pos": (0, 0, 0), "vel": (0, 0, 0)},
        ]

    for _p in planet_list:
        mass = _p["mass"]
        R = _p["R"]
        racc = _p["racc"]

        vphi = kep_profile(R)
        pos = (R, 0, 0)
        vel = (0, vphi, 0)

        sink_list.append({"mass": mass, "racc": racc, "pos": pos, "vel": vel})

    sum_mass = sum(s["mass"] for s in sink_list)
    vel_bary = (
        sum(s["mass"] * s["vel"][0] for s in sink_list) / sum_mass,
        sum(s["mass"] * s["vel"][1] for s in sink_list) / sum_mass,
        sum(s["mass"] * s["vel"][2] for s in sink_list) / sum_mass,
    )
    pos_bary = (
        sum(s["mass"] * s["pos"][0] for s in sink_list) / sum_mass,
        sum(s["mass"] * s["pos"][1] for s in sink_list) / sum_mass,
        sum(s["mass"] * s["pos"][2] for s in sink_list) / sum_mass,
    )

    print("sinks baryenceter : velocity {} position {}".format(vel_bary, pos_bary))

    # plot_sim_orbit(sink_list)
    # plot_sim_orbit2(sink_list)

    model.set_particle_mass(pmass)
    for s in sink_list:
        mass = s["mass"]
        x, y, z = s["pos"]
        vx, vy, vz = s["vel"]
        racc = s["racc"]

        x -= pos_bary[0]
        y -= pos_bary[1]
        z -= pos_bary[2]

        vx -= vel_bary[0]
        vy -= vel_bary[1]
        vz -= vel_bary[2]

        print(
            "add sink : mass {} pos {} vel {} racc {}".format(mass, (x, y, z), (vx, vy, vz), racc)
        )
        model.add_sink(mass, (x, y, z), (vx, vy, vz), racc)

    setup = model.get_setup()
    gen_disc = setup.make_generator_disc_mc(
        part_mass=pmass,
        disc_mass=disc_mass,
        r_in=rin,
        r_out=rout,
        sigma_profile=sigma_profile,
        H_profile=H_profile,
        rot_profile=rot_profile,
        cs_profile=cs_profile,
        random_seed=666,
    )
    # print(comb.get_dot())
    setup.apply_setup(gen_disc)

    model.set_cfl_cour(C_cour)
    model.set_cfl_force(C_force)

    run_post_setup = True


def plot_rho(ext, sinks, arr_rho, iplot):

    import matplotlib

    # Reset the figure using the same memory as the last one
    plt.figure(num=1, clear=True, dpi=200)
    import copy

    my_cmap = copy.copy(matplotlib.colormaps.get_cmap("gist_heat"))  # copy the default cmap
    my_cmap.set_bad(color="black")

    res = plt.imshow(
        arr_rho,
        cmap=my_cmap,
        origin="lower",
        extent=[-ext, ext, -ext, ext],
        norm="log",
        vmin=1e-8,
        vmax=2e-4,
    )
    # res = plt.imshow(arr[:,:,0], cmap="seismic",origin='lower', extent=[-ext, ext, -ext, ext])
    # plt.scatter(sinks[:,0],sinks[:,1], s=1)

    ax = plt.gca()

    output_list = []
    for s in sinks:
        # print(s)
        x, y, z = s["pos"]
        output_list.append(plt.Circle((x, y), s["accretion_radius"], color="blue", fill=False))
    for circle in output_list:
        ax.add_artist(circle)

    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("t = {:0.3f} [Binary orbit]".format(model.get_time() / (2 * np.pi)))

    cbar = plt.colorbar(res, extend="both")
    cbar.set_label(r"$\rho$ [code unit]")

    plt.savefig("plot_rho_{:04}.png".format(iplot))


def plot_rho_integ(ext, sinks, arr_rho, iplot):

    dpi = 200
    import matplotlib

    # Reset the figure using the same memory as the last one
    plt.figure(num=1, clear=True, dpi=dpi)
    import copy

    my_cmap = copy.copy(matplotlib.colormaps.get_cmap("gist_heat"))  # copy the default cmap
    my_cmap.set_bad(color="black")

    res = plt.imshow(
        arr_rho,
        cmap=my_cmap,
        origin="lower",
        extent=[-ext, ext, -ext, ext],
        norm="log",
        vmin=1e-8,
        vmax=1e-4,
    )
    # res = plt.imshow(arr_rho, cmap=my_cmap,origin='lower', extent=[-ext, ext, -ext, ext])
    # plt.scatter(sinks[:,0],sinks[:,1], s=1)

    ax = plt.gca()

    output_list = []
    for s in sinks:
        # print(s)
        x, y, z = s["pos"]
        output_list.append(plt.Circle((x, y), s["accretion_radius"], color="blue", fill=False))
    for circle in output_list:
        ax.add_artist(circle)

    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("t = {:0.3f} [Binary orbit]".format(model.get_time() / (2 * np.pi)))

    center_cmap_x = ext * 0.75
    center_cmap_y = 0
    cmap_width = ext * 0.125 / 2
    cmap_height = ext * 1.8

    cbar = plt.colorbar(res, extend="both")
    cbar.set_label(r"$\int \rho \, \mathrm{d}z$ [code unit]")

    plt.savefig("plot_rho_integ_{:04}.png".format(iplot))


def rot_plot_rho(ext, sinks, arr_rho, iplot, e_r, e_theta):

    import matplotlib

    # Reset the figure using the same memory as the last one
    plt.figure(num=1, clear=True, dpi=200)
    import copy

    my_cmap = copy.copy(matplotlib.colormaps.get_cmap("gist_heat"))  # copy the default cmap
    my_cmap.set_bad(color="black")

    res = plt.imshow(
        arr_rho,
        cmap=my_cmap,
        origin="lower",
        extent=[-ext, ext, -ext, ext],
        norm="log",
        vmin=1e-8,
        vmax=2e-4,
    )
    # res = plt.imshow(arr[:,:,0], cmap="seismic",origin='lower', extent=[-ext, ext, -ext, ext])
    # plt.scatter(sinks[:,0],sinks[:,1], s=1)

    ax = plt.gca()

    output_list = []
    for s in sinks:
        # print(s)
        x, y, z = s["pos"]
        e_rx, e_ry, e_rz = e_r
        e_thx, e_thy, e_thz = e_theta

        p_x = x * e_rx + y * e_ry
        p_y = x * e_thx + y * e_thy

        output_list.append(
            plt.Circle(
                (p_x / (2 * ext), p_y / (2 * ext)), s["accretion_radius"], color="blue", fill=False
            )
        )
    for circle in output_list:
        ax.add_artist(circle)

    plt.xlabel(r"$x_{binary}$")
    plt.ylabel(r"$y_{binary}$")
    plt.title("t = {:0.3f} [Binary orbit]".format(model.get_time() / (2 * np.pi)))

    cbar = plt.colorbar(res, extend="both")
    cbar.set_label(r"$\rho$ [code unit]")

    plt.savefig("plot_rho_rot_{:04}.png".format(iplot))


def plot_vx(ext, sinks, arr_vx, iplot):

    import matplotlib

    # Reset the figure using the same memory as the last one
    plt.figure(num=1, clear=True, dpi=200)

    v_ext = np.max(arr_vx)
    v_ext = max(v_ext, np.abs(np.min(arr_vx)))
    res = plt.imshow(
        arr_vx,
        cmap="seismic",
        origin="lower",
        extent=[-ext, ext, -ext, ext],
        vmin=-v_ext,
        vmax=v_ext,
    )
    # res = plt.imshow(arr[:,:,0], cmap="seismic",origin='lower', extent=[-ext, ext, -ext, ext])
    # plt.scatter(sinks[:,0],sinks[:,1], s=1)

    ax = plt.gca()

    output_list = []
    for s in sinks:
        # print(s)
        x, y, z = s["pos"]
        output_list.append(plt.Circle((x, y), s["accretion_radius"], color="blue", fill=False))
    for circle in output_list:
        ax.add_artist(circle)

    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("t = {:0.3f} [Binary orbit]".format(model.get_time() / (2 * np.pi)))

    cbar = plt.colorbar(res, extend="both")
    cbar.set_label(r"$v_x$ [code unit]")

    plt.savefig("plot_vx_{:04}.png".format(iplot))


def plot_vz_z(ext, sinks, arr_vz, iplot):

    import matplotlib

    # Reset the figure using the same memory as the last one
    plt.figure(num=1, clear=True, dpi=200)

    v_ext = np.max(arr_vz)
    v_ext = max(v_ext, np.abs(np.min(arr_vz)))
    res = plt.imshow(
        arr_vz,
        cmap="seismic",
        origin="lower",
        extent=[-ext, ext, -ext, ext],
        vmin=-v_ext,
        vmax=v_ext,
    )
    # res = plt.imshow(arr[:,:,0], cmap="seismic",origin='lower', extent=[-ext, ext, -ext, ext])
    # plt.scatter(sinks[:,0],sinks[:,1], s=1)

    ax = plt.gca()

    output_list = []
    for s in sinks:
        # print(s)
        x, y, z = s["pos"]
        output_list.append(plt.Circle((x, z), s["accretion_radius"], color="blue", fill=False))
    for circle in output_list:
        ax.add_artist(circle)

    plt.xlabel("x")
    plt.ylabel("z")
    plt.title("t = {:0.3f} [Binary orbit]".format(model.get_time() / (2 * np.pi)))

    cbar = plt.colorbar(res, extend="both")
    cbar.set_label(r"$v_z$ [code unit]")

    plt.savefig("plot_vz_z_{:04}.png".format(iplot))


def rot_plot_vz_z(ext, sinks, arr_vz, iplot, e_r, e_z):

    import matplotlib

    # Reset the figure using the same memory as the last one
    plt.figure(num=1, clear=True, dpi=200)

    v_ext = np.max(arr_vz)
    v_ext = max(v_ext, np.abs(np.min(arr_vz)))
    res = plt.imshow(
        arr_vz,
        cmap="seismic",
        origin="lower",
        extent=[-ext, ext, -ext, ext],
        vmin=-v_ext,
        vmax=v_ext,
    )
    # res = plt.imshow(arr[:,:,0], cmap="seismic",origin='lower', extent=[-ext, ext, -ext, ext])
    # plt.scatter(sinks[:,0],sinks[:,1], s=1)

    ax = plt.gca()

    output_list = []
    for s in sinks:
        # print(s)
        x, y, z = s["pos"]
        e_rx, e_ry, e_rz = e_r
        e_zx, e_zy, e_zz = e_z

        p_x = x * e_rx + y * e_ry
        p_z = z

        output_list.append(
            plt.Circle((p_x / (2 * ext), p_z), s["accretion_radius"], color="blue", fill=False)
        )
    for circle in output_list:
        ax.add_artist(circle)

    plt.xlabel(r"$x_{binary}$")
    plt.ylabel(r"$z_{binary}$")
    plt.title("t = {:0.3f} [Binary orbit]".format(model.get_time() / (2 * np.pi)))

    cbar = plt.colorbar(res, extend="both")
    cbar.set_label(r"$\rho$ [code unit]")

    plt.savefig("plot_vz_z_rot_{:04}.png".format(iplot))


def plot_state(iplot):
    sinks = model.get_sinks()

    x1, y1, z1 = sinks[0]["pos"]
    x2, y2, z2 = sinks[1]["pos"]

    ext = 5

    d_x = x2 - x1
    d_y = y2 - y1
    d_z = z2 - z1
    d = np.sqrt(d_x**2 + d_y**2 + d_z**2)
    d_x = d_x / d
    d_y = d_y / d
    d_z = d_z / d

    f = 2 * ext
    e_r = (f * d_x, f * d_y, 0)
    e_theta = (-f * d_y, f * d_x, 0)
    e_z = (0, 0, f * 1)

    arr_rho = model.render_cartesian_slice(
        "rho",
        "f64",
        center=(0.0, 0.0, 0.0),
        delta_x=(ext * 2, 0, 0.0),
        delta_y=(0.0, ext * 2, 0.0),
        nx=1000,
        ny=1000,
    )
    arr_rho2 = model.render_cartesian_column_integ(
        "rho",
        "f64",
        center=(0.0, 0.0, 0.0),
        delta_x=(ext * 2, 0, 0.0),
        delta_y=(0.0, ext * 2, 0.0),
        nx=1000,
        ny=1000,
    )
    arr_v = model.render_cartesian_slice(
        "vxyz",
        "f64_3",
        center=(0.0, 0.0, 0.0),
        delta_x=(ext * 2, 0, 0.0),
        delta_y=(0.0, ext * 2, 0.0),
        nx=1000,
        ny=1000,
    )
    arr_v_vslice = model.render_cartesian_slice(
        "vxyz",
        "f64_3",
        center=(0.0, 0.0, 0.0),
        delta_x=(ext * 2, 0, 0.0),
        delta_y=(0.0, 0.0, ext * 2),
        nx=1000,
        ny=1000,
    )
    rot_arr_rho = model.render_cartesian_slice(
        "rho", "f64", center=(0.0, 0.0, 0.0), delta_x=e_r, delta_y=e_theta, nx=1000, ny=1000
    )
    # rot_arr_v = model.render_cartesian_slice("vxyz","f64_3",center = (0.,0.,0.),delta_x = e_r,delta_y = e_theta, nx = 1000, ny = 1000)
    rot_arr_v_vslice = model.render_cartesian_slice(
        "vxyz", "f64_3", center=(0.0, 0.0, 0.0), delta_x=e_r, delta_y=e_z, nx=1000, ny=1000
    )

    if shamrock.sys.world_rank() == 0:

        plot_rho(ext, sinks, arr_rho, iplot)
        plot_rho_integ(ext, sinks, arr_rho2, iplot)
        rot_plot_rho(ext, sinks, rot_arr_rho, iplot, e_r, e_theta)
        plot_vx(ext, sinks, arr_v[:, :, 0], iplot)
        plot_vz_z(ext, sinks, arr_v_vslice[:, :, 2], iplot)
        rot_plot_vz_z(ext, sinks, rot_arr_v_vslice[:, :, 2], iplot, e_r, e_z)


def callback_dump(icall, istep, t):
    print(f"callback_dump: icall = {icall} istep = {istep} t = {t}")
    model.dump(get_dump_name(icall))


def callback_plot(icall, istep, t):
    print(f"callback_plot: icall = {icall} istep = {istep} t = {t}")
    plot_state(icall)


def attach_callbacks():
    model.attach_callback("callback_dump", callback_dump)
    model.attach_callback("callback_plot", callback_plot)


attach_callbacks()

if run_post_setup:
    model.change_htolerance(1.3)
    model.timestep()
    model.change_htolerance(1.1)

model.evolve_until(nstop * dt_stop)
