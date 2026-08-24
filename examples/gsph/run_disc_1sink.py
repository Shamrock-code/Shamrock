"""
Disc with sink in GSPH
====================================================================

"""

import os

import numpy as np

import shamrock

outputdir = "_to_trash/gsph_onesink/"
os.system("mkdir -p " + outputdir)
####################################################
# Setup parameters
####################################################
Npart = 10000
disc_mass = 0.001  # sol mass

rout = 350
rin = 90

# alpha_ss ~ alpha_AV * 0.08
alpha_AV = 0.3
alpha_u = 1
beta_AV = 2

q = 0.15
p = 1.0
r0 = 1
Rcav = 2.5
delta_0 = 1e-5

C_cour = 0.3
C_force = 0.25

H_r_in = 0.05

dump_prefix = "SPH_"

# central star params
center_mass = 2.5
center_racc = 1

# hierarichle split
split_list = [
    {
        "index": 0,
        "mass_ratio": 0.5 / 2,
        "a": 40,
        "e": 0.5,
        "euler_angle": (np.radians(90), 0.0, 0.0),
    },
    # {"index" : 0, "mass_ratio" : 0.5, "a": 0.33333333, "e":0., "euler_angle" :(0,0,0)}
]

do_plots = True

####################################################
####################################################
####################################################

si = shamrock.UnitSystem()
sicte = shamrock.Constants(si)
codeu = shamrock.UnitSystem(
    unit_time=3600 * 24 * 365,
    unit_length=sicte.au() * (39.42410494106729 ** (1 / 3)),  # GM=1
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
    return sigma_0 * (r / rin) ** (-p)


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
def kepler_to_cartesian_no_rotation(m1, m2, a, e, nu=np.radians(0)):

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


def rotation_matrix(roll, pitch, yaw):
    """
    Create a rotation matrix for a 3D rotation given Euler angles (roll, pitch, yaw).
    :param roll: Rotation about the X-axis (in radians)
    :param pitch: Rotation about the Y-axis (in radians)
    :param yaw: Rotation about the Z-axis (in radians)
    :return: 3x3 rotation matrix
    """
    # Rotation matrix around X-axis (roll)
    Rx = np.array([[1, 0, 0], [0, np.cos(roll), -np.sin(roll)], [0, np.sin(roll), np.cos(roll)]])

    # Rotation matrix around Y-axis (pitch)
    Ry = np.array(
        [[np.cos(pitch), 0, np.sin(pitch)], [0, 1, 0], [-np.sin(pitch), 0, np.cos(pitch)]]
    )

    # Rotation matrix around Z-axis (yaw)
    Rz = np.array([[np.cos(yaw), -np.sin(yaw), 0], [np.sin(yaw), np.cos(yaw), 0], [0, 0, 1]])

    # Combine the rotations (R = Rz * Ry * Rx)
    R = np.dot(Rz, np.dot(Ry, Rx))

    return R


def rotate_point(point, roll, pitch, yaw):
    """
    Rotate a 3D point using Euler angles.
    :param point: 3D point as a list or tuple (x, y, z)
    :param roll: Rotation about the X-axis (in radians)
    :param pitch: Rotation about the Y-axis (in radians)
    :param yaw: Rotation about the Z-axis (in radians)
    :return: Rotated point as a tuple (x', y', z')
    """
    # Ensure the point is a numpy array
    point = np.array(point)

    # Get the rotation matrix
    R = rotation_matrix(roll, pitch, yaw)

    # Perform the rotation by multiplying the point with the rotation matrix
    rotated_point = np.dot(R, point)

    return tuple(rotated_point)


def split_as_binary(sink, m1, m2, a, e, euler_angles=(0.0, 0.0, 0.0)):
    r1, r2, v1, v2 = kepler_to_cartesian_no_rotation(m1, m2, a, e)

    roll, pitch, yaw = euler_angles
    r1 = rotate_point((r1[0], r1[1], 0.0), roll, pitch, yaw)
    r2 = rotate_point((r2[0], r2[1], 0.0), roll, pitch, yaw)
    v1 = rotate_point((v1[0], v1[1], 0.0), roll, pitch, yaw)
    v2 = rotate_point((v2[0], v2[1], 0.0), roll, pitch, yaw)

    s1 = {
        "mass": m1,
        "racc": sink["racc"],
        "pos": (sink["pos"][0] + r1[0], sink["pos"][1] + r1[1], sink["pos"][2] + r1[2]),
        "vel": (sink["vel"][0] + v1[0], sink["vel"][1] + v1[1], sink["vel"][2] + v1[2]),
    }

    s2 = {
        "mass": m2,
        "racc": sink["racc"],
        "pos": (sink["pos"][0] + r2[0], sink["pos"][1] + r2[1], sink["pos"][2] + r2[2]),
        "vel": (sink["vel"][0] + v2[0], sink["vel"][1] + v2[1], sink["vel"][2] + v2[2]),
    }

    print("-------------")
    print(sink)
    print(s1)
    print(s2)
    print(r1, r2, v1, v2)
    print("-------------")

    return s1, s2


####################################################
# Dump handling
####################################################
def get_dump_name(idump):
    return outputdir + dump_prefix + f"{idump:04}" + ".sham"


def get_vtk_dump_name(idump):
    return outputdir + dump_prefix + f"{idump:04}" + ".vtk"


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
        plt.plot(positions[:, i, 0], positions[:, i, 1], label=f"Body {i + 1}")

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

model = shamrock.get_Model_GSPH(context=ctx, vector_type="f64_3", sph_kernel="M4")

idump_last_dump = None
if idump_last_dump is not None:
    model.load_from_dump(get_dump_name(idump_last_dump))
else:
    cfg = model.gen_default_config()
    cfg.set_eos_locally_isothermalFA2014(h_over_r=H_r_in)
    cfg.set_riemann_exact()
    cfg.set_force_inutsuka_v2()
    cfg.set_reconstruct_piecewise_constant()
    cfg.print_status()
    cfg.set_units(codeu)
    model.set_solver_config(cfg)

    model.init_scheduler(int(8e5), 1)

    model.resize_simulation_box(bmin, bmax)

    sink_list = [
        {"mass": center_mass, "racc": center_racc, "pos": (0, 0, 0), "vel": (0, 0, 0)},
    ]

    print(f"sink_list = {sink_list}")

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
    print(f"sinks baryenceter : velocity {vel_bary} position {pos_bary}")

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

        print(f"add sink : mass {mass} pos {(x, y, z)} vel {(vx, vy, vz)} racc {racc}")
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


import matplotlib.pyplot as plt
import numpy as np

sink_history = []

t_start = model.get_time()

freq_stop = 100
norbit = 0.5

dt_stop = (1.0 / freq_stop) * 2 * np.pi * 300
nstop = int(norbit * freq_stop)

print(f"dt_stop = {dt_stop}")

t_stop = [i * dt_stop for i in range(nstop + 1)]
print(f"t_stop = {t_stop}")

idump = 0
istop = 0
c = 0
for ttarg in t_stop:
    if ttarg >= t_start and c < 10:
        # model.evolve_until(ttarg)
        model.evolve_once()

        model.do_vtk_dump(get_vtk_dump_name(idump), True)
        model.dump(get_dump_name(idump))

        idump += 1
        c += 1
