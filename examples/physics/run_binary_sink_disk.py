"""
Binary orbit functions
=======================================

This example shows how to use binary orbit functions with the Post-Newtonian developments
and how to attach sink particles to an SPH model.
"""

import numpy as np
import matplotlib.pyplot as plt
import shamrock as chama

# %%
# Use shamrock documentation style for matplotlib
chama.matplotlib.set_shamrock_mpl_style()


# %%
# Define the unit system
si = chama.UnitSystem()
sicte = chama.Constants(si)
codeu = chama.UnitSystem(
    unit_time=sicte.year(),
    unit_length=sicte.au(),
    unit_mass=sicte.sol_mass(),
)
ucte = chama.Constants(codeu)
G = ucte.G()
c = ucte.c()

#Paramètres de la binaire à tracer
M1 = 1000.0 # masse du premier corps en masses solaires
M2 = 600 # masse du second corps en masses solaires
A = 1.0  # demi-grand axe en unités astronomiques
E = 0.2  # excentricité de l'orbite

#Paramètres du disque
center_mass = M1 + M2
Rg = G * center_mass / c**2

# Résolution du disque
Npart = 20000


# Disc parameters
# These values are used by the generator and for the visualization extents.
disc_mass = 0.1  # [sol mass]
rout = 10.0 *Rg # [au]
rin = 4*Rg  # [au]
H_r_0 = 0.05
q = 0.5
p = 3.0 / 2.0
r0 = rin


def sigma_profile(r):
    sigma_0 = 1.0  # We do not care as it will be renormalized
    return sigma_0 * (r / r0) ** (-p)


def kep_profile(r):
    return (G * center_mass / r) ** 0.5


def omega_k(r):
    return kep_profile(r) / r


def cs_profile(r):
    cs_in = (H_r_0 * r0) * omega_k(r0)
    return ((r / r0) ** (-q)) * cs_in


cs0 = cs_profile(r0)


def rot_profile(r):
    return ((kep_profile(r) ** 2) - (2 * p + q) * cs_profile(r) ** 2) ** 0.5


def H_profile(r):
    H = cs_profile(r) / omega_k(r)
    return H


# Spins (aligned with the orbital angular momentum)
a1 = 0.99  # entre 0 et 1, spin du premier corps
a2 = 0.5
theta =np.pi/4
spin_axis = np.array([0.0, np.sin(theta), np.cos(theta)])  # axis of spin (unit vector)
spin_mag_1 = a1 * G * M1 * M1 / c
spin_mag_2 = a2 * G * M2 * M2 / c
spin_vec_1 = spin_mag_1 * spin_axis
spin_vec_2 = spin_mag_2 * spin_axis

X = np.sqrt(A**3/(G*(M1+M2)))  # période orbitale en années
# %%
# Simulation parameters
T = 2*np.pi*np.sqrt(A*A*A/(G*(M1+M2)))                             # number of years
n_orbits = 1                                                      #nombre d'orbite qu'on veut
SF=10                                                              #safety factor, permet d'avoir plus de pas de temps par orbite pour une meilleure précision, nécessaire pour les cas extrêmes(excentricité proche de 1, spin très élevé, etc.)          
                                                                   #augmente également le temps de calcul, à ajuster selon les besoins      
N_per_orbits = SF*20/(np.sqrt(1+E)*(1-E)**(3/2))                   # number of time steps per orbit
n_steps = int(n_orbits*N_per_orbits)                               # number of steps to evolve
dt = T/N_per_orbits                                                # time step in years


# %%
# Orbital initialization without get_binary_rotated


def rotation_matrix(roll, pitch, yaw):
    cr, sr = np.cos(roll), np.sin(roll)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cy, sy = np.cos(yaw), np.sin(yaw)

    Rx = np.array([[1.0, 0.0, 0.0], [0.0, cr, -sr], [0.0, sr, cr]])
    Ry = np.array([[cp, 0.0, sp], [0.0, 1.0, 0.0], [-sp, 0.0, cp]])
    Rz = np.array([[cy, -sy, 0.0], [sy, cy, 0.0], [0.0, 0.0, 1.0]])

    return Rz @ Ry @ Rx


def binary_initial_conditions(
    m1,
    m2,
    a,
    e,
    nu=0.0,
    G=G,
    roll=0.0,
    pitch=0.0,
    yaw=0.0,
):
    M = m1 + m2

    # mêmes conditions que ton code Leapfrog
    r = a * (1 - e)
    v = np.sqrt((1 + e) * G * M / r)

    x_rel = np.array([r, 0.0, 0.0])
    v_rel = np.array([0.0, v, 0.0])

    x1 = -m2 / M * x_rel
    x2 =  m1 / M * x_rel

    v1 = -m2 / M * v_rel
    v2 =  m1 / M * v_rel

    if roll != 0.0 or pitch != 0.0 or yaw != 0.0:
        R = rotation_matrix(roll, pitch, yaw)
        x1 = R @ x1
        x2 = R @ x2
        v1 = R @ v1
        v2 = R @ v2

    return x1, x2, v1, v2


def build_binary_sph_model(
    m1,
    m2,
    a,
    e,
    roll=0.0,
    pitch=0.0,
    yaw=0.0,
    racc=0.1,
    dt_=dt,
    split_load=10_000_000,
    merge_load=1,
    generate_disk=True,
    compute_op=False,
    compute_so=False,
    compute_ss=False,
    compute_rr=False,
):
    ctx = chama.Context()
    ctx.pdata_layout_new()

    model = chama.get_Model_SPH(context=ctx, vector_type="f64_3", sph_kernel="M4")

    # Allow experimental features (required for self-gravity)
    chama.enable_experimental_features()

    cfg = model.gen_default_config()
    # Disable direct self-gravity in this simple example; direct mode requires
    # a single-patch setup which is not prepared here.
    cfg.set_self_gravity_none()
    cfg.set_artif_viscosity_Constant(alpha_u=1.0, alpha_AV=1.0, beta_AV=2.0)
    pmass = disc_mass / Npart
    cfg.set_particle_mass(pmass)
    cfg.set_eta_sink(1)
    cfg.set_eos_locally_isothermalLP07(cs0=cs_profile(r0), q=q, r0=r0)
    cfg.set_units(codeu)
    cfg.set_cfl_cour(0.3)
    cfg.set_cfl_force(0.25)
    cfg.set_smoothing_length_density_based()
    cfg.set_compute_OP(compute_op)
    cfg.set_compute_SO(compute_so)
    cfg.set_compute_SS(compute_ss)
    cfg.set_compute_RR(compute_rr)

    model.set_solver_config(cfg)

    x1, x2, v1, v2 = binary_initial_conditions(
        m1=m1,
        m2=m2,
        a=a,
        e=e,
        nu=0.0,
        G=G,
        roll=roll,
        pitch=pitch,
        yaw=yaw,
    )

    model.add_sink(
        m1,
        tuple(x1.tolist()),
        tuple(v1.tolist()),
        racc,
        tuple(spin_vec_1.tolist()),
    )
    model.add_sink(
        m2,
        tuple(x2.tolist()),
        tuple(v2.tolist()),
        racc,
        tuple(spin_vec_2.tolist()),
    )

    # Initialise the scheduler first, then set a simulation box large enough
    model.init_scheduler(split_load, merge_load)

    ext = max(1.0, float(a) * 5.0, float(rout) * 2.0)
    bmin = (-ext, -ext, -ext)
    bmax = (ext, ext, ext)
    model.resize_simulation_box(bmin, bmax)
    model.set_dt(dt_)

    if generate_disk:
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
            init_h_factor=0.03,
        )
        setup.apply_setup(gen_disc)

    return ctx, model


# %%
# Extract sink positions from the model
def get_sink_positions(model):
    sinks = model.get_sinks()
    positions = [tuple(sink["pos"]) for sink in sinks]
    velocities = [tuple(sink["velocity"]) for sink in sinks]
    return positions, velocities


# %%
# Run a simple orbit evolution and collect sink snapshots
def run_binary_orbit_PN(
    model,
    n_steps=n_steps,
    dt=dt,
    render=False,
    render_ext=2.5,
    nx=256,
    ny=256,
    render_stride=1,
):
    """Evolve the binary orbit and optionally store disk density frames for animation."""
    snapshots = []
    render_frames = []
    current_time = 0.0

    # Print initial conditions
    initial_sinks = model.get_sinks()
    print("\n=== INITIAL CONDITIONS ===")
    for i, sink in enumerate(initial_sinks):
        print(f"Sink {i + 1}: pos={sink['pos']}, vel={sink['velocity']}, mass={sink['mass']}")
    print()

    for istep in range(n_steps):
        # Use evolve_once_override_time for sink-only dynamics
        # (no SPH particles, so CFL would be zero with evolve_until)
        next_dt = model.evolve_once_override_time(current_time, dt)
        current_time += dt

        positions, velocities = get_sink_positions(model)

        sinks = model.get_sinks()

        spins = [
            np.array(sinks[0]["angular_momentum"]),
            np.array(sinks[1]["angular_momentum"]),
        ]

        # Compute distance between sinks
        pos1 = np.array(positions[0])
        pos2 = np.array(positions[1])
        distance = np.linalg.norm(pos2 - pos1)

        # DEBUG: verify dt was used and distance between sinks
        print(f"t = {current_time:.4f}, dt = {next_dt:.6f}, distance = {distance:.6f}")

        snapshots.append(
            {
                "time": current_time,
                "positions": positions,
                "velocities": velocities,
                "spins": spins,
            }
        )

        if render and istep % render_stride == 0:
            center = (0.0, 0.0, 0.0)
            delta_x = (render_ext * 2.0, 0.0, 0.0)
            delta_y = (0.0, render_ext * 2.0, 0.0)
            arr_rho = model.render_cartesian_column_integ(
                "rho",
                "f64",
                center=center,
                delta_x=delta_x,
                delta_y=delta_y,
                nx=nx,
                ny=ny,
            )
            render_frames.append(
                {
                    "time": current_time,
                    "positions": positions,
                    "rho": arr_rho,
                }
            )

    return snapshots, render_frames


# %%
# Plot sink particle positions for a single snapshot
def plot_sink_snapshot(snapshot):
    import matplotlib.pyplot as plt

    positions = np.array(snapshot["positions"])
    xs, ys, zs = positions[:, 0], positions[:, 1], positions[:, 2]

    fig = plt.figure()
    ax = fig.add_subplot(projection="3d")
    ax.plot(xs, ys, zs, "o-", color="tab:blue")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    ax.set_title(f"Binary sinks at t = {snapshot['time']:.3f}")
    ax.set_aspect("equal")
    plt.show()


# %%
# Plot complete orbital trajectories
def plot_orbit_trajectory(snapshots):
    import matplotlib.pyplot as plt

    # Extract trajectories for both sinks
    sink1_positions = np.array([snap["positions"][0] for snap in snapshots])
    sink2_positions = np.array([snap["positions"][1] for snap in snapshots])

    fig = plt.figure(figsize=(12, 5))

    # 3D plot
    ax3d = fig.add_subplot(121, projection="3d")
    ax3d.plot(
        sink1_positions[:, 0],
        sink1_positions[:, 1],
        sink1_positions[:, 2],
        "o-",
        label="Sink 1",
        markersize=0.050,
        linewidth=0.025,
    )
    ax3d.plot(
        sink2_positions[:, 0],
        sink2_positions[:, 1],
        sink2_positions[:, 2],
        "s-",
        label="Sink 2",
        markersize=0.050,
        linewidth=0.025,
    )
    ax3d.set_xlabel("x (AU)")
    ax3d.set_ylabel("y (AU)")
    ax3d.set_zlabel("z (AU)")
    ax3d.set_title("3D Binary Orbit")
    ax3d.legend()
    ax3d.set_aspect("equal")

    # 2D plot (xy plane)
    ax2d = fig.add_subplot(122)
    ax2d.plot(
        sink1_positions[:, 0],
        sink1_positions[:, 1],
        "o-",
        label="Sink 1",
        markersize=0.025,
        linewidth=0.025,
    )
    ax2d.plot(
        sink2_positions[:, 0],
        sink2_positions[:, 1],
        "s-",
        label="Sink 2",
        markersize=0.025,
        linewidth=0.025,
    )
    ax2d.set_xlabel("x (AU)")
    ax2d.set_ylabel("y (AU)")
    ax2d.set_title("Binary Orbit (xy plane)")
    ax2d.legend()
    ax2d.set_aspect("equal")
    ax2d.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()


def render_disk_and_orbit(render_frames, ext=2.5, nx=256, ny=256, interval=100):
    if len(render_frames) == 0:
        return

    x = np.linspace(-ext, ext, nx)
    y = np.linspace(-ext, ext, ny)
    X, Y = np.meshgrid(x, y)

    fig, ax = plt.subplots(figsize=(8, 8))
    rho_plot = np.log10(np.clip(render_frames[0]["rho"], 1e-30, None))
    img = ax.imshow(
        rho_plot.T,
        origin="lower",
        extent=(-ext, ext, -ext, ext),
        cmap="twilight_shifted",
        aspect="equal",
    )
    history1 = []
    history2 = []
    line1_hist, = ax.plot([], [], color="tab:red", lw=1.0, alpha=0.45)
    line2_hist, = ax.plot([], [], color="tab:blue", lw=1.0, alpha=0.45)
    point1, = ax.plot([], [], "o", color="tab:red", markersize=7)
    point2, = ax.plot([], [], "o", color="tab:blue", markersize=7)
    title = ax.text(0.02, 0.98, "", transform=ax.transAxes, va="top", ha="left")

    def update(frame_idx):
        frame = render_frames[frame_idx]
        rho = np.log10(np.clip(frame["rho"], 1e-30, None))
        img.set_data(rho.T)

        pos = np.array(frame["positions"])
        history1.append(pos[0])
        history2.append(pos[1])

        if len(history1) > 1:
            hist1 = np.array(history1)
            hist2 = np.array(history2)
            line1_hist.set_data(hist1[:, 0], hist1[:, 1])
            line2_hist.set_data(hist2[:, 0], hist2[:, 1])
        else:
            line1_hist.set_data([], [])
            line2_hist.set_data([], [])

        point1.set_data([pos[0, 0]], [pos[0, 1]])
        point2.set_data([pos[1, 0]], [pos[1, 1]])
        title.set_text(f"t = {frame['time']:.3f} yr")
        return img, line1_hist, line2_hist, point1, point2, title

    from matplotlib.animation import FuncAnimation

    ani = FuncAnimation(
        fig,
        update,
        frames=len(render_frames),
        interval=interval,
        blit=False,
    )

    ax.set_title("Disc + binary orbit")
    ax.set_xlabel("x (AU)")
    ax.set_ylabel("y (AU)")
    ax.set_xlim(-ext, ext)
    ax.set_ylim(-ext, ext)
    ax.set_aspect("equal")
    fig.colorbar(img, ax=ax, label="log10(rho)")
    plt.tight_layout()
    plt.show()


# %%
# Example usage
if __name__ == "__main__":
    m1 = M1
    m2 = M2
    a = A
    e = E

    # racc=0.001 AU is much smaller than binary separation (~0.7 AU at periapsis)
    ctx, model = build_binary_sph_model(
        m1,
        m2,
        a,
        e,
        roll=0.0,
        pitch=0.0,
        yaw=0.0,
        racc=0.001,
        compute_op=False,
        compute_so=False,
        compute_ss=False,
        compute_rr=False,
    )
    snapshots, render_frames = run_binary_orbit_PN(
        model,
        render=True,
        render_ext=2.5,
        nx=256,
        ny=256,
    )

    for snapshot in snapshots[:3]:
        print("time", snapshot["time"], "positions", snapshot["positions"])

    plot_orbit_trajectory(snapshots)
    render_disk_and_orbit(render_frames)

