"""
Sink integration test problems
=======================================

This example shows how to use the sink integration test problems.
"""

import numpy as np

import shamrock

# %%
# Use shamrock documentation style for matplotlib
shamrock.matplotlib.set_shamrock_mpl_style()


# %%
# Define the unit system
si = shamrock.UnitSystem()
sicte = shamrock.Constants(si)
codeu = shamrock.UnitSystem(
    unit_time=sicte.year(),
    unit_length=sicte.au(),
    unit_mass=sicte.sol_mass(),
)
ucte = shamrock.Constants(codeu)
G = ucte.G()


# %%
# Build the SPH model with the sink particles
def build_sink_sph_model(
    positions,
    velocities,
    masses,
    accretion_radii,
    box_extent,
    eta_sink=0.01,
    cfl_force=0.1,
    cfl_cour=0.1,
):
    ctx = shamrock.Context()
    ctx.pdata_layout_new()

    model = shamrock.get_Model_SPH(context=ctx, vector_type="f64_3", sph_kernel="M4")

    # Allow experimental features (required for self-gravity)
    shamrock.enable_experimental_features()

    cfg = model.gen_default_config()
    # Disable direct self-gravity in this simple example; direct mode requires
    # a single-patch setup which is not prepared here.
    cfg.set_self_gravity_none()
    cfg.set_artif_viscosity_Constant(alpha_u=1.0, alpha_AV=1.0, beta_AV=2.0)
    cfg.set_particle_mass(1e-6)
    cfg.set_eos_isothermal(1.0)
    cfg.set_show_cfl_detail(True)
    cfg.set_eta_sink(eta_sink)
    cfg.set_cfl_force(cfl_force)
    cfg.set_cfl_cour(cfl_cour)
    # Set code units so warnings about unit system disappear
    cfg.set_units(codeu)

    model.set_solver_config(cfg)

    for position, velocity, mass, accretion_radius in zip(
        positions, velocities, masses, accretion_radii
    ):
        model.add_sink(mass, tuple(position), tuple(velocity), accretion_radius)

    # Initialise the scheduler first, then set a simulation box large enough
    model.init_scheduler(int(1e7), 1)

    ext = box_extent
    bmin = (-ext, -ext, -ext)
    bmax = (ext, ext, ext)
    model.resize_simulation_box(bmin, bmax)

    return ctx, model


# %%
# Extract sink positions from the model
def get_sink_positions(model):
    sinks = model.get_sinks()
    positions = [tuple(sink["pos"]) for sink in sinks]
    velocities = [tuple(sink["velocity"]) for sink in sinks]
    return positions, velocities


# %%
def correct_sink_velocities_zero_momentum(velocities, masses):
    """Subtract center-of-mass velocity so total momentum vanishes."""
    masses = np.asarray(masses)
    vels = np.asarray(velocities)
    v_com = np.sum(masses[:, np.newaxis] * vels, axis=0) / np.sum(masses)
    return [tuple(v - v_com) for v in vels]


# %%
# Run a simple orbit evolution and collect sink snapshots
def run_sim(model, max_time, use_dt=None):
    """Evolve binary orbit until max_time"""
    snapshots = []
    current_time = 0.0

    # Print initial conditions
    initial_sinks = model.get_sinks()
    print("\n=== INITIAL CONDITIONS ===")
    for i, sink in enumerate(initial_sinks):
        print(f"Sink {i + 1}: pos={sink['pos']}, vel={sink['velocity']}, mass={sink['mass']}")
    print()

    while current_time < max_time:
        if use_dt is None:
            model.timestep()
        else:
            model.evolve_once_override_time(current_time + use_dt, use_dt)
        current_time = model.get_time()

        positions, velocities = get_sink_positions(model)

        snapshots.append(
            {
                "time": current_time,
                "positions": positions,
                "velocities": velocities,
            }
        )

    return snapshots


# %%
# Plot complete orbital trajectories
def plot_orbit_trajectory(snapshots, suptitle):
    import matplotlib.pyplot as plt

    sinks_positions = np.array([snap["positions"] for snap in snapshots])

    nstep, nsink, ndim = sinks_positions.shape

    print(sinks_positions.shape)

    # Extract trajectories for both sinks
    sink2_positions = np.array([snap["positions"][1] for snap in snapshots])

    fig = plt.figure(figsize=(12, 5))
    fig.suptitle(suptitle)

    # 3D plot
    ax3d = fig.add_subplot(121, projection="3d")

    for isink in range(nsink):
        ax3d.plot(
            sinks_positions[:, isink, 0],
            sinks_positions[:, isink, 1],
            sinks_positions[:, isink, 2],
            "o-",
            label=f"Sink {isink + 1}",
            markersize=3,
            linewidth=1,
        )
    ax3d.set_xlabel("x (AU)")
    ax3d.set_ylabel("y (AU)")
    ax3d.set_zlabel("z (AU)")
    ax3d.set_title("3D Orbit")
    ax3d.legend()
    ax3d.set_aspect("equal")

    # 2D plot (xy plane)
    ax2d = fig.add_subplot(122)

    for isink in range(nsink):
        ax2d.plot(
            sinks_positions[:, isink, 0],
            sinks_positions[:, isink, 1],
            "o-",
            label=f"Sink {isink + 1}",
            markersize=1,
            linewidth=1,
        )

    ax2d.set_xlabel("x (AU)")
    ax2d.set_ylabel("y (AU)")
    ax2d.set_title("Orbit (xy plane)")
    ax2d.legend()
    ax2d.set_aspect("equal")
    ax2d.grid(True, alpha=0.3)

    fig.tight_layout()
    plt.show()


# %%
# Circular orbit
m1 = 1.0
m2 = 0.2
a = 1.0
_x1, _x2, _v1, _v2 = shamrock.phys.get_binary_rotated(
    m1=1.0, m2=m2, a=a, e=0.2, nu=0.0, G=G, roll=0.0, pitch=0.0, yaw=0.0
)
ctx, model = build_sink_sph_model(
    positions=[_x1, _x2],
    velocities=[_v1, _v2],
    masses=[m1, m2],
    accretion_radii=[1, 1],
    box_extent=3,
    eta_sink=0.5,
)
snapshots = run_sim(model, 100, use_dt=None)
plot_orbit_trajectory(snapshots, "Circular orbit")

