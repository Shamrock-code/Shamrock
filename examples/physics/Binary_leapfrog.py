"""
Binary orbit functions
=======================================

This example shows how to use binary orbit functions with the Post-Newtonian developments
and how to attach sink particles to an SPH model.
"""


import numpy as np
import shamrock as chama                 # chama: argot brésilien  qui signifie "allez!" ou "courage!"

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
G = ucte.G()       #bien 4pi^2 (compatible Ms,années UA)
#c = ucte.c()      #bien 63000 (compatible Ms,années UA)


# %%
# Simulation parameters
T=100  #nombre de période de la Terre autour du Soleil (nombre d'années) 
dt = 0.0005 # time step in code units
n_steps=int(T / dt)  # number of steps to evolve

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


def binary_initial_conditions(m1, m2, a, e, nu=0.0, G=G, roll=0.0, pitch=0.0, yaw=0.0):
    M = m1 + m2
    r = a * (1.0 - e * e) / (1.0 + e * np.cos(nu))  #juste Newton pour les conditions initiales

    h = np.sqrt(G * M * a * (1.0 - e * e))
    vr = G * M / h * e * np.sin(nu)
    vtheta = h / r

    x_rel = np.array([r * np.cos(nu), r * np.sin(nu), 0.0])
    v_rel = np.array([
        vr * np.cos(nu) - vtheta * np.sin(nu),
        vr * np.sin(nu) + vtheta * np.cos(nu),
        0.0,
    ])

    x1 = -m2 / M * x_rel
    x2 = m1 / M * x_rel
    v1 = -m2 / M * v_rel
    v2 = m1 / M * v_rel

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
    cfg.set_particle_mass(1e-6)
    cfg.set_eta_sink(0.01)
    cfg.set_eos_isothermal(1.0)
    # Set code units so warnings about unit system disappear
    cfg.set_units(codeu)

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

    model.add_sink(m1, tuple(x1.tolist()), tuple(v1.tolist()), racc)
    model.add_sink(m2, tuple(x2.tolist()), tuple(v2.tolist()), racc)


    # Initialise the scheduler first, then set a simulation box large enough
    model.init_scheduler(split_load, merge_load)

    ext = max(1.0, float(a) * 5.0)
    bmin = (-ext, -ext, -ext)
    bmax = (ext, ext, ext)
    model.resize_simulation_box(bmin, bmax)
    model.set_dt(dt_)

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
def run_binary_orbit_PN(model):
    """Evolve binary orbit for n_steps with timestep dt"""
    snapshots = []
    current_time = 0.0
    
    # Print initial conditions
    initial_sinks = model.get_sinks()
    print("\n=== INITIAL CONDITIONS ===")
    for i, sink in enumerate(initial_sinks):
        print(f"Sink {i+1}: pos={sink['pos']}, vel={sink['velocity']}, mass={sink['mass']}")
    print()

    for _ in range(n_steps):
        # Use evolve_once_override_time for sink-only dynamics
        # (no SPH particles, so CFL would be zero with evolve_until)
        next_dt = model.evolve_once_override_time(current_time, dt)
        current_time += dt
        
        positions, velocities = get_sink_positions(model)
        
        # Compute distance between sinks
        pos1 = np.array(positions[0])
        pos2 = np.array(positions[1])
        distance = np.linalg.norm(pos2 - pos1)
        
        # DEBUG: verify dt was used and distance between sinks
        print(f"t = {current_time:.4f}, dt = {next_dt:.6f}, distance = {distance:.6f}")

        snapshots.append({
            "time": current_time,
            "positions": positions,
            "velocities": velocities,
        })

    return snapshots


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
    ax3d.plot(sink1_positions[:, 0], sink1_positions[:, 1], sink1_positions[:, 2], 
              "o-", label="Sink 1", markersize=3, linewidth=1)
    ax3d.plot(sink2_positions[:, 0], sink2_positions[:, 1], sink2_positions[:, 2], 
              "s-", label="Sink 2", markersize=3, linewidth=1)
    ax3d.set_xlabel("x (AU)")
    ax3d.set_ylabel("y (AU)")
    ax3d.set_zlabel("z (AU)")
    ax3d.set_title("3D Binary Orbit")
    ax3d.legend()
    ax3d.set_aspect("equal")
    
    # 2D plot (xy plane)
    ax2d = fig.add_subplot(122)
    ax2d.plot(sink1_positions[:, 0], sink1_positions[:, 1], "o-", 
              label="Sink 1", markersize=0.025, linewidth=0.025)
    ax2d.plot(sink2_positions[:, 0], sink2_positions[:, 1], "s-", 
              label="Sink 2", markersize=0.025, linewidth=0.025)
    ax2d.set_xlabel("x (AU)")
    ax2d.set_ylabel("y (AU)")
    ax2d.set_title("Binary Orbit (xy plane)")
    ax2d.legend()
    ax2d.set_aspect("equal")
    ax2d.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()


# %%
# Example usage
if __name__ == "__main__":
    m1 = 1 
    m2 = 0.000006
    a = 1.0
    e = 0.0

    # racc=0.001 AU is much smaller than binary separation (~0.7 AU at periapsis)
    ctx, model = build_binary_sph_model(m1, m2, a, e, roll=0.0, pitch=0.0, yaw=0.0, racc=0.001)
    snapshots = run_binary_orbit_PN(model)

    for snapshot in snapshots[:3]:
        print("time", snapshot["time"], "positions", snapshot["positions"])

    plot_orbit_trajectory(snapshots)

