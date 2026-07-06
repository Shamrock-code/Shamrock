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
n_orbits = 200                                                      #nombre d'orbite qu'on veut
SF=20                                                              #safety factor, permet d'avoir plus de pas de temps par orbite pour une meilleure précision, nécessaire pour les cas extrêmes(excentricité proche de 1, spin très élevé, etc.)          
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
    cfg.set_eta_sink(1)
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
def run_binary_orbit_PN(model, n_steps=n_steps, dt=dt):
    """Evolve binary orbit for n_steps with timestep dt"""
    snapshots = []
    current_time = 0.0

    # Print initial conditions
    initial_sinks = model.get_sinks()
    print("\n=== INITIAL CONDITIONS ===")
    for i, sink in enumerate(initial_sinks):
        print(f"Sink {i + 1}: pos={sink['pos']}, vel={sink['velocity']}, mass={sink['mass']}")
    print()

    for _ in range(n_steps):
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


# %%
# Example usage
if __name__ == "__main__":
    m1 = M1
    m2 = M2
    a = A
    e = E

    # racc=0.001 AU is much smaller than binary separation (~0.7 AU at periapsis)
    ctx, model = build_binary_sph_model(m1, m2, a, e, roll=0.0, pitch=0.0, yaw=0.0, racc=0.001)
    snapshots = run_binary_orbit_PN(model)

    for snapshot in snapshots[:3]:
        print("time", snapshot["time"], "positions", snapshot["positions"])

    plot_orbit_trajectory(snapshots)

def compute_eccentricity(snapshot, m1, m2, G=G):

    r1 = np.array(snapshot["positions"][0])
    r2 = np.array(snapshot["positions"][1])

    v1 = np.array(snapshot["velocities"][0])
    v2 = np.array(snapshot["velocities"][1])

    r_vec = r2 - r1
    v_vec = v2 - v1

    r = np.linalg.norm(r_vec)
    v = np.linalg.norm(v_vec)

    mu = G * (m1 + m2)

    # énergie spécifique
    eps = 0.5 * v**2 - mu / r

    # moment cinétique spécifique
    h = np.linalg.norm(np.cross(r_vec, v_vec))

    e = np.sqrt(1 + 2 * eps * h**2 / mu**2)      # e =sqrt(1 - h*2/a mu) = sqrt(1 + 2*eps*h^2/mu^2)

    return e

def plot_eccentricity(snapshots, m1, m2):

    times = np.array([snap["time"] for snap in snapshots])
    ecc = np.array([
        compute_eccentricity(snap, m1, m2)
        for snap in snapshots
    ])

    plt.figure(figsize=(8, 4))
    plt.plot(times, ecc, lw=2)

    plt.xlabel("Temps (années)")
    plt.ylabel("Excentricité e")
    plt.title("Excentricité en fct du temps")

    plt.ylim(0, 1)  

    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()



def compute_inclination(snapshot):

    r1 = np.array(snapshot["positions"][0])
    r2 = np.array(snapshot["positions"][1])

    v1 = np.array(snapshot["velocities"][0])
    v2 = np.array(snapshot["velocities"][1])

    r_vec = r2 - r1
    v_vec = v2 - v1

    h = np.cross(r_vec, v_vec)                 #"specific angular momentum" doc Cart2kep de G.Laibe

    h_norm = np.linalg.norm(h)
    if h_norm < 1e-15:
        return np.nan

    i = np.arccos(np.clip(h[2] / h_norm, -1.0, 1.0))      # cos(i)= h_z/|h|

    return np.degrees(i)

def plot_inclination(snapshots):


    times = np.array([snap["time"] for snap in snapshots])

    inc = np.array([
        compute_inclination(snap)
        for snap in snapshots
    ])

    plt.figure(figsize=(8,4))
    plt.plot(times, inc)

    plt.xlabel("Temps (années)")
    plt.ylabel("Inclinaison (°)")
    plt.title("Inclinaison orbitale")
    plt.ylim(-180, 180) 
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def compute_semimajor_axis(snapshot, m1, m2, G=G):

    r1 = np.array(snapshot["positions"][0])
    r2 = np.array(snapshot["positions"][1])

    v1 = np.array(snapshot["velocities"][0])
    v2 = np.array(snapshot["velocities"][1])

    r_vec = r2 - r1
    v_vec = v2 - v1

    r = np.linalg.norm(r_vec)
    v = np.linalg.norm(v_vec)

    mu = G * (m1 + m2)

    eps = 0.5 * v**2 - mu / r

    if abs(eps) < 1e-15:
        return np.nan

    return -mu / (2 * eps)      # = a

def plot_semimajor_axis(snapshots, m1, m2):


    times = np.array([snap["time"] for snap in snapshots])

    a = np.array([
        compute_semimajor_axis(snap, m1, m2)
        for snap in snapshots
    ])

    plt.figure(figsize=(8,4))
    plt.plot(times, a)

    plt.xlabel("Temps (années)")
    plt.ylabel("Demi-grand axe (AU)")
    plt.title("Évolution du demi-grand axe")
    plt.ylim(0, 1.5*A) 
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def compute_eccentricity_vector(snapshot, m1, m2, G=G):

    r1 = np.array(snapshot["positions"][0])
    r2 = np.array(snapshot["positions"][1])

    v1 = np.array(snapshot["velocities"][0])
    v2 = np.array(snapshot["velocities"][1])

    r_vec = r2 - r1
    v_vec = v2 - v1

    r = np.linalg.norm(r_vec)

    mu = G * (m1 + m2)

    h = np.cross(r_vec, v_vec)

    e_vec = np.cross(v_vec, h)/mu - r_vec/r
    e=np.linalg.norm(e_vec)
    e_vec = e_vec / e if e != 0 else np.zeros_like(e_vec)  #vecteur excentricité normé ou de Laplace-Runge-Lenz

    return e_vec

def plot_eccentricity_vector(snapshots, m1, m2):

    times = np.array([snap["time"] for snap in snapshots])

    e_vec = np.array([
        compute_eccentricity_vector(snap, m1, m2)
        for snap in snapshots
    ])

    plt.figure(figsize=(9,5))

    plt.plot(times, e_vec[:,0], label=r"$e_x (OP) $")
    plt.plot(times, e_vec[:,1], label=r"$e_y (OP) $")
    plt.plot(times, e_vec[:,2], label=r"$e_z  (SO) $")

    plt.xlabel("Temps (années)")
    plt.ylabel("Composantes du vecteur d'excentricité")
    plt.title("Évolution du vecteur d'excentricité")
    plt.grid(True, alpha=0.3)
    plt.legend()

    plt.tight_layout()
    plt.show()


def plot_spins(snapshots):

    times = np.array([s["time"] for s in snapshots])

    a1 = c/(G*m1*m1)*np.array([s["spins"][0] for s in snapshots])
    a2 = c/(G*m2*m2)*np.array([s["spins"][1] for s in snapshots])

    plt.figure(figsize=(10,6))

    # BH1
    plt.plot(times, a1[:,0], label=r"$a_{1x}$")
    plt.plot(times, a1[:,1], label=r"$a_{1y}$")
    plt.plot(times, a1[:,2], label=r"$a_{1z}$")

    # BH2
    plt.plot(times, a2[:,0], "--", label=r"$a_{2x}$")
    plt.plot(times, a2[:,1], "--", label=r"$a_{2y}$")
    plt.plot(times, a2[:,2], "--", label=r"$a_{2z}$")

    plt.xlabel("Temps")
    plt.ylabel("Spin")

    plt.title("Évolution des composantes des spins")

    plt.grid(alpha=0.3)
    plt.legend(ncol=2)

    plt.tight_layout()
    plt.show()


#Différents plots pour visualiser l'évolution de l'orbite binaire
plot_eccentricity(snapshots, m1, m2)
plot_eccentricity_vector(snapshots, m1, m2)
plot_inclination(snapshots)
plot_semimajor_axis(snapshots, m1, m2)

sinks = model.get_sinks()
print(sinks[0])


plot_spins(snapshots)