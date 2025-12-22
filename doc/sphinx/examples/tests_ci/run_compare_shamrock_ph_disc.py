"""
Comparing Shamrock disc with Phantom disc
=================================================

Check that both codes generate the same profiles for a disc.
"""

import os
import subprocess

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

import shamrock

si = shamrock.UnitSystem()
sicte = shamrock.Constants(si)
codeu = shamrock.UnitSystem(
    unit_time=sicte.year() / (2 * np.pi),
    unit_length=sicte.au(),
    unit_mass=sicte.sol_mass(),
)
ucte = shamrock.Constants(codeu)
G = ucte.G()

time_end = 6.0
R_in = 1.0
R_out = 10.0
R_ref = 1.0
disc_mass = 0.05
Npart = 100000
p_index = 3.0 / 2.0
q_index = 0.50
H_R = 0.05
alpha_SS = 0.005
m1 = 1.0

pmass = disc_mass / Npart


# Disc profiles
def sigma_profile(r):
    sigma_0 = 1.0  # We do not care as it will be renormalized
    return sigma_0 * (r / R_ref) ** (-p_index)


def kep_profile(r):
    return (G * m1 / r) ** 0.5


def omega_k(r):
    return kep_profile(r) / r


def cs_profile(r):
    cs_in = (H_R * R_ref) * omega_k(R_ref)
    return ((r / R_ref) ** (-q_index)) * cs_in


def get_sigma_norm():
    x_list = np.linspace(R_in, R_out, 2048)
    term = [sigma_profile(x) * 2 * np.pi * x for x in x_list]
    return disc_mass / (np.sum(term) * (x_list[1] - x_list[0]))


sigma_norm = get_sigma_norm()


def rot_profile(r):
    return ((kep_profile(r) ** 2) - (2 * p_index + q_index) * cs_profile(r) ** 2) ** 0.5


def H_profile(r):
    H = cs_profile(r) / omega_k(r)
    # fact = (2.**0.5) * 3. # factor taken from phantom, to fasten thermalizing
    fact = 1.0
    return fact * H


MAKE_PH_DISC = R"""
set -e

#rm -rf phantom_run
mkdir -p phantom_run

cd phantom_run
#git clone https://github.com/danieljprice/phantom.git
cd phantom
mkdir -p disc; cd disc

echo "Writing Makefile"
../scripts/writemake.sh disc > Makefile

echo "Running Makefile"

export SYSTEM=gfortran
make; make setup

cp ../../../disc_setup.setup disc.setup
./phantomsetup disc

# change end time to 6.0
sed -i 's/^\(\s*tmax\s*=\s*\).*\(! end time\)/\1  6.0    \2/' disc.in

echo "Running Phantom"
./phantom disc.in
"""


DISC_SETUP = """
# input file for disc setup routine

# resolution
                  np =      100000    ! number of gas particles

# units
           mass_unit =      solarm    ! mass unit (e.g. solarm,jupiterm,1e6*solarm)
           dist_unit =          au    ! distance unit (e.g. au,pc,kpc,0.1pc)

# central object(s)/potential
            icentral =           1    ! use sink particles or external potential (0=potential,1=sinks)
              nsinks =           1    ! number of sinks

# options for central star
                  m1 =       1.000    ! star mass
               accr1 =       1.000    ! star accretion radius

# oblateness
            J2_body1 =       0.000    ! J2 moment (oblateness)

# options for gas accretion disc
             isetgas =           0    ! how to set gas density profile (0=total disc mass,1=mass within annulus,2=surface density normalisation,3=surface density at reference radius,4=minimum Toomre Q,5=minimum Toomre Q and Lstar)
          sigma_file =           F    ! reading gas profile from file sigma_grid.dat
           itapergas =           F    ! exponentially taper the outer disc profile
          ismoothgas =           T    ! smooth inner disc
               iwarp =           F    ! warp disc
                iecc =           F    ! eccentric disc
                R_in =       1.000    ! inner radius
               R_ref =         1.     ! reference radius
               R_out =        10.     ! outer radius
              disc_m =       0.050    ! disc mass
             lumdisc =           0    ! Set qindex from stellar luminosity (ieos=24) (0=no 1=yes)
              pindex =       1.500    ! power law index of surface density sig=sig0*r^-p
              qindex =       0.500    ! power law index of sound speed cs=cs0*r^-q
             posangl =       0.000    ! position angle (deg)
                incl =       0.000    ! inclination (deg)
                 H_R =       0.050    ! H/R at R=R_ref
             alphaSS =       0.005    ! desired alphaSS (0 for minimal needed for shock capturing)

# Minimum Temperature in the Simulation
             T_floor =       0.000    ! The minimum temperature in the simulation (for any locally isothermal EOS).

# set sphere around disc
          add_sphere =           F    ! add sphere around disc?

# set planets
            nplanets =           0    ! number of planets

# thermal stratification
           discstrat =           0    ! stratify disc? (0=no,1=yes)

# timestepping
             norbits =           1    ! maximum number of orbits at outer disc
              deltat =       0.100    ! output interval as fraction of orbital period
"""


def run_phantom_disc():
    # write the script to make the phantom disc
    with open("make_ph_disc.sh", "w") as f:
        f.write(MAKE_PH_DISC)

    with open("disc_setup.setup", "w") as f:
        f.write(DISC_SETUP)

    # run the script
    subprocess.run(["bash", "./make_ph_disc.sh"], check=True)


def run_shamrock_disc():

    ctx = shamrock.Context()
    ctx.pdata_layout_new()
    model = shamrock.get_Model_SPH(context=ctx, vector_type="f64_3", sph_kernel="M4")

    alpha_AV = 0.0765404518492
    alpha_u = 1.0
    beta_AV = 2.0
    C_cour = 0.3
    C_force = 0.25
    cs0 = cs_profile(R_ref)
    bsize = R_out * 1.2

    # Generate the default config
    cfg = model.gen_default_config()
    cfg.set_artif_viscosity_ConstantDisc(alpha_u=alpha_u, alpha_AV=alpha_AV, beta_AV=beta_AV)
    cfg.set_eos_locally_isothermalLP07(cs0=cs0, q=q_index, r0=R_ref)

    cfg.add_kill_sphere(center=(0, 0, 0), radius=bsize)  # kill particles outside the simulation box

    cfg.set_units(codeu)
    cfg.set_particle_mass(pmass)
    # Set the CFL
    cfg.set_cfl_cour(C_cour)
    cfg.set_cfl_force(C_force)

    cfg.set_smoothing_length_density_based()

    # Set the solver config to be the one stored in cfg
    model.set_solver_config(cfg)

    # Print the solver config
    model.get_current_config().print_status()

    # Init the scheduler & fields
    model.init_scheduler(int(1e8), 1)

    # Set the simulation box size
    ext = R_out * 1.2
    model.resize_simulation_box((-ext, -ext, -ext), (ext, ext, ext))

    # Create the setup
    setup = model.get_setup()
    gen_disc = setup.make_generator_disc_mc(
        part_mass=pmass,
        disc_mass=disc_mass,
        r_in=R_in,
        r_out=R_out,
        sigma_profile=sigma_profile,
        H_profile=H_profile,
        rot_profile=rot_profile,
        cs_profile=cs_profile,
        random_seed=666,
    )

    # Print the dot graph of the setup
    print(gen_disc.get_dot())

    # Apply the setup
    setup.apply_setup(gen_disc, insert_step=1000000)

    # now that the barycenter & momentum are 0, we can add the sink
    model.add_sink(m1, (0, 0, 0), (0, 0, 0), 1.0)

    model.evolve_until(time_end)

    dump = model.make_phantom_dump()
    dump.save_dump("shamrock_disc.phdump")

    del model
    del ctx


run_phantom_disc()
run_shamrock_disc()


def positions_to_rays(positions):
    return [shamrock.math.Ray_f64_3(tuple(position), (0.0, 0.0, 1.0)) for position in positions]


def compute_avg_sigma_profile(model, ntheta, r):

    theta = np.linspace(0, 2 * np.pi, ntheta)

    r_grid, theta_grid = np.meshgrid(r, theta)
    x_grid = r_grid * np.cos(theta_grid)
    y_grid = r_grid * np.sin(theta_grid)
    z_grid = np.zeros_like(r_grid)

    positions = np.column_stack([x_grid.ravel(), y_grid.ravel(), z_grid.ravel()])

    rays = positions_to_rays(positions)
    arr_sigma = model.render_column_integ("rho", "f64", rays)

    arr_sigma = np.array(arr_sigma).reshape(ntheta, len(r))

    # average over the theta direction
    arr_sigma = np.mean(arr_sigma, axis=0)
    return arr_sigma


def get_profiles(model):
    x_list = np.linspace(0, R_out * 1.2, 2049)[1:]
    positions = [(x, 0.0, 0.0) for x in x_list.tolist()]

    rays = positions_to_rays(positions)

    arr_rho = model.render_slice("rho", "f64", positions)
    arr_rho_integ = model.render_column_integ("rho", "f64", rays)
    arr_sigma_avg = compute_avg_sigma_profile(model, 32, x_list)

    arr_sigma = np.zeros_like(arr_rho_integ)
    for i in range(len(arr_rho_integ)):
        x, _, _ = positions[i]
        sigma = sigma_profile(x) * sigma_norm

        if x < R_in:
            sigma = 0.0
        elif x > R_out:
            sigma = 0.0
        arr_sigma[i] = sigma

    return x_list, arr_rho, arr_rho_integ, arr_sigma_avg, arr_sigma


def plot_profile(ph_dump, suffix_label):

    dump = shamrock.load_phantom_dump(ph_dump)

    dump.print_state()

    ctx = shamrock.Context()
    ctx.pdata_layout_new()
    model = shamrock.get_Model_SPH(context=ctx, vector_type="f64_3", sph_kernel="M4")

    cfg = model.gen_config_from_phantom_dump(dump)
    # Set the solver config to be the one stored in cfg
    model.set_solver_config(cfg)
    # Print the solver config
    model.get_current_config().print_status()

    model.init_scheduler(int(1e8), 1)

    model.init_from_phantom_dump(dump)

    x_list, arr_rho, arr_rho_integ, arr_sigma_avg, arr_sigma = get_profiles(model)

    H = np.array([H_profile(x) for x in x_list])

    plt.plot(x_list, arr_rho, label="rho" + suffix_label)
    # plt.plot(x_list, arr_rho_integ, label="rho integ" + suffix_label)
    # plt.plot(x_list, arr_sigma_avg, label="rho integ avg avg" + suffix_label)
    plt.plot(
        x_list,
        arr_sigma_avg / (np.sqrt(2 * np.pi) * H),
        label="sigma avg/(sqrt(2*pi)*H)" + suffix_label,
    )

    return x_list, arr_sigma, arr_rho, arr_sigma_avg


def render_profile(ph_dump, name):
    dump = shamrock.load_phantom_dump(ph_dump)
    ctx = shamrock.Context()
    ctx.pdata_layout_new()
    model = shamrock.get_Model_SPH(context=ctx, vector_type="f64_3", sph_kernel="M4")
    cfg = model.gen_config_from_phantom_dump(dump)
    model.set_solver_config(cfg)
    model.init_scheduler(int(1e8), 1)
    model.init_from_phantom_dump(dump)

    ext = R_out * 1.2
    nx = 1024
    ny = 1024

    arr_rho = model.render_cartesian_column_integ(
        "rho",
        "f64",
        center=(0.0, 0.0, 0.0),
        delta_x=(ext * 2, 0, 0.0),
        delta_y=(0.0, ext * 2, 0.0),
        nx=nx,
        ny=ny,
    )
    metadata = {"extent": [-ext, ext, -ext, ext], "time": model.get_time()}

    ext = metadata["extent"]

    dpi = 200

    # Reset the figure using the same memory as the last one
    fig = plt.figure(dpi=dpi)
    import copy

    my_cmap = matplotlib.colormaps["gist_heat"].copy()  # copy the default cmap
    my_cmap.set_bad(color="black")

    res = plt.imshow(
        arr_rho, cmap=my_cmap, origin="lower", extent=ext, norm="log", vmin=1e-6, vmax=3e-4
    )

    plt.xlabel("x")
    plt.ylabel("y")
    plt.title(f"t = {metadata['time']:0.3f} [years / 2*pi]")

    cbar = plt.colorbar(res, extend="both")
    cbar.set_label(r"$\int \rho \, \mathrm{d}z$ [code unit]")
    return fig


dpi = 200
plt.figure(dpi=dpi)
x_list, arr_sigma, arr_rho_ph, arr_sigma_avg_ph = plot_profile(
    "phantom_run/phantom/disc/disc_00001", "(phantom)"
)
x_list, arr_sigma, arr_rho_sh, arr_sigma_avg_sh = plot_profile("shamrock_disc.phdump", "(shamrock)")
plt.plot(x_list, arr_sigma, label="sigma")


H = np.array([H_profile(x) for x in x_list])
# I'm not sure why but this fit fairly well the profile
fun_weight = arr_sigma * (3) / (H / H_R) ** 0.5
plt.plot(x_list, fun_weight, label="fitted")

plt.xlabel("x")
plt.ylabel("rho")
plt.title(f"t = {time_end:0.3f} [years]")
plt.legend()
plt.ylim(1e-5, 1e-2)
plt.yscale("log")


plt.figure(dpi=dpi)

delta_rho = (np.array(arr_rho_sh) - np.array(arr_rho_ph)) / fun_weight
delta_sigma_avg = (np.array(arr_sigma_avg_sh) - np.array(arr_sigma_avg_ph)) / (
    (np.sqrt(2 * np.pi) * H) * fun_weight
)

# set to 0 when arr_sigma is 0
delta_rho[arr_sigma == 0] = 0
delta_sigma_avg[arr_sigma == 0] = 0

# null before 2
delta_rho[x_list < 2] = 0
delta_sigma_avg[x_list < 2] = 0

plt.plot(x_list, np.abs(delta_rho), label="delta rho")
plt.plot(x_list, np.abs(delta_sigma_avg), label="delta sigma avg")
plt.xlabel("x")
plt.ylabel("rho")
plt.yscale("log")
plt.title(f"t = {time_end:0.3f} [years]")
plt.legend()
# plt.ylim(1e-5,1e-2)
# plt.yscale("log")


f1 = render_profile("phantom_run/phantom/disc/disc_00001", "phantom")
f2 = render_profile("shamrock_disc.phdump", "shamrock")
plt.show()

print("max delta rho:", np.max(np.abs(delta_rho)))
print("max delta sigma avg:", np.max(np.abs(delta_sigma_avg)))

if np.max(np.abs(delta_sigma_avg)) > 0.2:
    raise ValueError("max delta sigma avg is too high")
