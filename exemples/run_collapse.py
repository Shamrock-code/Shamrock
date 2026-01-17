import glob
import json
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import numpy

import shamrock

# Particle tracking is an experimental feature

shamrock.enable_experimental_features()

si = shamrock.UnitSystem()
sicte = shamrock.Constants(si)
codeu = shamrock.UnitSystem(
    unit_time=sicte.year(),
    unit_length=sicte.parsec(),
    unit_mass=sicte.sol_mass(),
)
ucte = shamrock.Constants(codeu)

m_s_codeu = codeu.get("m") * codeu.get("s", power=-1)
kg_m3_codeu = codeu.get("kg") * codeu.get("m", power=-3)

print(f"m_s_codeu: {kg_m3_codeu}")

cs = 190.0  # m/s
rho_g = kg_m3_codeu * 1.e-18
initial_u = 1.0
Npart = int(0.1e6)
ampl = 3

sim_radius = 2


kb = ucte.kb()
print(f"kb: {kb}")
mu = 2.375
mh = 1.00784 * ucte.dalton()
print(f"mu * mh * kb: {mu * mh * kb}")

cs = cs * m_s_codeu  # m/s
rho_c1 = 1.92e-13 * 1000 * kg_m3_codeu  # g/cm^3 -> kg/m^3
rho_c2 = 3.84e-8 * 1000 * kg_m3_codeu  # g/cm^3 -> kg/m^3
rho_c3 = 1.92e-3 * 1000 * kg_m3_codeu  # g/cm^3 -> kg/m^3

sim_directory = f"collapse5_{Npart}"
import os

os.makedirs(sim_directory, exist_ok=True)


tsound = sim_radius / cs
tff = shamrock.phys.free_fall_time(rho_g, codeu)

P_init, _cs_init, T_init = shamrock.phys.eos.eos_Machida06(
        cs=cs, rho=rho_g, rho_c1=rho_c1, rho_c2=rho_c2, rho_c3=rho_c3, mu=mu, mh=mh, kb=kb
    )

print("---------------------------------------------------")
print(f"Npart                         : {Npart}")
print(f"sim_radius                    : {sim_radius} [pc]")
print(f"total mass                    : {rho_g * (2*sim_radius)**3:.3e} [solar mass]")
print(f"rho                           : {rho_g / kg_m3_codeu:.3e} [kg/m^3]")
print(f"P                             : {P_init / codeu.get("Pa"):.3e} [Pa]")
print(f"T                             : {T_init:.3e} [K]")
print(f"cs                            : {cs / m_s_codeu:.3e} [m/s]")
print(f"tsound (= R/cs)               : {tsound:9.1f} [years]")
print(f"tff (= sqrt(3*pi/(32*G*rho))) : {tff:9.1f} [years]")
print(f"tff/tsound                    : {tff/tsound:.4f} (<1 = collapse)")
print("---------------------------------------------------")










class rho_xy_plot:
    def __init__(self, model, center, ext_r, nx,ny, analysis_folder, analysis_prefix):
        self.model = model
        self.center = center
        self.ext_r = ext_r
        self.nx = nx
        self.ny = ny

        self.analysis_prefix = os.path.join(analysis_folder, analysis_prefix) + "_"
        self.plot_prefix = os.path.join(analysis_folder, "plot_" + analysis_prefix) + "_"

        self.npy_data_filename = self.analysis_prefix + "{:07}.npy"
        self.json_data_filename = self.analysis_prefix + "{:07}.json"
        self.json_glob = self.analysis_prefix + "*.json"
        self.plot_filename = self.plot_prefix + "{:07}.png"
        self.img_glob = self.analysis_prefix + "*.png"

    def compute_rho_xy(self):
        
        arr_rho_xy = model.render_cartesian_column_integ(
            "rho",
            "f64",
            center=self.center,
            delta_x=(self.ext_r * 2, 0, 0.0),
            delta_y=(0.0, self.ext_r * 2, 0.0),
            nx=self.nx,
            ny=self.ny,
        )
        
        return arr_rho_xy

    def analysis_save(self, iplot):
        arr_rho_xy = self.compute_rho_xy()
        arr_rho_xy /= kg_m3_codeu
        if shamrock.sys.world_rank() == 0:
            cx,cy,cz = self.center
            metadata = {"extent": [cx-self.ext_r, cx+self.ext_r, cy-self.ext_r, cy+self.ext_r], "time": self.model.get_time()}
            np.save(self.npy_data_filename.format(iplot), arr_rho_xy)

            with open(self.json_data_filename.format(iplot), "w") as fp:
                json.dump(metadata, fp)

    def load_analysis(self, iplot):
        with open(self.json_data_filename.format(iplot), "r") as fp:
            metadata = json.load(fp)
        return np.load(self.npy_data_filename.format(iplot)), metadata

    def plot_rho_xy(self, iplot):
        arr_rho_xy, metadata = self.load_analysis(iplot)
        if shamrock.sys.world_rank() == 0:

            # Reset the figure using the same memory as the last one
            plt.figure(num=1, clear=True, dpi=200)

            import copy

            my_cmap = matplotlib.colormaps["inferno"].copy()  # copy the default cmap
            my_cmap.set_bad(color="black")

            min_val = np.min(arr_rho_xy)
            if min_val < 1e-20:
                min_val = 1e-20

            v_ext = 0.04
            res = plt.imshow(
                arr_rho_xy,
                cmap=my_cmap,
                origin="lower",
                extent=metadata["extent"],
                norm="log",
                vmin= min_val
            )

            plt.xlabel("x [pc]")
            plt.ylabel("y [pc]")
            plt.title("t = {:0.3f} [Year]".format(metadata["time"]))

            cbar = plt.colorbar(res, extend="both")
            cbar.set_label(r"$\int \rho \, \mathrm{d} z$ [kg.m$^{-2}$]")

            plt.savefig(self.plot_filename.format(iplot))
            plt.close()

    def get_list_dumps_id(self):

        list_files = glob.glob(self.json_glob)
        list_files.sort()
        list_json_files = []
        for f in list_files:
            list_json_files.append(int(f.split("_")[-1].split(".")[0]))
        return list_json_files

    def render_all(self):
        if shamrock.sys.world_rank() == 0:
            for iplot in self.get_list_dumps_id():
                print("Rendering rho xy plot for dump", iplot)
                self.plot_rho_xy(iplot)



class v_slice_xy_plot:
    def __init__(self, model, center, ext_r, nx,ny, analysis_folder, analysis_prefix):
        self.model = model
        self.center = center
        self.ext_r = ext_r
        self.nx = nx
        self.ny = ny

        self.analysis_prefix = os.path.join(analysis_folder, analysis_prefix) + "_"
        self.plot_prefix = os.path.join(analysis_folder, "plot_" + analysis_prefix) + "_"

        self.npy_data_filename = self.analysis_prefix + "{:07}.npy"
        self.json_data_filename = self.analysis_prefix + "{:07}.json"
        self.json_glob = self.analysis_prefix + "*.json"
        self.plot_filename = self.plot_prefix + "{:07}.png"
        self.img_glob = self.analysis_prefix + "*.png"

    def compute_v_slice(self):
        
        arr_v_slice = model.render_cartesian_slice(
            "vxyz",
            "f64_3",
            center=self.center,
            delta_x=(self.ext_r * 2, 0, 0.0),
            delta_y=(0.0, self.ext_r * 2, 0.0),
            nx=self.nx,
            ny=self.ny,
        )
        
        return arr_v_slice

    def analysis_save(self, iplot):
        arr_v_slice = self.compute_v_slice()
        arr_v_slice /= cs
        if shamrock.sys.world_rank() == 0:
            cx,cy,cz = self.center
            metadata = {"extent": [cx-self.ext_r, cx+self.ext_r, cy-self.ext_r, cy+self.ext_r], "time": self.model.get_time()}
            np.save(self.npy_data_filename.format(iplot), arr_v_slice)

            with open(self.json_data_filename.format(iplot), "w") as fp:
                json.dump(metadata, fp)

    def load_analysis(self, iplot):
        with open(self.json_data_filename.format(iplot), "r") as fp:
            metadata = json.load(fp)
        return np.load(self.npy_data_filename.format(iplot)), metadata

    def plot_v_slice(self, iplot):
        arr_v_slice, metadata = self.load_analysis(iplot)

        v_norm = np.sqrt(arr_v_slice[:, :, 0] ** 2 + arr_v_slice[:, :, 1] ** 2 + arr_v_slice[:, :, 2] ** 2)
        if shamrock.sys.world_rank() == 0:

            # Reset the figure using the same memory as the last one
            plt.figure(num=1, clear=True, dpi=200)

            import copy

            my_cmap = matplotlib.colormaps["inferno"].copy()  # copy the default cmap
            my_cmap.set_bad(color="black")

            v_ext = 0.04
            res = plt.imshow(
                v_norm,
                cmap=my_cmap,
                origin="lower",
                extent=metadata["extent"]
            )

            plt.xlabel("x [pc]")
            plt.ylabel("y [pc]")
            plt.title("t = {:0.3f} [Year]".format(metadata["time"]))

            cbar = plt.colorbar(res, extend="both")
            cbar.set_label(r"$||\mathbf{v}||/c_s$ ")

            plt.savefig(self.plot_filename.format(iplot))
            plt.close()

    def get_list_dumps_id(self):

        list_files = glob.glob(self.json_glob)
        list_files.sort()
        list_json_files = []
        for f in list_files:
            list_json_files.append(int(f.split("_")[-1].split(".")[0]))
        return list_json_files

    def render_all(self):
        if shamrock.sys.world_rank() == 0:
            for iplot in self.get_list_dumps_id():
                print("Rendering v slice plot for dump", iplot)
                self.plot_v_slice(iplot)





dump_prefix = sim_directory + "/dump_"
def get_dump_name(idump):
    return dump_prefix + f"{idump:07}" + ".sham"


def get_last_dump():
    res = glob.glob(dump_prefix + "*.sham")

    num_max = -1

    for f in res:
        try:
            dump_num = int(f[len(dump_prefix) : -5])
            if dump_num > num_max:
                num_max = dump_num
        except ValueError:
            pass

    if num_max == -1:
        return None
    else:
        return num_max


def purge_old_dumps():
    if shamrock.sys.world_rank() == 0:
        res = glob.glob(dump_prefix + "*.sham")
        res.sort()

        # The list of dumps to remove (keep the first and last 3 dumps)
        to_remove = res[1:-3]

        for f in to_remove:
            os.remove(f)



bmin = (-sim_radius, -sim_radius, -sim_radius)
bmax = (sim_radius, sim_radius, sim_radius)


scheduler_split_val = int(2e7)
scheduler_merge_val = int(1)


N_target = Npart
xm, ym, zm = bmin
xM, yM, zM = bmax
vol_b = (xM - xm) * (yM - ym) * (zM - zm)

if shamrock.sys.world_rank() == 0:
    print("Npart", Npart)
    print("scheduler_split_val", scheduler_split_val)
    print("scheduler_merge_val", scheduler_merge_val)
    print("N_target", N_target)
    print("vol_b", vol_b)

part_vol = vol_b / N_target

# lattice volume
part_vol_lattice = 0.74 * part_vol

dr = (part_vol_lattice / ((4.0 / 3.0) * 3.1416)) ** (1.0 / 3.0)


pmass = -1

ctx = shamrock.Context()
ctx.pdata_layout_new()

model = shamrock.get_Model_SPH(context=ctx, vector_type="f64_3", sph_kernel="M4")


def is_in_sphere(pt):
    x, y, z = pt
    return (x**2 + y**2 + z**2) < sim_radius*sim_radius

def config_model():
    cfg = model.gen_default_config()
    # cfg.set_artif_viscosity_Constant(alpha_u = 1, alpha_AV = 1, beta_AV = 2)
    # cfg.set_artif_viscosity_VaryingMM97(alpha_min = 0.1,alpha_max = 1,sigma_decay = 0.1, alpha_u = 1, beta_AV = 2)
    cfg.set_artif_viscosity_VaryingCD10(
        alpha_min=0.0, alpha_max=1, sigma_decay=0.1, alpha_u=1, beta_AV=2
    )
    #cfg.set_boundary_periodic()
    cfg.set_eos_machida06(rho_c1=rho_c1, rho_c2=rho_c2, rho_c3=rho_c3, cs=cs, mu=mu, mh=mh, kb=kb)

    cfg.set_self_gravity_fmm(order=5, opening_angle=0.5)

    cfg.set_units(codeu)
    #cfg.print_status()
    model.set_solver_config(cfg)
    model.init_scheduler(scheduler_split_val, scheduler_merge_val)


    bsim_min = (-sim_radius*2, -sim_radius*2, -sim_radius*2)
    bsim_max = (sim_radius*2, sim_radius*2, sim_radius*2)
    model.resize_simulation_box(bsim_min, bsim_max)
    cfg.add_kill_sphere(center=(0, 0, 0), radius=sim_radius*2)  # kill particles outside the simulation box

    setup = model.get_setup()
    gen = setup.make_generator_lattice_hcp(dr, bmin, bmax)

    thesphere = setup.make_modifier_filter(parent=gen, filter=is_in_sphere)

    # On aurora /2 was correct to avoid out of memory
    setup.apply_setup(thesphere, insert_step=int(scheduler_split_val / 2))

    vol_b = (xM - xm) * (yM - ym) * (zM - zm)
    totmass = rho_g * vol_b
    pmass = model.total_mass_to_part_mass(totmass)
    model.set_particle_mass(pmass)

    model.set_value_in_a_box("uint", "f64", initial_u, bmin, bmax)

    model.set_cfl_cour(0.1)
    model.set_cfl_force(0.1)
    
    # if corrector is trigger CFL will become normal again after 1000 steps
    model.set_cfl_mult_stiffness(5) 


def TurbField(res=256, minmode=2, maxmode=64, sol_weight=1.0, seed=42):
    from scipy import  fftpack

    freqs = fftpack.fftfreq(res)
    freq3d = numpy.array(numpy.meshgrid(freqs, freqs, freqs, indexing="ij"))
    intfreq = numpy.around(freq3d * res)
    kSqr = numpy.sum(numpy.abs(freq3d) ** 2, axis=0)
    intkSqr = numpy.sum(numpy.abs(intfreq) ** 2, axis=0)
    VK = []

    # apply ~k^-2 exp(-k^2/kmax^2) filter to white noise to get x, y, and z components of velocity field
    for i in range(3):
        numpy.random.seed(seed + i)
        rand_phase = fftpack.fftn(
            numpy.random.normal(size=kSqr.shape)
        )  # fourier transform of white noise
        vk = rand_phase * (float(minmode) / res) ** 2 / (kSqr + 1e-300)
        vk[intkSqr == 0] = 0.0
        vk[intkSqr < minmode**2] *= (
            intkSqr[intkSqr < minmode**2] ** 2 / minmode**4
        )  # smoother filter than mode-freezing; should give less "ringing" artifacts
        vk *= numpy.exp(-intkSqr / maxmode**2)

        VK.append(vk)
    VK = numpy.array(VK)
    # bin = numpy.logspace(0,2.5,50)
    # plt.hist(vk.flatten(),bins=bin)
    # #plt.xlim(0,10**2.)
    # plt.xscale("log")
    # plt.yscale("log")
    # plt.show()
    # plt.imshow(vk[:,25,:].real)
    # plt.show()
    vk_new = numpy.zeros_like(VK)

    # do projection operator to get the correct mix of compressive and solenoidal
    for i in range(3):
        for j in range(3):
            if i == j:
                vk_new[i] += sol_weight * VK[j]
            vk_new[i] += (
                (1 - 2 * sol_weight) * freq3d[i] * freq3d[j] / (kSqr + 1e-300) * VK[j]
            )
    vk_new[:, kSqr == 0] = 0.0
    VK = vk_new

    vel = numpy.array(
        [fftpack.ifftn(vk).real for vk in VK]
    )  # transform back to real space
    vel -= numpy.average(vel, axis=(1, 2, 3))[:, numpy.newaxis, numpy.newaxis, numpy.newaxis]
    vel = vel / numpy.sqrt(numpy.sum(vel**2, axis=0).mean())  # normalize so that RMS is 1
    return numpy.array(vel,dtype='f4')




seed = 42

#avx,avy,avz = make_turb_field(res,power,seed)
vx,vy,vz = TurbField(128,2,64,0.7,seed)

print(f"vx shape: {vx.shape}, dtype: {vx.dtype}")
print(f"vy shape: {vy.shape}, dtype: {vy.dtype}")
print(f"vz shape: {vz.shape}, dtype: {vz.dtype}")
print(f"vx min/max: {vx.min():.6f}/{vx.max():.6f}")
print(f"vy min/max: {vy.min():.6f}/{vy.max():.6f}")
print(f"vz min/max: {vz.min():.6f}/{vz.max():.6f}")

# Set global velocity fields
vx_global = vx
vy_global = vy
vz_global = vz
domain_size_global = 1.0
sim_bmin = bmin  # (-0.5, -0.5, -0.5)
sim_bmax = bmax  # (0.5, 0.5, 0.5)

res = vx_global.shape[0]

# Create coordinate arrays for the grid
coords = numpy.linspace(0, domain_size_global, res)

# Create interpolators for each velocity component
from scipy.interpolate import RegularGridInterpolator
interp_vx = RegularGridInterpolator((coords, coords, coords), vx_global, 
                                        bounds_error=False, fill_value=0.0)
interp_vy = RegularGridInterpolator((coords, coords, coords), vy_global, 
                                        bounds_error=False, fill_value=0.0)
interp_vz = RegularGridInterpolator((coords, coords, coords), vz_global, 
                                        bounds_error=False, fill_value=0.0)


def vel_field(pos):
    """
    Interpolate velocity at position (x, y, z) using global velocity fields.
    
    Parameters:
    -----------
    pos : tuple
        (x, y, z) position in simulation coordinates
    
    Returns:
    --------
    tuple
        (vx, vy, vz) velocity components at the given position(s)
    """
    
    
    global interp_vx, interp_vy, interp_vz, domain_size_global, sim_bmin, sim_bmax, ampl, cs
    
    x, y, z = pos
    
    # Transform from simulation coordinates to velocity field coordinates [0, 1]
    x_vf = (x - sim_bmin[0]) / (sim_bmax[0] - sim_bmin[0])
    y_vf = (y - sim_bmin[1]) / (sim_bmax[1] - sim_bmin[1])
    z_vf = (z - sim_bmin[2]) / (sim_bmax[2] - sim_bmin[2])
    
    
    
    points = numpy.column_stack([numpy.atleast_1d(x_vf), 
                                  numpy.atleast_1d(y_vf), 
                                  numpy.atleast_1d(z_vf)])
    
    vx_interp = interp_vx(points)[0]
    vy_interp = interp_vy(points)[0]
    vz_interp = interp_vz(points)[0]

    #print(f"vx_interp = {vx_interp}, vy_interp = {vy_interp}, vz_interp = {vz_interp}")
    
    return vx_interp*ampl *cs, vy_interp*ampl *cs, vz_interp*ampl *cs

# Example: Interpolate velocity at a specific position (in simulation coordinates)
test_pos = (0.0, 0.0, 0.0)  # Center of simulation domain
vel_x, vel_y, vel_z = vel_field(test_pos)
print(f"\nVelocity at {test_pos} (sim coords):")
print(f"  vx = {vel_x:.6f}")
print(f"  vy = {vel_y:.6f}")
print(f"  vz = {vel_z:.6f}")

# Test at edge of domain
test_pos2 = (0.25, -0.25, 0.1)
vel_x2, vel_y2, vel_z2 = vel_field(test_pos2)
print(f"\nVelocity at {test_pos2} (sim coords):")
print(f"  vx = {vel_x2:.6f}")
print(f"  vy = {vel_y2:.6f}")
print(f"  vz = {vel_z2:.6f}")

def setup_particles():

    model.set_field_value_lambda_f64_3("vxyz", vel_field)

    model.timestep()


rho_plot_large = rho_xy_plot(model, (0.0, 0.0, 0.0), sim_radius *1.2, 2048, 2048, sim_directory, "rho_xy")
rho_plot_mid = rho_xy_plot(model, (0.0, 0.0, 0.0), sim_radius /1.2, 2048, 2048, sim_directory, "rho_mid")
rho_plot_zoom = rho_xy_plot(model, (0.0, 0.0, 0.0), 0.25, 2048, 2048, sim_directory, "rho_zoom")

v_slice_plot = v_slice_xy_plot(model, (0.0, 0.0, 0.0), sim_radius *1.2, 2048, 2048, sim_directory, "v_slice")

def analysis(i):
    
    rho_plot_large.analysis_save(i)
    rho_plot_mid.analysis_save(i)
    v_slice_plot.analysis_save(i)

    dat = ctx.collect_data()

    #get index with max rho
    max_rho_index = np.argmin(dat["hpart"])
    hpart_min = dat["hpart"][max_rho_index]
    max_rho = model.get_particle_mass() * (model.get_hfact() / hpart_min) ** 3

    max_rho_x = dat["xyz"][max_rho_index, 0]
    max_rho_y = dat["xyz"][max_rho_index, 1]
    max_rho_z = dat["xyz"][max_rho_index, 2]
    print(f"max rho: {max_rho/kg_m3_codeu:.3e} at ({max_rho_x:.3e}, {max_rho_y:.3e}, {max_rho_z:.3e})")

    #render around max rho
    rho_plot_zoom.center = (max_rho_x, max_rho_y, max_rho_z)
    
    rho_plot_zoom.analysis_save(i)


idump_last_dump = get_last_dump()

if shamrock.sys.world_rank() == 0:
    print("Last dump:", idump_last_dump)

# %%
# Load the last dump if it exists, setup otherwise

run_simulation = "1"
i = 0
next_t = 0

if run_simulation == "1" and idump_last_dump is not None:
    model.load_from_dump(get_dump_name(idump_last_dump))
    i = idump_last_dump + 1
    next_t =  model.get_time() + tff * 0.1
elif run_simulation == "1":
    config_model()
    setup_particles()


rho_plot_large.render_all()
rho_plot_mid.render_all()
rho_plot_zoom.render_all()
v_slice_plot.render_all()

while i < 10000:

    model.evolve_until(next_t, niter_max=250)

    next_t = model.get_time() + tff * 1e-2

    analysis(i)
    model.dump(get_dump_name(i))
    purge_old_dumps()

    if i % 2 == 0:
        rho_plot_large.render_all()
        rho_plot_mid.render_all()
        rho_plot_zoom.render_all()
        v_slice_plot.render_all()
    i += 1


rho_plot_large.render_all()
rho_plot_mid.render_all()
rho_plot_zoom.render_all()
v_slice_plot.render_all()