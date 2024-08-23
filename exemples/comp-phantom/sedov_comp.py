import shamrock
import matplotlib.pyplot as plt
import numpy as np

import sarracen

plt.style.use('custom_style.mplstyle')


'''
compile phantom using :

export SYSTEM=gfortran
../scripts/writemake.sh sedov > Makefile
vim Makefile
make IND_TIMESTEPS=no
make setup IND_TIMESTEPS=no

and then move the phantom & phantomsetup executable in the same folder as sedov_comp.py
'''

import os
os.chdir("../../exemples/comp-phantom")
os.system("./phantomsetup blast")
os.system("./phantom blast.in")


gamma = 5./3.
rho_g = 1
target_tot_u = 1

dr = 0.01 # number of part
bmin = (-0.6 ,-0.6,-0.6)
bmax = (0.6,0.6,0.6)

pmass = 5.7511351536505855E-006



ctx = shamrock.Context()
ctx.pdata_layout_new()

model = shamrock.get_SPHModel(context = ctx, vector_type = "f64_3",sph_kernel = "M4")

cfg = model.gen_default_config() #configuration of the solver: unit syst, art visc config, bundary config
#cfg.set_artif_viscosity_Constant(alpha_u = 1, alpha_AV = 1, beta_AV = 2)
#cfg.set_artif_viscosity_VaryingMM97(alpha_min = 0.1,alpha_max = 1,sigma_decay = 0.1, alpha_u = 1, beta_AV = 2)
cfg.set_artif_viscosity_VaryingCD10(alpha_min = 0.0,alpha_max = 1,sigma_decay = 0.1, alpha_u = 1, beta_AV = 2)# cullen dennen 2010
cfg.set_eos_adiabatic(gamma)
cfg.print_status()
model.set_solver_config(cfg)

model.init_scheduler(int(1e6),1)

xm,ym,zm = bmin
xM,yM,zM = bmax

model.resize_simulation_box(bmin,bmax)


def push_dump_state(phantom_dump):
    global model

    sdf = sarracen.read_phantom(phantom_dump)
    tuple_of_lists = (list(sdf['x']), list(sdf['y']), list(sdf['z']))
    list_of_tuples = [tuple(item) for item in zip(*tuple_of_lists)]
    model.push_particle(list_of_tuples, list(sdf['h']), list(sdf['u']))


push_dump_state("../../exemples/comp-phantom/blast_00000")

vol_b = (xM - xm)*(yM - ym)*(zM - zm)

totmass = (rho_g*vol_b)
print("Total mass :", totmass)

#pmass = model.total_mass_to_part_mass(totmass)

model.set_particle_mass(pmass)

model.set_cfl_cour(0.3)
model.set_cfl_force(0.25)



smarker = 1
marker_ph = "o"
marker_sham = "o"

def load_phantom_dudt(fig,axs,phantom_dump, phantom_dumpm1):

    sdf = sarracen.read_phantom(phantom_dump)

    print(sdf)

    x = np.array(list(sdf['x']))
    y = np.array(list(sdf['y']))
    z = np.array(list(sdf['z']))

    r = np.sqrt(x**2 + y**2 + z**2)
    u = np.array(list(sdf['u']))


    sdf = sarracen.read_phantom(phantom_dumpm1)

    um1 = np.array(list(sdf['u']))

    dudt =  1e5*(u-um1)

    axs[0,1].scatter(r,dudt, s=2)




def plot_phantom_dump(fig,axs,phantom_dump):

    sdf = sarracen.read_phantom(phantom_dump)

    print(sdf)

    x = np.array(list(sdf['x']))
    y = np.array(list(sdf['y']))
    z = np.array(list(sdf['z']))

    r = np.sqrt(x**2 + y**2 + z**2)


    vx = np.array(list(sdf['vx']))
    vy = np.array(list(sdf['vy']))
    vz = np.array(list(sdf['vz']))

    vr = np.sqrt(vx**2 + vy**2 + vz**2)

    h = np.array(list(sdf['h']))

    rho = pmass*(1.2/h)**3

    u = np.array(list(sdf['u']))
    alpha = np.array(list(sdf['alpha']))

    axs[0,0].scatter(r,rho,s=smarker,c = 'red', marker=marker_ph, rasterized=True,label = "phantom")
    axs[0,1].scatter(r,u,s=smarker,c = 'red', marker=marker_ph, rasterized=True)
    axs[1,0].scatter(r,vr,s=smarker,c = 'red', marker=marker_ph, rasterized=True)
    axs[1,1].scatter(r,alpha,s=smarker,c = 'red', marker=marker_ph, rasterized=True)


last_u = ""


def comp_state(i, savename ):

    global last_u


    dic = ctx.collect_data()

    r = np.sqrt(dic['xyz'][:,0]**2 + dic['xyz'][:,1]**2 +dic['xyz'][:,2]**2)
    vr = np.sqrt(dic['vxyz'][:,0]**2 + dic['vxyz'][:,1]**2 +dic['vxyz'][:,2]**2)

    hpart = dic["hpart"]
    uint = dic["uint"]
    alpha_AV = dic["alpha_AV"]
    rho = pmass*(model.get_hfact()/hpart)**3



    #if i == 0:
    #    last_u = uint
    #    return

    phantom_dump = "../../exemples/comp-phantom/blast_{:05}".format(i)
    last_dump = "../../exemples/comp-phantom/blast_{:05}".format(i-1)


    fig,axs = plt.subplots(nrows=2,ncols=2,figsize=(9,6),dpi=125)

    plot_phantom_dump(fig,axs,phantom_dump)

    #load_phantom_dudt(fig,axs, phantom_dump, last_dump)


    axs[0,0].scatter(r,rho,s=smarker,c = 'black', marker=marker_sham, rasterized=True,label = "shamrock")
    #axs[0,1].scatter(r,1e5*(uint-last_u),s=smarker)
    axs[0,1].scatter(r,uint,s=smarker,c = 'black', marker=marker_sham, rasterized=True)
    axs[1,0].scatter(r,vr,s=smarker,c = 'black', marker=marker_sham, rasterized=True)
    axs[1,1].scatter(r,alpha_AV, s=smarker,c = 'black', marker=marker_sham, rasterized=True)


    last_u = uint

    axs[0,0].set_ylabel(r"$\rho$")
    axs[1,0].set_ylabel(r"$vr$")
    axs[0,1].set_ylabel(r"$u$")
    axs[1,1].set_ylabel(r"$\alpha$")

    axs[0,0].set_xlabel("$r$")
    axs[1,0].set_xlabel("$r$")
    axs[0,1].set_xlabel("$r$")
    axs[1,1].set_xlabel("$r$")

    axs[0,0].set_xlim(0,0.4)
    axs[1,0].set_xlim(0,0.4)
    axs[0,1].set_xlim(0,0.4)
    axs[1,1].set_xlim(0,0.4)

    axs[0,0].legend()


    plt.tight_layout()

    plt.savefig(savename, dpi = 300)


model.evolve(0,0, False, "", False)
for i in range(1001):
    if i%10 == 0:
        comp_state(i, "comp"+str(i)+".pdf")
    model.evolve(0,1e-5, False, "", False)

for i in range(1000, 1001):
    if i%10 == 0:
        comp_state(i, "comp"+str(i)+".pdf")
    model.evolve(0,1e-5, False, "", False)
