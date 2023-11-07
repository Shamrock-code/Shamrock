import shamrock
import matplotlib.pyplot as plt
import numpy as np
import sarracen

outputdir = "/home/ylapeyre/track_bug1/KH_0ssss/"
ph_dir = "/home/ylapeyre/track_bug1/KH_0/"
ph_file = ph_dir + "kh_00000"
ev_f = ph_dir + "kh01.ev"

ev_dic = {}
with open(ev_f, 'r') as phantom_ev:
    # read the col names
    #columns = phantom_ev.readline().strip().split()
    #print(columns)

    columns = ['time', 
               'ekin',
               'etherm',
               'emag',
               'epot',
               'etot',
               'erad',
               'totmom',
               'angtot',
               'rho_max',
               'rho_avg',
               'dt',
               'totentrop',
               'rmdmzch',
               'vrms',
               'xcom',
               'ycom',
               'zcom'
               'alpha_max']
    ev_data = np.genfromtxt(ev_f, skip_header=1)
    ev_data = ev_data.T

    i_dic = 0
    for column in columns:
        ev_dic[column] = ev_data[i_dic]
        i_dic +=1

gamma = 5./3.
rho_g = 1
target_tot_u = 1


dr = 0.01

bmin = (0, 0,-0.08)
bmax = (1, 1, 0.08)
pmass = -1


#ctx = shamrock.Context()
#ctx.pdata_layout_new()
#model = shamrock.get_SPHModel(context = ctx, vector_type = "f64_3",sph_kernel = "M6")
#model.init_scheduler(int(1e7),1)
#bmin,bmax = model.get_ideal_fcc_box(dr,bmin,bmax)
xm,ym,zm = bmin
xM,yM,zM = bmax
#model.resize_simulation_box(bmin,bmax)
#model.add_cube_fcc_3d(dr, bmin,bmax)
#xc,yc,zc = model.get_closest_part_to((0,0,0))
#ctx.close_sched()
#del model
#del ctx


ctx = shamrock.Context()
ctx.pdata_layout_new()

model = shamrock.get_SPHModel(context = ctx, vector_type = "f64_3",sph_kernel = "M4")

cfg = model.gen_default_config()
#cfg.set_artif_viscosity_Constant(alpha_u = 1, alpha_AV = 1, beta_AV = 2)
#cfg.set_artif_viscosity_VaryingMM97(alpha_min = 0.1,alpha_max = 1,sigma_decay = 0.1, alpha_u = 1, beta_AV = 2)
cfg.set_artif_viscosity_VaryingCD10(alpha_min = 0.0,alpha_max = 1,sigma_decay = 0.1, alpha_u = 1, beta_AV = 2)
cfg.set_boundary_periodic()
cfg.set_eos_adiabatic(gamma)
cfg.print_status()
model.set_solver_config(cfg)

model.init_scheduler(int(1e6),1)

#bmin = (xm - xc,ym - yc, zm - zc)
#bmax = (xM - xc,yM - yc, zM - zc)
#xm,ym,zm = bmin
#xM,yM,zM = bmax

model.resize_simulation_box(bmin,bmax)


#model.add_cube_fcc_3d(dr, bmin,bmax)

sdf = sarracen.read_phantom(ph_file)
tuple_of_lists = (list(sdf['x']), list(sdf['y']), list(sdf['z']))
list_of_tuples = [tuple(item) for item in zip(*tuple_of_lists)]
model.push_particle(list_of_tuples, list(sdf['h']), list(sdf['u']))

vol_b = (xM - xm)*(yM - ym)*(zM - zm)

totmass = (rho_g*vol_b)
#print("Total mass :", totmass)

pmass = model.total_mass_to_part_mass(totmass)

#model.set_value_in_a_box("uint","f64", 0 , bmin,bmax)

rinj = 0.008909042924642563*2/2
#rinj = 0.008909042924642563*2*2
#rinj = 0.01718181
#u_inj = 1
#model.add_kernel_value("uint","f64", u_inj,(0,0,0),rinj)



#print("Current part mass :", pmass)

#for it in range(5):
#    setup.update_smoothing_length(ctx)



#print("Current part mass :", pmass)
model.set_particle_mass(pmass)


tot_u = pmass*model.get_sum("uint","f64")
#print("total u :",tot_u)

delta = 0.25
rho1 = 1.
rho2 = 2.
v1 = -0.5
v2 = 0.5
P = 2.5


def rho_map(r):
    x, y, z = rmin

    f = 1. / (1 + np.exp(2 * (y - 0.25) / delta))
    g = 1. / (1 + np.exp(2 * (0.75 - y) / delta))

    R = (1 -f) * (1 - g)

    return rho1 + R * (rho2 - rho1)

def vx(y):
    f = 1. / (1 + np.exp(2 * (y - 0.25) / delta))
    g = 1. / (1 + np.exp(2 * (0.75 - y) / delta))

    R = (1 -f) * (1 - g)
    return v1 + R * (v2 - v1)

def vy(x):
    return 0.1 * np.sin(2 * np.pi * x)

def vel_func(r) -> tuple[float,float,float]:
    x,y,z = r
    return (vx(y), vy(x), 0.)

#model.set_field_value_lambda_f64_3("rho", rho_map)

print("####################################AHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHH")
model.set_field_value_lambda_f64_3("vxyz", vel_func)

model.set_cfl_cour(0.3)
model.set_cfl_force(0.25)


i = 0
i_dump = 0
t_sum = 0
t_target = 8.
current_dt = 0.002 #ev_dic['dt'][0]
while t_sum < t_target:
#for next_dt in ev_dic['dt'][1:]:

    #print("step : t=",t_sum)
    
    next_dt = model.evolve(t_sum,current_dt, True, outputdir + "dump_"+str(i_dump)+".vtk", True)

    if i % 1 == 0:
        i_dump += 1

    t_sum += current_dt
    current_dt = next_dt

    if (t_target - t_sum) < next_dt:
        current_dt = t_target - t_sum

    i +=1


dic = ctx.collect_data()

r = np.sqrt(dic['xyz'][:,0]**2 + dic['xyz'][:,1]**2 +dic['xyz'][:,2]**2)
vr = np.sqrt(dic['vxyz'][:,0]**2 + dic['vxyz'][:,1]**2 +dic['vxyz'][:,2]**2)


hpart = dic["hpart"]
uint = dic["uint"]

gamma = 5./3.

rho = pmass*(model.get_hfact()/hpart)**3
P = (gamma-1) * rho *uint


plt.style.use('custom_style.mplstyle')
fig,axs = plt.subplots(nrows=2,ncols=2,figsize=(9,6),dpi=125)

axs[0,0].scatter(r, vr,c = 'black',s=1,label = "v")
axs[1,0].scatter(r, uint,c = 'black',s=1,label = "u")
axs[0,1].scatter(r, rho,c = 'black',s=1,label = "rho")
axs[1,1].scatter(r, P,c = 'black',s=1,label = "P")


axs[0,0].set_ylabel(r"$v$")
axs[1,0].set_ylabel(r"$u$")
axs[0,1].set_ylabel(r"$\rho$")
axs[1,1].set_ylabel(r"$P$")

axs[0,0].set_xlabel("$r$")
axs[1,0].set_xlabel("$r$")
axs[0,1].set_xlabel("$r$")
axs[1,1].set_xlabel("$r$")

axs[0,0].set_xlim(0,0.55)
axs[1,0].set_xlim(0,0.55)
axs[0,1].set_xlim(0,0.55)
axs[1,1].set_xlim(0,0.55)

plt.tight_layout()
plt.show()

