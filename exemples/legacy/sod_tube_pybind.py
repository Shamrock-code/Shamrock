import shamrock
import matplotlib.pyplot as plt



ctx = shamrock.Context()
ctx.pdata_layout_new()

ctx.pdata_layout_add_field("xyz",1,"f32_3")
ctx.pdata_layout_add_field("hpart",1,"f32")

#field for leapfrog integrator
ctx.pdata_layout_add_field("vxyz",1,"f32_3")
ctx.pdata_layout_add_field("axyz",1,"f32_3")
ctx.pdata_layout_add_field("axyz_old",1,"f32_3")


ctx.pdata_layout_add_field("uint",1,"f32")
ctx.pdata_layout_add_field("duint",1,"f32")
ctx.pdata_layout_add_field("duint_old",1,"f32")


#start the scheduler
ctx.init_sched(int(1e8),1)






gamma = 5./3.



rho_g = 1
rho_d = 0.125


P_g = 1
P_d = 0.1

u_g = P_g/((gamma - 1)*rho_g)
u_d = P_d/((gamma - 1)*rho_d)


pmass = -1






setup = shamrock.SetupSPH(kernel = "M4", precision = "single")
setup.init(ctx)

(xs,ys,zs) = setup.get_box_dim(1,256,24,24)
dr = 1/xs
(xs,ys,zs) = setup.get_box_dim(dr,256,24,24)

ctx.set_coord_domain_bound((-xs,-ys/2,-zs/2),(xs,ys/2,zs/2))

setup.set_boundaries("periodic")


fact = (rho_g/rho_d)**(1./3.)

setup.add_particules_fcc(ctx,dr, (-xs,-ys/2,-zs/2),(0,ys/2,zs/2))
setup.add_particules_fcc(ctx,dr*fact, (0,-ys/2,-zs/2),(xs,ys/2,zs/2))

setup.set_value_in_box(ctx, "f32", u_g, "uint",(-xs,-ys/2,-zs/2),(0,ys/2,zs/2))
setup.set_value_in_box(ctx, "f32", u_d, "uint",(0,-ys/2,-zs/2),(xs,ys/2,zs/2))



vol_b = xs*ys*zs

totmass = (rho_d*vol_b) + (rho_g*vol_b)

print("Total mass :", totmass)

setup.set_total_mass(totmass)

pmass = setup.get_part_mass()

print("Current part mass :", pmass)

for it in range(5):
    setup.update_smoothing_length(ctx)

del setup



model = shamrock.BasicSPHUinterne(kernel = "M4", precision = "single")
model.init()
model.set_cfl_cour(1e-1)
model.set_cfl_force(0.3)


print("Current part mass :", pmass)


model.set_particle_mass(pmass)
t_end = 0
nstep = 15
for i in range(nstep):
    print(f"---{i}/{nstep-1}---")
    t_end = model.simulate_until(ctx, t_end,t_end+1e-2 ,1,1,"dump_")

del model


dic = ctx.collect_data()

xyz = dic["xyz"]
vxyz = dic["vxyz"]
hpart = dic["hpart"]
uint = dic["uint"]

gamma = 5./3.

rho = pmass*(1.2/hpart)**3
P = (gamma-1) * rho *uint


fig,axs = plt.subplots(nrows=2,ncols=2,figsize=(9,6),dpi=300)

axs[0,0].scatter(xyz[:,0], vxyz[:,0],c = 'black',s=1,label = "v")
axs[1,0].scatter(xyz[:,0], uint,c = 'black',s=1,label = "u")
axs[0,1].scatter(xyz[:,0], rho,c = 'black',s=1,label = "rho")
axs[1,1].scatter(xyz[:,0], P,c = 'black',s=1,label = "P")


axs[0,0].set_ylabel(r"$v$")
axs[1,0].set_ylabel(r"$u$")
axs[0,1].set_ylabel(r"$\rho$")
axs[1,1].set_ylabel(r"$P$")

axs[0,0].set_xlabel("$x$")
axs[1,0].set_xlabel("$x$")
axs[0,1].set_xlabel("$x$")
axs[1,1].set_xlabel("$x$")

axs[0,0].set_xlim(-0.3,0.4)


axs[0,0].set_ylim(-0.1,1)

axs[1,0].set_xlim(-0.3,0.4)
axs[0,1].set_xlim(-0.3,0.4)
axs[1,1].set_xlim(-0.3,0.4)

plt.tight_layout()
plt.show()