import shamrock
import matplotlib.pyplot as plt



ctx = shamrock.Context()
ctx.pdata_layout_new()


sim = shamrock.BasicGasSPH(ctx)
sim.setup_fields()


#start the scheduler
ctx.init_sched(int(1e7),1)




gamma = 5./3.

rho_g = 1
rho_d = 0.125


P_g = 1
P_d = 0.1

u_g = P_g/((gamma - 1)*rho_g)
u_d = P_d/((gamma - 1)*rho_d)


pmass = -1






setup = shamrock.SetupSPH(kernel = "M4", precision = "double")
setup.init(ctx)

(xs,ys,zs) = setup.get_box_dim(1,256,24,24)
dr = 1/xs
(xs,ys,zs) = setup.get_box_dim(dr,256,24,24)

ctx.set_coord_domain_bound((-xs,-ys/2,-zs/2),(xs,ys/2,zs/2))

setup.set_boundaries("periodic")


fact = (rho_g/rho_d)**(1./3.)

setup.add_particules_fcc(ctx,dr, (-xs,-ys/2,-zs/2),(0,ys/2,zs/2))
setup.add_particules_fcc(ctx,dr*fact, (0,-ys/2,-zs/2),(xs,ys/2,zs/2))

setup.set_value_in_box(ctx, "f64", u_g, "uint",(-xs,-ys/2,-zs/2),(0,ys/2,zs/2))
setup.set_value_in_box(ctx, "f64", u_d, "uint",(0,-ys/2,-zs/2),(xs,ys/2,zs/2))



vol_b = xs*ys*zs

totmass = (rho_d*vol_b) + (rho_g*vol_b)

print("Total mass :", totmass)

setup.set_total_mass(totmass)

pmass = setup.get_part_mass()

print("Current part mass :", pmass)

#for it in range(5):
#    setup.update_smoothing_length(ctx)

del setup



sim.set_cfl_cour(0.25)
sim.set_cfl_force(0.3)


print("Current part mass :", pmass)


sim.set_particle_mass(pmass)


t_sum = 0
t_target = 0.2
current_dt = 1e-7
i = 0
i_dump = 0
while t_sum < t_target:

    print("step : t=",t_sum)
    
    next_dt = model.evolve(current_dt, True, "dump_"+str(i_dump)+".vtk", True)

    if i % 1 == 0:
        i_dump += 1

    t_sum += current_dt
    current_dt = next_dt

    if (t_target - t_sum) < next_dt:
        current_dt = t_target - t_sum

    i+= 1