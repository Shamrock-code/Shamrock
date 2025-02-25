import shamrock
import numpy as np
import matplotlib.pyplot as plt
import os


ctx = shamrock.Context()
ctx.pdata_layout_new()

model = shamrock.get_Model_Zeus(
    context = ctx,
    vector_type = "f64_3",
    grid_repr = "i64_3")

model.init_scheduler(int(1e7),1)

multx = 1
multy = 1
multz = 1

sz = 1 << 1
base = 64
#model.make_base_grid((0,0,0),(sz,sz,sz),(base*multx,base*multy,base*multz))

cfg = model.gen_default_config()
scale_fact = 1/(sz*base*multx)
cfg.set_scale_factor(scale_fact)
model.set_config(cfg)

file_data = "ghost_dump_debug0.000000patch_0000.txt"
block_size = 8

block_min = []
block_max = []
cell_rho = []
cell_eint = []
cell_vel = []

cell_force_press = []

cell_eint_post_source = []
cell_vel_post_source = []

cell_eint_start_transp = []
cell_vel_start_transp = []

cell_rho_end_transp = []
cell_eint_end_transp = []
cell_vel_end_transp = []

cell_divv_source = []

cell_vel_n_xp = []

cell_Q = []
cell_ax = []
cell_ay = []
cell_az = []
cell_Qstar_x = []
cell_Qstar_y = []
cell_Qstar_z = []
cell_Flux_x = []
cell_Flux_y = []
cell_Flux_z = []

f = open(file_data,"r")
lines = f.readlines()

active_field = ""
for line in lines:
    if line.startswith("-->"):
        if line == "--> cell_min type=i64_3\n":
            active_field = "cell_min"
        elif line == "--> cell_max type=i64_3\n":
            active_field = "cell_max"
        elif line == "--> rho type=f64\n":
            active_field = "rho"
        elif line == "--> eint type=f64\n":
            active_field = "eint"
        elif line == "--> vel type=f64_3\n":
            active_field = "vel"


        elif line == "--> eint_post_source type=f64\n":
            active_field = "eint_post_source"
        elif line == "--> vel_post_source type=f64_3\n":
            active_field = "vel_post_source"

        elif line == "--> eint_start_transp type=f64\n":
            active_field = "eint_start_transp"
        elif line == "--> vel_start_transp type=f64_3\n":
            active_field = "vel_start_transp"


        elif line == "--> rho_end_transp type=f64\n":
            active_field = "rho_end_transp"
        elif line == "--> eint_end_transp type=f64\n":
            active_field = "eint_end_transp"
        elif line == "--> vel_end_transp type=f64_3\n":
            active_field = "vel_end_transp"

        elif line == "--> divv_source type=f64\n":
            active_field = "divv_source"

        elif line == "--> vel_n_xp type=f64_3\n":
            active_field = "vel_n_xp"

        elif line == "--> Q type=f64_8\n":
            active_field = "Q"
        elif line == "--> ax type=f64_8\n":
            active_field = "ax"
        elif line == "--> ay type=f64_8\n":
            active_field = "ay"
        elif line == "--> az type=f64_8\n":
            active_field = "az"

        elif line == "--> Qstar_x type=f64_8\n":
            active_field = "Qstar_x"
        elif line == "--> Qstar_y type=f64_8\n":
            active_field = "Qstar_y"
        elif line == "--> Qstar_z type=f64_8\n":
            active_field = "Qstar_z"

        elif line == "--> Flux_x type=f64_8\n":
            active_field = "Flux_x"
        elif line == "--> Flux_y type=f64_8\n":
            active_field = "Flux_y"
        elif line == "--> Flux_z type=f64_8\n":
            active_field = "Flux_z"

        elif line == "--> force_press type=f64_3\n":
            active_field = "force_press"

        elif line == "--> Nobj_original type=u32\n":
            active_field = "nobj_or"
        elif line == "--> Nobj_total type=u32\n":
            active_field = "nobj_tot"
        else:
            print("unregistered field ", line)
            raise "error"

    else:
        if active_field == "cell_min":
            splt = (line[:-1].split())
            x = int(splt[0])
            y = int(splt[1])
            z = int(splt[2])
            block_min.append((x,y,z))
        elif active_field == "cell_max":
            splt = (line[:-1].split())
            x = int(splt[0])
            y = int(splt[1])
            z = int(splt[2])
            block_max.append((x,y,z))
        elif active_field == "rho":
            val = float(line[:-1])
            cell_rho.append(val)
        elif active_field == "eint":
            val = float(line[:-1])
            cell_eint.append(val)
        elif active_field == "vel":
            splt = (line[:-1].split())
            x = float(splt[0])
            y = float(splt[1])
            z = float(splt[2])
            cell_vel.append((x,y,z))

        elif active_field == "force_press":
            splt = (line[:-1].split())
            x = float(splt[0])
            y = float(splt[1])
            z = float(splt[2])
            cell_force_press.append((x,y,z))

        elif active_field == "eint_post_source":
            val = float(line[:-1])
            cell_eint_post_source.append(val)
        elif active_field == "vel_post_source":
            splt = (line[:-1].split())
            x = float(splt[0])
            y = float(splt[1])
            z = float(splt[2])
            cell_vel_post_source.append((x,y,z))

        elif active_field == "eint_start_transp":
            val = float(line[:-1])
            cell_eint_start_transp.append(val)
        elif active_field == "vel_start_transp":
            splt = (line[:-1].split())
            x = float(splt[0])
            y = float(splt[1])
            z = float(splt[2])
            cell_vel_start_transp.append((x,y,z))

        elif active_field == "rho_end_transp":
            val = float(line[:-1])
            cell_rho_end_transp.append(val)
        elif active_field == "eint_end_transp":
            val = float(line[:-1])
            cell_eint_end_transp.append(val)
        elif active_field == "vel_end_transp":
            splt = (line[:-1].split())
            x = float(splt[0])
            y = float(splt[1])
            z = float(splt[2])
            cell_vel_end_transp.append((x,y,z))


        elif active_field == "divv_source":
            val = float(line[:-1])
            cell_divv_source.append(val)

        elif active_field == "vel_n_xp":
            splt = (line[:-1].split())
            x = float(splt[0])
            y = float(splt[1])
            z = float(splt[2])
            cell_vel_n_xp.append((x,y,z))

        elif active_field == "Q":
            splt = (line[:-1].split())
            s0 = float(splt[0])
            s1 = float(splt[1])
            s2 = float(splt[2])
            s3 = float(splt[3])
            s4 = float(splt[4])
            s5 = float(splt[5])
            s6 = float(splt[6])
            s7 = float(splt[7])
            cell_Q.append((s0,s1,s2,s3,s4,s5,s6,s7))
        elif active_field == "ax":
            splt = (line[:-1].split())
            s0 = float(splt[0])
            s1 = float(splt[1])
            s2 = float(splt[2])
            s3 = float(splt[3])
            s4 = float(splt[4])
            s5 = float(splt[5])
            s6 = float(splt[6])
            s7 = float(splt[7])
            cell_ax.append((s0,s1,s2,s3,s4,s5,s6,s7))
        elif active_field == "ay":
            splt = (line[:-1].split())
            s0 = float(splt[0])
            s1 = float(splt[1])
            s2 = float(splt[2])
            s3 = float(splt[3])
            s4 = float(splt[4])
            s5 = float(splt[5])
            s6 = float(splt[6])
            s7 = float(splt[7])
            cell_ay.append((s0,s1,s2,s3,s4,s5,s6,s7))
        elif active_field == "az":
            splt = (line[:-1].split())
            s0 = float(splt[0])
            s1 = float(splt[1])
            s2 = float(splt[2])
            s3 = float(splt[3])
            s4 = float(splt[4])
            s5 = float(splt[5])
            s6 = float(splt[6])
            s7 = float(splt[7])
            cell_az.append((s0,s1,s2,s3,s4,s5,s6,s7))

        elif active_field == "Qstar_x":
            splt = (line[:-1].split())
            s0 = float(splt[0])
            s1 = float(splt[1])
            s2 = float(splt[2])
            s3 = float(splt[3])
            s4 = float(splt[4])
            s5 = float(splt[5])
            s6 = float(splt[6])
            s7 = float(splt[7])
            cell_Qstar_x.append((s0,s1,s2,s3,s4,s5,s6,s7))
        elif active_field == "Qstar_y":
            splt = (line[:-1].split())
            s0 = float(splt[0])
            s1 = float(splt[1])
            s2 = float(splt[2])
            s3 = float(splt[3])
            s4 = float(splt[4])
            s5 = float(splt[5])
            s6 = float(splt[6])
            s7 = float(splt[7])
            cell_Qstar_y.append((s0,s1,s2,s3,s4,s5,s6,s7))
        elif active_field == "Qstar_z":
            splt = (line[:-1].split())
            s0 = float(splt[0])
            s1 = float(splt[1])
            s2 = float(splt[2])
            s3 = float(splt[3])
            s4 = float(splt[4])
            s5 = float(splt[5])
            s6 = float(splt[6])
            s7 = float(splt[7])
            cell_Qstar_z.append((s0,s1,s2,s3,s4,s5,s6,s7))

        elif active_field == "Flux_x":
            splt = (line[:-1].split())
            s0 = float(splt[0])
            s1 = float(splt[1])
            s2 = float(splt[2])
            s3 = float(splt[3])
            s4 = float(splt[4])
            s5 = float(splt[5])
            s6 = float(splt[6])
            s7 = float(splt[7])
            cell_Flux_x.append((s0,s1,s2,s3,s4,s5,s6,s7))
        elif active_field == "Flux_y":
            splt = (line[:-1].split())
            s0 = float(splt[0])
            s1 = float(splt[1])
            s2 = float(splt[2])
            s3 = float(splt[3])
            s4 = float(splt[4])
            s5 = float(splt[5])
            s6 = float(splt[6])
            s7 = float(splt[7])
            cell_Flux_y.append((s0,s1,s2,s3,s4,s5,s6,s7))
        elif active_field == "Flux_z":
            splt = (line[:-1].split())
            s0 = float(splt[0])
            s1 = float(splt[1])
            s2 = float(splt[2])
            s3 = float(splt[3])
            s4 = float(splt[4])
            s5 = float(splt[5])
            s6 = float(splt[6])
            s7 = float(splt[7])
            cell_Flux_z.append((s0,s1,s2,s3,s4,s5,s6,s7))


cell_min = []
cell_max = []

for bmin,bmax in zip(block_min,block_max):
    for i in range(block_size):
        cmin,cmax = model.get_cell_coords((bmin,bmax),i)
        cell_min.append(cmin)
        cell_max.append(cmax)

print(len(block_min))
print(len(block_max))
print(len(cell_rho ))
print(len(cell_eint))
print(len(cell_vel ))
print(len(cell_min ))
print(len(cell_max ))

print(len(cell_Qstar_x ))
print(len(cell_Qstar_y ))
print(len(cell_Qstar_z ))

select_cell_rho  = []
select_cell_eint = []
select_cell_force_x  = []
select_cell_force_y  = []
select_cell_force_z  = []
select_cell_velx  = []
select_cell_vely  = []
select_cell_velz  = []
select_cell_x    = []
select_cell_y    = []
select_cell_z    = []
select_cell_eint_post_source = []
select_cell_velx_post_source  = []
select_cell_vely_post_source  = []
select_cell_velz_post_source  = []
select_cell_divv_source  = []

select_cell_eint_start_transp = []
select_cell_velx_start_transp  = []
select_cell_vely_start_transp  = []
select_cell_velz_start_transp  = []

select_cell_rho_end_transp = []
select_cell_eint_end_transp = []
select_cell_velx_end_transp  = []
select_cell_vely_end_transp  = []
select_cell_velz_end_transp  = []

select_cell_velx_n_xp  = []
select_cell_vely_n_xp  = []
select_cell_velz_n_xp  = []

select_cell_Q = [[],[],[],[],[],[],[],[]]
select_cell_ax = [[],[],[],[],[],[],[],[]]
select_cell_ay = [[],[],[],[],[],[],[],[]]
select_cell_az = [[],[],[],[],[],[],[],[]]

select_cell_Qstar_x = [[],[],[],[],[],[],[],[]]
select_cell_Qstar_y = [[],[],[],[],[],[],[],[]]
select_cell_Qstar_z = [[],[],[],[],[],[],[],[]]

select_cell_Flux_x = [[],[],[],[],[],[],[],[]]
select_cell_Flux_y = [[],[],[],[],[],[],[],[]]
select_cell_Flux_z = [[],[],[],[],[],[],[],[]]


for i in range(len(cell_rho)):
    rho,eint,vel,cmin,cmax = cell_rho [i],cell_eint[i],cell_vel[i],cell_min[i],cell_max[i]

    eint_src, vel_src = cell_eint_post_source[i], cell_vel_post_source[i],
    eint_trsp, vel_trsp = cell_eint_start_transp[i], cell_vel_start_transp[i],
    rho_etrsp, eint_etrsp, vel_etrsp = cell_rho_end_transp[i], cell_eint_end_transp[i], cell_vel_end_transp[i]

    x,y,z = cmin


    if y == 10 and z == 10:
        select_cell_rho .append(rho )
        select_cell_eint.append(eint)

        vx,vy,vz = vel
        select_cell_velx .append(vx )
        select_cell_vely .append(vy )
        select_cell_velz .append(vz )

        select_cell_eint_post_source.append(eint_src)
        vx,vy,vz = vel_src
        select_cell_velx_post_source .append(vx )
        select_cell_vely_post_source .append(vy )
        select_cell_velz_post_source .append(vz )

        select_cell_eint_start_transp.append(eint_trsp)
        vx,vy,vz = vel_trsp
        select_cell_velx_start_transp .append(vx )
        select_cell_vely_start_transp .append(vy )
        select_cell_velz_start_transp .append(vz )


        fx,fy,fz = cell_force_press[i]
        select_cell_force_x .append(fx )
        select_cell_force_y .append(fy )
        select_cell_force_z .append(fz )

        select_cell_rho_end_transp.append(rho_etrsp)
        select_cell_eint_end_transp.append(eint_etrsp)
        vx,vy,vz = vel_etrsp
        select_cell_velx_end_transp .append(vx )
        select_cell_vely_end_transp .append(vy )
        select_cell_velz_end_transp .append(vz )


        select_cell_divv_source.append(cell_divv_source[i])

        vx,vy,vz = cell_vel_n_xp[i]
        select_cell_velx_n_xp .append(vx )
        select_cell_vely_n_xp .append(vy )
        select_cell_velz_n_xp .append(vz )

        select_cell_x .append(x )
        select_cell_y .append(y )
        select_cell_z .append(z )

        s0,s1,s2,s3,s4,s5,s6,s7 = cell_Q[i]
        select_cell_Q[0].append(s0)
        select_cell_Q[1].append(s1)
        select_cell_Q[2].append(s2)
        select_cell_Q[3].append(s3)
        select_cell_Q[4].append(s4)
        select_cell_Q[5].append(s5)
        select_cell_Q[6].append(s6)
        select_cell_Q[7].append(s7)

        s0,s1,s2,s3,s4,s5,s6,s7 = cell_ax[i]
        select_cell_ax[0].append(s0)
        select_cell_ax[1].append(s1)
        select_cell_ax[2].append(s2)
        select_cell_ax[3].append(s3)
        select_cell_ax[4].append(s4)
        select_cell_ax[5].append(s5)
        select_cell_ax[6].append(s6)
        select_cell_ax[7].append(s7)

        s0,s1,s2,s3,s4,s5,s6,s7 = cell_ay[i]
        select_cell_ay[0].append(s0)
        select_cell_ay[1].append(s1)
        select_cell_ay[2].append(s2)
        select_cell_ay[3].append(s3)
        select_cell_ay[4].append(s4)
        select_cell_ay[5].append(s5)
        select_cell_ay[6].append(s6)
        select_cell_ay[7].append(s7)

        s0,s1,s2,s3,s4,s5,s6,s7 = cell_az[i]
        select_cell_az[0].append(s0)
        select_cell_az[1].append(s1)
        select_cell_az[2].append(s2)
        select_cell_az[3].append(s3)
        select_cell_az[4].append(s4)
        select_cell_az[5].append(s5)
        select_cell_az[6].append(s6)
        select_cell_az[7].append(s7)

        s0,s1,s2,s3,s4,s5,s6,s7 = cell_Qstar_x[i]
        select_cell_Qstar_x[0].append(s0)
        select_cell_Qstar_x[1].append(s1)
        select_cell_Qstar_x[2].append(s2)
        select_cell_Qstar_x[3].append(s3)
        select_cell_Qstar_x[4].append(s4)
        select_cell_Qstar_x[5].append(s5)
        select_cell_Qstar_x[6].append(s6)
        select_cell_Qstar_x[7].append(s7)

        s0,s1,s2,s3,s4,s5,s6,s7 = cell_Qstar_y[i]
        select_cell_Qstar_y[0].append(s0)
        select_cell_Qstar_y[1].append(s1)
        select_cell_Qstar_y[2].append(s2)
        select_cell_Qstar_y[3].append(s3)
        select_cell_Qstar_y[4].append(s4)
        select_cell_Qstar_y[5].append(s5)
        select_cell_Qstar_y[6].append(s6)
        select_cell_Qstar_y[7].append(s7)

        s0,s1,s2,s3,s4,s5,s6,s7 = cell_Qstar_z[i]
        select_cell_Qstar_z[0].append(s0)
        select_cell_Qstar_z[1].append(s1)
        select_cell_Qstar_z[2].append(s2)
        select_cell_Qstar_z[3].append(s3)
        select_cell_Qstar_z[4].append(s4)
        select_cell_Qstar_z[5].append(s5)
        select_cell_Qstar_z[6].append(s6)
        select_cell_Qstar_z[7].append(s7)

        s0,s1,s2,s3,s4,s5,s6,s7 = cell_Flux_x[i]
        select_cell_Flux_x[0].append(s0)
        select_cell_Flux_x[1].append(s1)
        select_cell_Flux_x[2].append(s2)
        select_cell_Flux_x[3].append(s3)
        select_cell_Flux_x[4].append(s4)
        select_cell_Flux_x[5].append(s5)
        select_cell_Flux_x[6].append(s6)
        select_cell_Flux_x[7].append(s7)

        s0,s1,s2,s3,s4,s5,s6,s7 = cell_Flux_y[i]
        select_cell_Flux_y[0].append(s0)
        select_cell_Flux_y[1].append(s1)
        select_cell_Flux_y[2].append(s2)
        select_cell_Flux_y[3].append(s3)
        select_cell_Flux_y[4].append(s4)
        select_cell_Flux_y[5].append(s5)
        select_cell_Flux_y[6].append(s6)
        select_cell_Flux_y[7].append(s7)

        s0,s1,s2,s3,s4,s5,s6,s7 = cell_Flux_z[i]
        select_cell_Flux_z[0].append(s0)
        select_cell_Flux_z[1].append(s1)
        select_cell_Flux_z[2].append(s2)
        select_cell_Flux_z[3].append(s3)
        select_cell_Flux_z[4].append(s4)
        select_cell_Flux_z[5].append(s5)
        select_cell_Flux_z[6].append(s6)
        select_cell_Flux_z[7].append(s7)


print(len(select_cell_rho ))
print(len(select_cell_eint))
print(len(select_cell_velx ))
print(len(select_cell_vely ))
print(len(select_cell_velz ))
print(len(select_cell_x   ))
print(len(select_cell_y   ))
print(len(select_cell_z   ))

print(len(select_cell_Qstar_x ))
print(len(select_cell_Qstar_y ))
print(len(select_cell_Qstar_z ))

import matplotlib.pyplot as plt

plt.style.use('custom_short_cycler.mplstyle')

fig,axs = plt.subplots(nrows=3,ncols=3,figsize=(9,6),dpi=125)

axs[0,0].scatter(select_cell_x,select_cell_rho, s = 1, label = "initial")
axs[0,0].scatter(select_cell_x,select_cell_rho_end_transp, s = 1, label = "end transp")
axs[0,0].set_title("rho")
axs[0,0].legend()

axs[0,1].scatter(select_cell_x,select_cell_velx, s = 1, label = "initial")
axs[0,1].scatter(select_cell_x,select_cell_velx_post_source, s = 1, label = "end source")
axs[0,1].scatter(select_cell_x,select_cell_velx_start_transp,marker = "+", s = 20, label = "begin transp")
axs[0,1].scatter(select_cell_x,select_cell_velx_end_transp, s = 1, label = "end transp")
axs[0,1].scatter(select_cell_x,select_cell_velx_n_xp,marker = "x", s = 20, label = "begin transp x+1")
axs[0,1].set_title("velx")
axs[0,1].legend()

axs[0,2].scatter(select_cell_x,select_cell_eint, s = 1, label = "initial")
axs[0,2].scatter(select_cell_x,select_cell_eint_post_source, s = 1, label = "end source")
axs[0,2].scatter(select_cell_x,select_cell_eint_start_transp, s = 1, label = "begin transp")
axs[0,2].scatter(select_cell_x,select_cell_eint_end_transp, s = 1, label = "end transp")
axs[0,2].set_title("eint")
axs[0,2].legend()

axs[1,1].scatter(select_cell_x,select_cell_force_x, s = 1, label = "fx")
axs[1,1].set_title("fx")
axs[1,1].legend()

axs[1,2].scatter(select_cell_x,select_cell_divv_source, s = 1, label = "divv_source")
axs[1,2].set_title("divv_source")
axs[1,2].legend()

axs[1,0].scatter(select_cell_x,select_cell_Q[0], s = 1, label = "Q")
axs[1,0].scatter(select_cell_x,select_cell_ax[0], s = 1, label = "ax")
axs[1,0].scatter(select_cell_x,select_cell_ay[0], s = 1, label = "ay")
axs[1,0].scatter(select_cell_x,select_cell_az[0], s = 1, label = "az")
axs[1,0].scatter(select_cell_x,select_cell_Qstar_x[0], s = 1, label = "Qstar_x")
#axs[1,0].scatter(select_cell_x,select_cell_Qstar_y[0], s = 1, label = "Qstar_y")
#axs[1,0].scatter(select_cell_x,select_cell_Qstar_z[0], s = 1, label = "Qstar_z")
axs[1,0].scatter(select_cell_x,select_cell_Flux_x[0], s = 1, label = "Flux_x")
#axs[1,0].scatter(select_cell_x,select_cell_Flux_y[0], s = 1, label = "Flux_y")
#axs[1,0].scatter(select_cell_x,select_cell_Flux_z[0], s = 1, label = "Flux_z")
axs[1,0].set_title("Q rho")
axs[1,0].legend()

axs[2,0].scatter(select_cell_x,select_cell_Q[1], s = 1, label = "Q")
axs[2,0].scatter(select_cell_x,select_cell_ax[1], s = 1, label = "ax")
axs[2,0].scatter(select_cell_x,select_cell_ay[1], s = 1, label = "ay")
axs[2,0].scatter(select_cell_x,select_cell_az[1], s = 1, label = "az")
axs[2,0].scatter(select_cell_x,select_cell_Qstar_x[1], s = 1, label = "Qstar_x")
#axs[2,0].scatter(select_cell_x,select_cell_Qstar_y[1], s = 1, label = "Qstar_y")
#axs[2,0].scatter(select_cell_x,select_cell_Qstar_z[1], s = 1, label = "Qstar_z")
axs[2,0].scatter(select_cell_x,select_cell_Flux_x[1], s = 1, label = "Flux_x")
#axs[2,0].scatter(select_cell_x,select_cell_Flux_y[1], s = 1, label = "Flux_y")
#axs[2,0].scatter(select_cell_x,select_cell_Flux_z[1], s = 1, label = "Flux_z")
axs[2,0].set_title("Q pi-x")
axs[2,0].legend()

axs[2,1].scatter(select_cell_x,select_cell_Q[4], s = 1, label = "Q")
axs[2,1].scatter(select_cell_x,select_cell_ax[4], s = 1, label = "ax")
axs[2,1].scatter(select_cell_x,select_cell_ay[4], s = 1, label = "ay")
axs[2,1].scatter(select_cell_x,select_cell_az[4], s = 1, label = "az")
axs[2,1].scatter(select_cell_x,select_cell_Qstar_x[4], s = 1, label = "Qstar_x")
#axs[2,1].scatter(select_cell_x,select_cell_Qstar_y[4], s = 1, label = "Qstar_y")
#axs[2,1].scatter(select_cell_x,select_cell_Qstar_z[4], s = 1, label = "Qstar_z")
axs[2,1].scatter(select_cell_x,select_cell_Flux_x[4], s = 1, label = "Flux_x")
#axs[2,1].scatter(select_cell_x,select_cell_Flux_y[4], s = 1, label = "Flux_y")
#axs[2,1].scatter(select_cell_x,select_cell_Flux_z[4], s = 1, label = "Flux_z")
axs[2,1].set_title("Q pi+x")
axs[2,1].legend()

axs[2,2].scatter(select_cell_x,select_cell_Q[7], s = 1, label = "Q")
axs[2,2].scatter(select_cell_x,select_cell_ax[7], s = 1, label = "ax")
axs[2,2].scatter(select_cell_x,select_cell_ay[7], s = 1, label = "ay")
axs[2,2].scatter(select_cell_x,select_cell_az[7], s = 1, label = "az")
axs[2,2].scatter(select_cell_x,select_cell_Qstar_x[7], s = 1, label = "Qstar_x")
#axs[2,2].scatter(select_cell_x,select_cell_Qstar_y[7], s = 1, label = "Qstar_y")
#axs[2,2].scatter(select_cell_x,select_cell_Qstar_z[7], s = 1, label = "Qstar_z")
axs[2,2].scatter(select_cell_x,select_cell_Flux_x[7], s = 1, label = "Flux_x")
#axs[2,2].scatter(select_cell_x,select_cell_Flux_y[7], s = 1, label = "Flux_y")
#axs[2,2].scatter(select_cell_x,select_cell_Flux_z[7], s = 1, label = "Flux_z")
axs[2,2].set_title("Q e")
axs[2,2].legend()



#axs[1,1].scatter(select_cell_x,select_cell_Q[0], s = 1, label = "rho")
#axs[1,1].scatter(select_cell_x,select_cell_Q[1], s = 1, label = "pi-x")
#axs[1,1].scatter(select_cell_x,select_cell_Q[2], s = 1, label = "pi-y")
#axs[1,1].scatter(select_cell_x,select_cell_Q[3], s = 1, label = "pi-z")
#axs[1,1].scatter(select_cell_x,select_cell_Q[4], s = 1, label = "pi+x")
#axs[1,1].scatter(select_cell_x,select_cell_Q[5], s = 1, label = "pi+y")
#axs[1,1].scatter(select_cell_x,select_cell_Q[6], s = 1, label = "pi+z")
#axs[1,1].scatter(select_cell_x,select_cell_Q[7], s = 1, label = "e")



plt.show()
