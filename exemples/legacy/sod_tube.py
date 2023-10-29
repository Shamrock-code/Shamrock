import shamrock
import numpy as np
import matplotlib.pyplot as plt 
import os


import vtk
import vtkmodules



def to_tuple(arr):
    t = tuple(e for e in arr)
    return t

def write_dic_to_vtk(dic,filename):

    npart = len(dic["xyz"])

    points = vtk.vtkPoints()
    points.SetNumberOfPoints(npart)
    for i in range(npart):
        points.SetPoint(i,to_tuple(dic["xyz"][i]))

    ugrid = vtkmodules.vtkCommonDataModel.vtkUnstructuredGrid()

    ugrid.SetPoints(points)



    for k in dic.keys():

        if not (k == "xyz"):
            
            if type(dic[k][0]) == list:

                vect = vtk.vtkDoubleArray()
                vect.SetName(k)


                ncomp = len(dic[k][0])
                vect.SetNumberOfComponents(ncomp)
                vect.SetNumberOfValues(npart*ncomp)

                for i in range(npart):
                    vect.SetTuple(i,to_tuple(dic[k][i]))

                    ugrid.GetPointData().AddArray(vect)

            else:
                values = vtk.vtkDoubleArray()
                values.SetName(k)
                values.SetNumberOfValues(npart)
                for i in range(npart):
                    values.SetValue(i,dic[k][i])

                ugrid.GetPointData().AddArray(values)



    fn = filename + '.vtu'
    writer = vtk.vtkUnstructuredGridWriter()
    writer.SetFileName(fn)
    writer.SetFileTypeToBinary()
    writer.SetInputData(ugrid)
    writer.Write()


###
#export DPCPP_HOME=$(pwd)/sycl_cpl/dpcpp 
#export LD_LIBRARY_PATH=$DPCPP_HOME/../../sycl_cpl_src/dpcpp/llvm/build/lib:$LD_LIBRARY_PATH
#export PATH=$DPCPP_HOME/llvm/build/bin:$PATH
###


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


def sim_setup(ctx : shamrock.Context):

    global pmass

    setup = shamrock.SetupSPH_M4_single()
    setup.init(ctx)

    bdim = (256,24,24)
    #bdim = (512,48,48)
    (xs,ys,zs) = setup.get_box_dim_icnt(1,bdim)

    #todo set box size to otherwise split patchdata won't work in the setup

    dr = 1/xs 

    (xs,ys,zs) = setup.get_box_dim_icnt(dr,bdim)


    ctx.set_box_size(((-xs,-ys/2,-zs/2),(xs,ys/2,zs/2)))
    setup.set_boundaries(True)

    fact = (rho_g/rho_d)**(1./3.)

    setup.add_cube_fcc(ctx,dr, ((-xs,-ys/2,-zs/2),(0,ys/2,zs/2)))
    setup.add_cube_fcc(ctx,dr*fact, ((0,-ys/2,-zs/2),(xs,ys/2,zs/2)))

    setup.set_value_in_box(ctx,((-xs,-ys/2,-zs/2),(0,ys/2,zs/2)),"uint","f32",u_g)
    setup.set_value_in_box(ctx,((0,-ys/2,-zs/2),(xs,ys/2,zs/2)),"uint","f32",u_d)

    vol_b = xs*ys*zs

    totmass = (rho_d*vol_b) + (rho_g*vol_b)

    print("Total mass :", totmass)

    setup.set_total_mass(totmass)

    pmass = setup.get_part_mass()

    print("Current part mass :", pmass)

    for it in range(5):
        setup.update_smoothing_length(ctx)

    setup.clear()


def print_dist(ctx : shamrock.Context, cname : str,fname : str):

    dic = ctx.collect_data()


    write_dic_to_vtk(dic, fname)

 



sim_setup(ctx)
print_dist(ctx,"setup","setup")

#exit()


model = shamrock.BasicSPHGasUInterne_M4_single()
model.init()
model.set_cfl_cour(1e-1)
model.set_cfl_force(0.3)


print("Current part mass :", pmass)


model.set_particle_mass(pmass)
t_end = 0
nstep = 15
for i in range(nstep):
    t_end = model.simulate_until(ctx, t_end,t_end+1e-2 ,1,1,"dump_")
    print_dist(ctx,"t = " + str(t_end),"step_"+str(i))


model.clear()

print_dist(ctx,"t = " + str(t_end),"end")

import json
dic = ctx.collect_data()
tmp = open("result2.json",'w')
json.dump(dic,tmp ,indent=4)
tmp.close()