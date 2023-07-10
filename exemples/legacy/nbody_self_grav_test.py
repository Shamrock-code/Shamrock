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


ctx = shamrock.Context()
ctx.pdata_layout_new()

ctx.pdata_layout_add_field("xyz",1,"f32_3")

#field for leapfrog integrator
ctx.pdata_layout_add_field("vxyz",1,"f32_3")
ctx.pdata_layout_add_field("axyz",1,"f32_3")
ctx.pdata_layout_add_field("axyz_old",1,"f32_3")


#start the scheduler
ctx.init_sched(int(1e5),1)






setup = shamrock.Nbody_setup_f32()
setup.init(ctx)


rho_g = 1

pmass = -1

bsz = 1
dr = 0.02


aspect_rat = 1



#set box size
#bdim = setup.get_ideal_box(dr,((-bsz*aspect_rat,-bsz*aspect_rat,-bsz),(bsz*aspect_rat,bsz*aspect_rat,bsz)))
#dimm,dimM = bdim
#xm,ym,zm = dimm
#xM,yM,zM = dimM
#vol_b = (xM-xm)*(yM-ym)*(zM-zm)
#print("box resized to :",bdim,"| volume :",vol_b)
bdim = ((-1,-1,-1),(1,1,1))
dimm,dimM = bdim
xm,ym,zm = dimm
xM,yM,zM = dimM
vol_b = (xM-xm)*(yM-ym)*(zM-zm)
print("box resized to :",bdim,"| volume :",vol_b)
ctx.set_box_size(bdim)

#set BC & add particles
setup.set_boundaries(False)
setup.add_cube_fcc(ctx,dr, bdim)

#set particle mass
totmass = rho_g*vol_b
print("Total mass :", totmass)
setup.set_total_mass(totmass)
pmass = setup.get_part_mass()
print("Current part mass :", pmass)



#clean setup
setup.clear()


def print_dist(dic, cname : str,fname : str,tval):

    #plt.figure()

    xyz = np.array(dic["xyz"])
    axyz = np.array(dic["axyz"])



    plt.title(cname)
    plt.scatter((xyz[:,0]**2 +xyz[:,1]**2 +xyz[:,2]**2)**0.5 , (axyz[:,0]**2 +axyz[:,1]**2 +axyz[:,2]**2)**0.5,label = "simulation")
    plt.legend()
    plt.savefig(fname)

    



dic_setup = ctx.collect_data()

print_dist(dic_setup,"$t = 0$","setup.pdf",0)

write_dic_to_vtk(dic_setup, "initial")



model = shamrock.NBody_selfgrav_f32()
model.init()
model.set_cfl_force(0.3)
model.set_particle_mass(pmass)

model.simulate_until(ctx, 0,1e-2 ,1,1,"dump_")

#t_end = 0
#nstep = 10
#for i in range(nstep):
#    t_end = model.simulate_until(ctx, t_end,t_end+1e-1 ,1,1,"dump_")
#    write_dic_to_vtk(ctx.collect_data(),"step_"+str(i))

model.clear()

dic_final = ctx.collect_data()
print_dist(dic_final,"$t = 0$","end.pdf",0)
write_dic_to_vtk(dic_final,"end")

plt.ylim(-1,10)

plt.xlabel(r"$\vert \mathbf{r} \vert$")

plt.ylabel(r"$\vert \mathbf{f} \vert$")

plt.show()