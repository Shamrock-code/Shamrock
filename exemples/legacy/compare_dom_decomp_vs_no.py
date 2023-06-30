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


def get_result(cnt: int):

    ctx = shamrock.Context()
    ctx.pdata_layout_new()

    ctx.pdata_layout_add_field("xyz",1,"f32_3")

    #field for leapfrog integrator
    ctx.pdata_layout_add_field("vxyz",1,"f32_3")
    ctx.pdata_layout_add_field("axyz",1,"f32_3")
    ctx.pdata_layout_add_field("axyz_old",1,"f32_3")


    #start the scheduler
    ctx.init_sched(cnt,1)


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
    ctx.reset()

    return dic_final


dic_mpi = get_result(int(1e5))

dic_nompi = get_result(int(1e6))

def classify_parts(dic):


    xyz = np.array(dic["xyz"])
    axyz = np.array(dic["axyz"])

    dic2 = {}

    for i in range(len(xyz[:,0])):

        dic2[str((xyz[i,0], xyz[i,1], xyz[i,2]))] = (xyz[i,0], xyz[i,1], xyz[i,2],axyz[i,0],axyz[i,1],axyz[i,2])

    keys =[k for k in dic2.keys()]
    keys.sort()

    X = []
    Y = []
    Z = []

    aX = []
    aY = []
    aZ = []

    for k in keys:
        x,y,z,ax,ay,az = dic2[k]

        X.append(x)
        Y.append(y)
        Z.append(z)
        aX.append(ax)
        aY.append(ay)
        aZ.append(az)

    return np.array(X),np.array(Y),np.array(Z),np.array(aX),np.array(aY),np.array(aZ)


X_mpi,Y_mpi,Z_mpi,aX_mpi,aY_mpi,aZ_mpi = classify_parts(dic_mpi)
X_nompi,Y_nompi,Z_nompi,aX_nompi,aY_nompi,aZ_nompi = classify_parts(dic_nompi)

plt.scatter((X_mpi**2 + Y_mpi**2 + Z_mpi**2)**0.5,(aX_mpi**2 + aY_mpi**2 + aZ_mpi**2)**0.5, label = "mpi")
plt.scatter((X_nompi**2 + Y_nompi**2 + Z_nompi**2)**0.5,(aX_nompi**2 + aY_nompi**2 + aZ_nompi**2)**0.5, label = "nompi")

plt.legend()

plt.ylim(-1,10)

plt.xlabel(r"$\vert \mathbf{r} \vert$")
plt.ylabel(r"$\vert \mathbf{f} \vert$")

plt.figure()



plt.scatter((X_mpi**2 + Y_mpi**2 + Z_mpi**2)**0.5,((aX_mpi - aX_nompi)**2 + (aY_mpi - aY_nompi)**2 + (aZ_mpi - aZ_nompi)**2)**0.5)

plt.legend()

plt.ylim(-1,10)

plt.xlabel(r"$\vert \mathbf{r} \vert$")
plt.ylabel(r"$\vert \mathbf{f} \vert$")

plt.savefig("grav_diff.pdf")
plt.show()


dic_out = {
    "xyz" : [[X_mpi[i],Y_mpi[i],Z_mpi[i]] for i in range(len(X_mpi))],
    "ampi" : [[aX_mpi[i],aY_mpi[i],aZ_mpi[i]] for i in range(len(X_mpi))],
    "anompi" : [[aX_nompi[i],aY_nompi[i],aZ_nompi[i]] for i in range(len(X_mpi))],
    "diff" : [[aX_mpi[i] - aX_nompi[i],aY_mpi[i] - aY_nompi[i],aZ_mpi[i] - aZ_nompi[i]] for i in range(len(X_mpi))]
}

write_dic_to_vtk(dic_out, "comp")