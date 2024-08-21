import shamrock
import matplotlib.pyplot as plt

gamma = 1.4

rho_g = 1
rho_d = 0.125

fact = (rho_g/rho_d)**(1./3.)

P_g = 1
P_d = 0.1

u_g = P_g/((gamma - 1)*rho_g)
u_d = P_d/((gamma - 1)*rho_d)

resol = 128

ctx = shamrock.Context()
ctx.pdata_layout_new()

model = shamrock.get_SPHModel(context = ctx, vector_type = "f64_3",sph_kernel = "M6")


model.load_from_dump("outfile")

t_target = 0.245

model.evolve_until(t_target)

#model.evolve_once()

sod = shamrock.phys.SodTube(gamma = gamma, rho_1 = 1,P_1 = 1,rho_5 = 0.125,P_5 = 0.1)
sodanalysis = model.make_analysis_sodtube(sod, (1,0,0), t_target, 0.0, -0.5,0.5)
print(sodanalysis.compute_L2_dist())


model.do_vtk_dump("end.vtk", True)
dump = model.make_phantom_dump()
dump.save_dump("end.phdump")

import numpy as np
dic = ctx.collect_data()

x =np.array(dic['xyz'][:,0]) + 0.5
vx = dic['vxyz'][:,0]
uint = dic['uint'][:]

hpart = dic["hpart"]
alpha = dic["alpha_AV"]

rho = pmass*(model.get_hfact()/hpart)**3
P = (gamma-1) * rho *uint


plt.plot(x,rho,'.',label="rho")
plt.plot(x,vx,'.',label="v")
plt.plot(x,P,'.',label="P")
plt.plot(x,alpha,'.',label="alpha")
#plt.plot(x,hpart,'.',label="hpart")
#plt.plot(x,uint,'.',label="uint")


#### add analytical soluce
x = np.linspace(-0.5,0.5,1000)

rho = []
P = []
vx = []

for i in range(len(x)):
    x_ = x[i]

    _rho,_vx,_P = sod.get_value(t_target, x_)
    rho.append(_rho)
    vx.append(_vx)
    P.append(_P)

x += 0.5
plt.plot(x,rho,color = "black",label="analytic")
plt.plot(x,vx,color = "black")
plt.plot(x,P,color = "black")
#######



plt.legend()
plt.grid()
plt.ylim(0,1.1)
plt.xlim(0,1)
plt.title("t="+str(t_target))
plt.show()