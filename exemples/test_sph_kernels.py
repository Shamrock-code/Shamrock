import shamrock
import numpy as np
import matplotlib.pyplot as plt

Rkern = shamrock.sphkernel.M4_Rkern()
q = np.linspace(0,Rkern,1000)

f_M4 = [shamrock.sphkernel.M4_f(x) for x in q]

fint_M4_4 = np.array([shamrock.sphkernel.M4_f3d_integ_z(x,4) for x in q])
fint_M4_8 = np.array([shamrock.sphkernel.M4_f3d_integ_z(x,8) for x in q])
fint_M4_16 = np.array([shamrock.sphkernel.M4_f3d_integ_z(x,16) for x in q])
fint_M4_32 = np.array([shamrock.sphkernel.M4_f3d_integ_z(x,32) for x in q])
fint_M4_64 = np.array([shamrock.sphkernel.M4_f3d_integ_z(x,64) for x in q])
fint_M4_128 = np.array([shamrock.sphkernel.M4_f3d_integ_z(x,128) for x in q])
fint_M4_1024 = np.array([shamrock.sphkernel.M4_f3d_integ_z(x,1024) for x in q])

plt.plot(q,f_M4)
plt.figure()

plt.plot(q,np.abs(fint_M4_4 - fint_M4_1024),label="4")
plt.plot(q,np.abs(fint_M4_8 - fint_M4_1024),label="8")
plt.plot(q,np.abs(fint_M4_16 - fint_M4_1024),label="16")
plt.plot(q,np.abs(fint_M4_32 - fint_M4_1024),label="32")
plt.plot(q,np.abs(fint_M4_64 - fint_M4_1024),label="64")
plt.plot(q,np.abs(fint_M4_128 - fint_M4_1024),label="128")
plt.legend()

plt.yscale("log")
plt.ylabel(r"$\vert f_{int} - f_{int,1024} \vert $")
plt.show()
