"""
SPH projected kernels
========================

This example shows the convergence of the projected M4 kernels
"""

# sphinx_gallery_multi_image = "single"

import matplotlib.pyplot as plt
import numpy as np

import shamrock

Rkern = shamrock.math.sphkernel.M4_Rkern()
q = np.linspace(0, Rkern, 1000)

f_M4 = [shamrock.math.sphkernel.M4_f(x) for x in q]

fint_M4_4 = np.array([shamrock.math.sphkernel.M4_f3d_integ_z(x, 4) for x in q])
fint_M4_8 = np.array([shamrock.math.sphkernel.M4_f3d_integ_z(x, 8) for x in q])
fint_M4_16 = np.array([shamrock.math.sphkernel.M4_f3d_integ_z(x, 16) for x in q])
fint_M4_32 = np.array([shamrock.math.sphkernel.M4_f3d_integ_z(x, 32) for x in q])
fint_M4_64 = np.array([shamrock.math.sphkernel.M4_f3d_integ_z(x, 64) for x in q])
fint_M4_128 = np.array([shamrock.math.sphkernel.M4_f3d_integ_z(x, 128) for x in q])
fint_M4_1024 = np.array([shamrock.math.sphkernel.M4_f3d_integ_z(x, 1024) for x in q])

plt.plot(q, f_M4, label="$f_{M4}(q)$")
plt.plot(q, fint_M4_4, label="$f_{int,M4,n=4}(q)$")
plt.plot(q, fint_M4_1024, label="$f_{int,M4,n=1024}(q)$")
plt.legend()
plt.xlabel(r"$q$")
# plt.savefig("integ_kernel.pdf")
# plt.savefig("integ_kernel.svg")
plt.figure()

plt.plot(q, np.abs(fint_M4_4 - fint_M4_1024), label="n=4")
plt.plot(q, np.abs(fint_M4_8 - fint_M4_1024), label="n=8")
plt.plot(q, np.abs(fint_M4_16 - fint_M4_1024), label="n=16")
plt.plot(q, np.abs(fint_M4_32 - fint_M4_1024), label="n=32")
plt.plot(q, np.abs(fint_M4_64 - fint_M4_1024), label="n=64")
plt.plot(q, np.abs(fint_M4_128 - fint_M4_1024), label="n=128")
plt.legend()

plt.yscale("log")
plt.ylabel(r"$\vert f_{int,n} - f_{int,1024} \vert $")
plt.xlabel(r"$q$")
# plt.savefig("estim_integ_kernel.pdf")
# plt.savefig("estim_integ_kernel.svg")
plt.show()
