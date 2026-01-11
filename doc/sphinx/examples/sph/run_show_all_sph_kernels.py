"""
SPH kernels
===========

This example shows the all the SPH kernels
"""

import matplotlib.pyplot as plt
import numpy as np

import shamrock

# %%
# utilities & check integral == 1


def compute_integ_3d(q, W):
    if hasattr(np, "trapezoid"):
        integrate_func = getattr(np, "trapezoid")
    else:
        integrate_func = getattr(np, "trapz")
    return integrate_func(4 * np.pi * q**2 * W, q)


def plot_test_sph_kernel(q, f, df, W, dW, title, ax):

    integral_result = compute_integ_3d(q, W)
    assert np.abs(integral_result - 1) < 1e-6, (
        "3D integration of 4\pi q^2 W(q) is not 1, kernel: " + title
    )

    ax[0].plot(q, W, label=f"$W_{{{title}}}(q)$")
    ax[1].plot(q, dW, label=f"$dW_{{{title}}}(q)$")


q = np.linspace(0, 4, 1000)

# %%
# Cubic splines

fig, axs = plt.subplots(1, 2, figsize=(10, 5))

f_M4 = [shamrock.math.sphkernel.M4_f(x) for x in q]
w3d_M4 = [shamrock.math.sphkernel.M4_W3d(x, 1) for x in q]
df_M4 = [shamrock.math.sphkernel.M4_df(x) for x in q]
dW3d_M4 = [shamrock.math.sphkernel.M4_dW3d(x, 1) for x in q]
plot_test_sph_kernel(q, f_M4, df_M4, w3d_M4, dW3d_M4, "M4", axs)

f_M6 = [shamrock.math.sphkernel.M6_f(x) for x in q]
w3d_M6 = [shamrock.math.sphkernel.M6_W3d(x, 1) for x in q]
df_M6 = [shamrock.math.sphkernel.M6_df(x) for x in q]
dW3d_M6 = [shamrock.math.sphkernel.M6_dW3d(x, 1) for x in q]
plot_test_sph_kernel(q, f_M6, df_M6, w3d_M6, dW3d_M6, "M6", axs)

f_M8 = [shamrock.math.sphkernel.M8_f(x) for x in q]
w3d_M8 = [shamrock.math.sphkernel.M8_W3d(x, 1) for x in q]
df_M8 = [shamrock.math.sphkernel.M8_df(x) for x in q]
dW3d_M8 = [shamrock.math.sphkernel.M8_dW3d(x, 1) for x in q]
plot_test_sph_kernel(q, f_M8, df_M8, w3d_M8, dW3d_M8, "M8", axs)

axs[0].legend()
axs[1].legend()
axs[0].set_xlabel(r"$q$")
axs[1].set_xlabel(r"$q$")
plt.show()


# %%
# Wendland kernels

fig, axs = plt.subplots(1, 2, figsize=(10, 5))

f_C2 = [shamrock.math.sphkernel.C2_f(x) for x in q]
w3d_C2 = [shamrock.math.sphkernel.C2_W3d(x, 1) for x in q]
df_C2 = [shamrock.math.sphkernel.C2_df(x) for x in q]
dW3d_C2 = [shamrock.math.sphkernel.C2_dW3d(x, 1) for x in q]
plot_test_sph_kernel(q, f_C2, df_C2, w3d_C2, dW3d_C2, "C2", axs)

f_C4 = [shamrock.math.sphkernel.C4_f(x) for x in q]
w3d_C4 = [shamrock.math.sphkernel.C4_W3d(x, 1) for x in q]
df_C4 = [shamrock.math.sphkernel.C4_df(x) for x in q]
dW3d_C4 = [shamrock.math.sphkernel.C4_dW3d(x, 1) for x in q]
plot_test_sph_kernel(q, f_C4, df_C4, w3d_C4, dW3d_C4, "C4", axs)

f_C6 = [shamrock.math.sphkernel.C6_f(x) for x in q]
w3d_C6 = [shamrock.math.sphkernel.C6_W3d(x, 1) for x in q]
df_C6 = [shamrock.math.sphkernel.C6_df(x) for x in q]
dW3d_C6 = [shamrock.math.sphkernel.C6_dW3d(x, 1) for x in q]
plot_test_sph_kernel(q, f_C6, df_C6, w3d_C6, dW3d_C6, "C6", axs)

axs[0].legend()
axs[1].legend()
axs[0].set_xlabel(r"$q$")
axs[1].set_xlabel(r"$q$")
plt.show()


# %%
# Double hump kernels

fig, axs = plt.subplots(1, 2, figsize=(10, 5))

f_M4DH = [shamrock.math.sphkernel.M4DH_f(x) for x in q]
w3d_M4DH = [shamrock.math.sphkernel.M4DH_W3d(x, 1) for x in q]
df_M4DH = [shamrock.math.sphkernel.M4DH_df(x) for x in q]
dW3d_M4DH = [shamrock.math.sphkernel.M4DH_dW3d(x, 1) for x in q]
plot_test_sph_kernel(q, f_M4DH, df_M4DH, w3d_M4DH, dW3d_M4DH, "M4DH", axs)

f_M4DH3 = [shamrock.math.sphkernel.M4DH3_f(x) for x in q]
w3d_M4DH3 = [shamrock.math.sphkernel.M4DH3_W3d(x, 1) for x in q]
df_M4DH3 = [shamrock.math.sphkernel.M4DH3_df(x) for x in q]
dW3d_M4DH3 = [shamrock.math.sphkernel.M4DH3_dW3d(x, 1) for x in q]
plot_test_sph_kernel(q, f_M4DH3, df_M4DH3, w3d_M4DH3, dW3d_M4DH3, "M4DH3", axs)

f_M4DH5 = [shamrock.math.sphkernel.M4DH5_f(x) for x in q]
w3d_M4DH5 = [shamrock.math.sphkernel.M4DH5_W3d(x, 1) for x in q]
df_M4DH5 = [shamrock.math.sphkernel.M4DH5_df(x) for x in q]
dW3d_M4DH5 = [shamrock.math.sphkernel.M4DH5_dW3d(x, 1) for x in q]
plot_test_sph_kernel(q, f_M4DH5, df_M4DH5, w3d_M4DH5, dW3d_M4DH5, "M4DH5", axs)

f_M4DH7 = [shamrock.math.sphkernel.M4DH7_f(x) for x in q]
w3d_M4DH7 = [shamrock.math.sphkernel.M4DH7_W3d(x, 1) for x in q]
df_M4DH7 = [shamrock.math.sphkernel.M4DH7_df(x) for x in q]
dW3d_M4DH7 = [shamrock.math.sphkernel.M4DH7_dW3d(x, 1) for x in q]
plot_test_sph_kernel(q, f_M4DH7, df_M4DH7, w3d_M4DH7, dW3d_M4DH7, "M4DH7", axs)

axs[0].legend()
axs[1].legend()
axs[0].set_xlabel(r"$q$")
axs[1].set_xlabel(r"$q$")
plt.show()


# %%
# shifted cubic splines kernels

fig, axs = plt.subplots(1, 2, figsize=(10, 5))

f_M4Shift2 = [shamrock.math.sphkernel.M4Shift2_f(x) for x in q]
w3d_M4Shift2 = [shamrock.math.sphkernel.M4Shift2_W3d(x, 1) for x in q]
df_M4Shift2 = [shamrock.math.sphkernel.M4Shift2_df(x) for x in q]
dW3d_M4Shift2 = [shamrock.math.sphkernel.M4Shift2_dW3d(x, 1) for x in q]
plot_test_sph_kernel(q, f_M4Shift2, df_M4Shift2, w3d_M4Shift2, dW3d_M4Shift2, "M4Shift2", axs)

f_M4Shift4 = [shamrock.math.sphkernel.M4Shift4_f(x) for x in q]
w3d_M4Shift4 = [shamrock.math.sphkernel.M4Shift4_W3d(x, 1) for x in q]
df_M4Shift4 = [shamrock.math.sphkernel.M4Shift4_df(x) for x in q]
dW3d_M4Shift4 = [shamrock.math.sphkernel.M4Shift4_dW3d(x, 1) for x in q]
plot_test_sph_kernel(q, f_M4Shift4, df_M4Shift4, w3d_M4Shift4, dW3d_M4Shift4, "M4Shift4", axs)

f_M4Shift8 = [shamrock.math.sphkernel.M4Shift8_f(x) for x in q]
w3d_M4Shift8 = [shamrock.math.sphkernel.M4Shift8_W3d(x, 1) for x in q]
df_M4Shift8 = [shamrock.math.sphkernel.M4Shift8_df(x) for x in q]
dW3d_M4Shift8 = [shamrock.math.sphkernel.M4Shift8_dW3d(x, 1) for x in q]
plot_test_sph_kernel(q, f_M4Shift8, df_M4Shift8, w3d_M4Shift8, dW3d_M4Shift8, "M4Shift8", axs)

f_M4Shift16 = [shamrock.math.sphkernel.M4Shift16_f(x) for x in q]
w3d_M4Shift16 = [shamrock.math.sphkernel.M4Shift16_W3d(x, 1) for x in q]
df_M4Shift16 = [shamrock.math.sphkernel.M4Shift16_df(x) for x in q]
dW3d_M4Shift16 = [shamrock.math.sphkernel.M4Shift16_dW3d(x, 1) for x in q]
plot_test_sph_kernel(q, f_M4Shift16, df_M4Shift16, w3d_M4Shift16, dW3d_M4Shift16, "M4Shift16", axs)

axs[0].legend()
axs[1].legend()
axs[0].set_xlabel(r"$q$")
axs[1].set_xlabel(r"$q$")
plt.show()
