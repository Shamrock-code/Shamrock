import numpy as np
import matplotlib.pyplot as plt

filename_ph = "/home/ylapeyre/phantom_tests/warp_newtilt/angm00000"
filename_sh = "/home/ylapeyre/Shamrock_tests/new_tilt/angm00000"

file_ph = np.genfromtxt(filename_ph, skip_header=2)
file_sh = np.genfromtxt(filename_sh, skip_header=2)

r_ph = file_ph[:, 0]
lx_ph = file_ph[:, 3]
ly_ph = file_ph[:, 4]
lz_ph = file_ph[:, 5]
tilt_ph = file_ph[:, 6]
twist_ph = file_ph[:, 7]

r_sh = file_sh[:, 0]
lx_sh = file_sh[:, 3]
ly_sh = file_sh[:, 4]
lz_sh = file_sh[:, 5]
tilt_sh = file_sh[:, 6]
twist_sh = file_sh[:, 7]

############## do rotation





fig, axs = plt.subplots(1, 6)

axs[0].scatter(r_ph, lx_ph, c='k', marker='x', s=1, label='phantom')
axs[0].scatter(r_sh, lx_sh, c='r', marker='x', s=1, label='shamrock')
axs[0].set_title('Lx')
axs[0].set_xlabel('radius')

axs[1].scatter(r_ph, ly_ph, c='k', marker='x', s=1)
axs[1].scatter(r_sh, ly_sh, c='r', marker='x', s=1)
axs[1].set_title('Ly')
axs[1].set_xlabel('radius')

axs[2].scatter(r_ph, lz_ph, c='k', marker='x', s=1, label='phantom')
axs[2].scatter(r_sh, lz_sh, c='r', marker='x', s=1, label='shamrock')
axs[2].set_title('Lz')
axs[2].set_xlabel('radius')

axs[3].scatter(r_ph, tilt_ph, c='k', marker='x', s=1)
axs[3].scatter(r_sh, tilt_sh, c='r', marker='x', s=1)
axs[3].set_title('tilt [deg]')
axs[3].set_xlabel('radius')

axs[4].scatter(r_ph, twist_ph, c='k', marker='x', s=1)
axs[4].scatter(r_sh, twist_sh, c='r', marker='x', s=1)
axs[4].set_title('twist [deg]')
axs[4].set_xlabel('radius')

axs[5].scatter(r_ph, tilt_sh - tilt_ph, c='k', marker='x', s=1)
axs[5].set_title('sh-ph')
axs[5].set_xlabel('radius')

axs[0].legend()
plt.show()


