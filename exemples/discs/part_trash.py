import numpy as np
import matplotlib.pyplot as plt
import sarracen
import vtk

filename_ph = "/home/ylapeyre/phantom_tests/warp_newtilt/warp_00000"
filename_sh = "/home/ylapeyre/Shamrock_tests/new_tilt/test_LT_0000"


file_ph = sarracen.read_phantom(filename_ph)
file_ph['r'] = np.sqrt(file_ph['x']**2 + file_ph['y']**2 + file_ph['z']**2)
file_ph['Lz'] = file_ph['x'] * file_ph['vy'] - file_ph['y'] * file_ph['vx']
ctxt_ph = file_ph.describe()
#params_ph = file_ph.params()
#print(params_ph) 

file_sh = sarracen.read_phantom(filename_sh)
file_sh['r'] = np.sqrt(file_sh['x']**2 + file_sh['y']**2 + file_sh['z']**2)
file_sh['Lz'] = file_sh['x'] * file_sh['vy'] - file_sh['y'] * file_sh['vx']
ctxt_sh = file_sh.describe()
#params_sh = file_sh.params()
#print(params_sh) 

print(file_sh)

fig, axs = plt.subplots(2, 4)

axs[0, 0].scatter(file_ph['x'][::100], file_ph['vx'][::100], c='k', marker='.', s=1, label='phantom')
axs[0, 0].scatter(file_sh['x'][::100], file_sh['vx'][::100], c='r', marker='.', s=1, label='shamrock')
axs[0, 0].set_title('vx')
axs[0, 0].set_xlabel('x')
axs[0, 0].legend()

axs[0, 1].scatter(file_ph['y'][::100], file_ph['vy'][::100], c='k', marker='.', s=1, label='phantom')
axs[0, 1].scatter(file_sh['y'][::100], file_sh['vy'][::100], c='r', marker='.', s=1, label='shamrock')
axs[0, 1].set_title('vy')
axs[0, 1].set_xlabel('y')
axs[0, 1].legend()

axs[0, 2].scatter(file_ph['z'][::100], file_ph['vz'][::100], c='k', marker='.', s=1, label='phantom')
axs[0, 2].scatter(file_sh['z'][::100], file_sh['vz'][::100], c='g', marker='.', s=1, label='shamrock')
axs[0, 2].set_title('vz')
axs[0, 2].set_xlabel('vx')
axs[0, 2].legend()

axs[0, 3].scatter(file_ph['vy'][::100], file_ph['Lz'][::100], c='k', marker='.', s=1, label='phantom')
axs[0, 3].scatter(file_sh['vy'][::100], file_sh['Lz'][::100], c='r', marker='.', s=1, label='shamrock')
axs[0, 3].set_title('Lz')
axs[0, 3].set_xlabel('vy')
axs[0, 3].legend()

######################
axs[1, 0].scatter(file_sh['x'][::100], file_sh['Lz'][::100], c='r', marker='.', s=1, label='shamrock')
axs[1, 0].set_title('Lz')
axs[1, 0].set_xlabel('x')
axs[1, 0].legend()


axs[1, 1].scatter(file_sh['y'][::100], file_sh['Lz'][::100], c='r', marker='.', s=1, label='shamrock')
axs[1, 1].set_title('Lz')
axs[1, 1].set_xlabel('y')
axs[1, 1].legend()

axs[1, 2].scatter(file_sh['vx'][::100], file_sh['Lz'][::100], c='r', marker='.', s=1, label='shamrock')
axs[1, 2].set_title('Lz')
axs[1, 2].set_xlabel('vx')
axs[1, 2].legend()

axs[1, 3].scatter(file_sh['vy'][::100], file_sh['Lz'][::100], c='r', marker='.', s=1, label='shamrock')
axs[1, 3].set_title('Lz')
axs[1, 3].set_xlabel('vy')
axs[1, 3].legend()

plt.show()

