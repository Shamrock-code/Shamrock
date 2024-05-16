"""
compare Shamrock and Phantom's fortran dumps through sarracen
"""

import sys
sys.path.append('./') #address of sarracen's dev branch on your machine
import sarracen
import matplotlib.pyplot as plt  
import numpy as np 

file_sh = '/Users/ylapeyre/Documents/Shamwork/09_04/test1/dump_0001'
file_ph = '/Users/ylapeyre/Documents/githubrepos/phantomtests/warp3/disc_00000'

sdf_sh = sarracen.read_phantom(file_sh)
unit_am_sh = sarracen.disc.angular_momentum(sdf_sh, r_in= 0.1, r_out= 10, bins=300,retbins=True, origin=[0.,0.,0.])
twist_sh = np.arctan(unit_am_sh[2] / unit_am_sh[0]) #ly / lx in Nealon 2015
tilt_sh = np.arccos(unit_am_sh[2])

sdf_ph      = sarracen.read_phantom(file_ph)
unit_am_ph  = sarracen.disc.angular_momentum(sdf_ph,bins=100,retbins=True, origin=[0.,0.,0.])
twist_ph    = np.arctan(unit_am_ph[2] / unit_am_ph[0]) #ly / lx in Nealon 2015
tilt_ph     = np.arccos(unit_am_ph[2])


"""
plt.plot(unit_am_sh[3], unit_am_sh[0], label='Lx', c='g')
plt.plot(unit_am_sh[3], -unit_am_sh[1], label='Ly', c='r')
plt.plot(unit_am_sh[3], unit_am_sh[2], label='Lz', c='b')

plt.plot(unit_am_ph[3], unit_am_ph[0], label='Lx', c='g', linestyle='--')
plt.plot(unit_am_ph[3], unit_am_ph[1], label='Ly', c='r', linestyle='--')
plt.plot(unit_am_ph[3], unit_am_ph[2], label='Lz', c='b', linestyle='--')
plt.legend()
plt.show()

"""
############################ several plots ##############################
for i in range (0, 51, 10):
    file = '/Users/ylapeyre/Documents/Shamwork/23_04/test_mass/dump_' + str(i).zfill(4)
    sdf = sarracen.read_phantom(file)
    unit_am = sarracen.disc.angular_momentum(sdf,r_in= 0.1, r_out= 2, bins=300,retbins=True, origin=[0.,0.,0.])
    #tilt = np.arccos(unit_am[2])
    twist = np.arctan(unit_am[2] / unit_am[0])
    plt.scatter(unit_am[3], twist, marker='.', label=str(i).zfill(4))
#plt.scatter(unit_am_ph[3], tilt_ph, c='k', marker='.', label='phantom')
plt.title('test 1')
plt.legend()
plt.show()

"""
############################ 1 plot ##############################
plt.scatter(unit_am_sh[3], twist_sh, marker='.', label='shamrock')
#plt.scatter(unit_am_ph[3], tilt_ph, c='k', marker='.', label='phantom')
plt.title('tilt')
plt.legend()
plt.show()



############################ full comparison ##############################
fig, ax = plt.subplots(1,5,figsize=(10,12))
#ax.plot(unit_am[3],unit_am[0], label='unit x')

ax[0].plot(unit_am_sh[3], unit_am_sh[0], label='Lx', c='g')
ax[0].plot(unit_am_sh[3], -unit_am_sh[1], label='Ly', c='r')
ax[0].plot(unit_am_sh[3], unit_am_sh[2], label='Lz', c='b')

ax[0].plot(unit_am_ph[3], unit_am_ph[0], label='Lx', c='g', linestyle='--')
ax[0].plot(unit_am_ph[3], unit_am_ph[1], label='Ly', c='r', linestyle='--')
ax[0].plot(unit_am_ph[3], unit_am_ph[2], label='Lz', c='b', linestyle='--')

ax[0].set_xlabel('Radius')
ax[0].set_title('Unit angular momenta')
ax[0].legend()

ax[1].scatter(unit_am_sh[3], tilt_sh, c='r', marker='.', label='shamrock')
ax[1].scatter(unit_am_ph[3], tilt_ph, c='k', marker='.', label='phantom')
ax[1].set_title('tilt')
ax[1].legend()

ax[2].scatter(unit_am_sh[3], twist_sh, c='r', marker='.', label='shamrock')
ax[2].scatter(unit_am_ph[3], twist_ph, c='k', marker='.', label='phantom')
ax[2].set_title('twist')
ax[2].legend()

ax[3].scatter(sdf_sh['x'], sdf_sh['x'], label='x', c='g', marker='.', alpha=0.5)
#ax[3].scatter(sdf_sh['x'], sdf_sh['y'], label='y', c='r', marker='.', alpha=0.5)
ax[3].scatter(sdf_sh['x'], sdf_sh['z'], label='z', c='b', marker='.', alpha=0.5)
ax[3].legend()

ax[4].scatter(sdf_sh['x'], sdf_sh['vx'], label='vx', c='g', marker='.', alpha=0.5)
#[4].scatter(sdf_sh['x'], sdf_sh['vy'], label='vy', c='r', marker='.', alpha=0.5)
ax[4].scatter(sdf_sh['x'], sdf_sh['vz'], label='vz', c='b', marker='.', alpha=0.5)
ax[4].legend()

plt.show()




############################## TIME LOG ###############################

R_th = np.linspace(0.2, 2, 50)

def tp_th(u):
    a = 0,9
    return np.log(u**3 / (2*a))

Y_th = [tp_th(u) for u in R_th]

plt.plot(R_th, Y_th)
plt.show()

"""
