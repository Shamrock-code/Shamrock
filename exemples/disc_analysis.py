import sys
sys.path.append('/home/ylapeyre/Documents/githubrepo/sarracen/sarracen') #/home/ylapeyre/Documents/githubrepo/ShamrockWorkspace/Shamrock-venv/lib/python3.11/site-packages/
import sarracen
import matplotlib.pyplot as plt  
import numpy as np 

file_sh = '/home/ylapeyre/Shamrock_tests/v2'
file_ph = '/home/ylapeyre/phantom_tests/warp_newtilt/warp_00000'

sdf_sh = sarracen.read_phantom(file_sh)
unit_am_sh = sarracen.disc.angular_momentum(sdf_sh,bins=100,retbins=True, origin=[0.,0.,0.])
twist_sh = np.arctan(unit_am_sh[2] / unit_am_sh[0]) #ly / lx in Nealon 2015
tilt_sh = np.arccos(unit_am_sh[1])

sdf_ph      = sarracen.read_phantom(file_ph)
unit_am_ph  = sarracen.disc.angular_momentum(sdf_ph,bins=100,retbins=True, origin=[0.,0.,0.])
twist_ph    = np.arctan(unit_am_ph[2] / unit_am_ph[0]) #ly / lx in Nealon 2015
tilt_ph     = np.arccos(unit_am_ph[2])

fig, ax = plt.subplots(1,1,figsize=(5,4))
#ax.plot(unit_am[3],unit_am[0], label='unit x')
ax.plot(unit_am_sh[3],tilt_sh, label='tilt sham')
ax.plot(unit_am_ph[3],tilt_ph, label='tilt phant')
ax.set_xlabel('Radius')
ax.set_ylabel('Unit angular momenta')
ax.legend()
plt.show()



