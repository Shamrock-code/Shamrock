import sys
sys.path.append('/home/ylapeyre/Documents/githubrepo/sarracen/sarracen') #/home/ylapeyre/Documents/githubrepo/ShamrockWorkspace/Shamrock-venv/lib/python3.11/site-packages/
import sarracen
import matplotlib.pyplot as plt  
import numpy as np 

file = '/home/ylapeyre/phantom_tests/warp_newtilt/warp_00000'
sdf = sarracen.read_phantom(file)
unit_am = sarracen.disc.angular_momentum(sdf,bins=100,retbins=True)

twist = np.arctan(unit_am[2] / unit_am[0]) #ly / lx in Nealon 2015
tilt = np.arccos(unit_am[1])

fig, ax = plt.subplots(1,1,figsize=(5,4))
#ax.plot(unit_am[3],unit_am[0], label='unit x')
ax.plot(unit_am[3],tilt, label='tilt')
#ax.plot(unit_am[3],twist, label='twist')
ax.set_xlabel('Radius')
ax.set_ylabel('Unit angular momenta')
ax.legend()
plt.show()



