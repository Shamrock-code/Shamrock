#!/usr/bin/python3

########################################### IMPORTATIONS ###########################################
# general modules
import numpy as np
import matplotlib.pyplot as plt
import pylab


# needed for phantom
import sarracen

# path to phantom file
ph_dir = "/home/ylapeyre/track_bug1/sedov4/"
ph_file = ph_dir + "sedov_00300"
ph_file_init = ph_dir + "blast_00300"


########################################### READING ###########################################
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ reading phantom file ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ 
#sdf = sarracen.read_phantom(ph_file)

sdf = sarracen.read_phantom(ph_file)
sdf['r'] = np.sqrt(sdf['x']**2 + sdf['y']**2 + sdf['z']**2)
ctxt = sdf.describe()
params = sdf.params()
print(params) 

sdf.calc_density()



def myfunction(param1, param2):
    A = np.cos(param1) + param2

    return A

A = [1, 2, 3]
B = [12, 27, 49]

plt.plot(A, B)
plt.scatter(A, B, s=20)
plr.xlabel('mon array A')

plt.title('mon titre')

plt.show()
