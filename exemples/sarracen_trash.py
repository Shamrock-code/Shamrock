
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
