# -*- coding: utf-8 -*-
"""
Created on Thu Jun  2 16:16:25 2022

@author: yona

animation for gas only
"""

import numpy as np
from matplotlib import pyplot as plt
#plt.rcParams['animation.ffmpeg_path'] = r'C:\\ffmpeg\\bin\\ffmpeg.exe'
import matplotlib.pylab as pylab
import matplotlib.animation as manimation
import pandas as pd
import sarracen

import os

Rin = 0.2 #/ sicte.au() 
G = ucte.G()
orbital_period = 2 * np.pi * np.sqrt((Rin**3) / (central_mass * G)) #physical units
print("################## orbital period = {} ##################".format(orbital_period))

N_time = 130
bins = 500
N_size = 50
columnF = 5
columnX = 1
title = 'twist'
outputdir = "/Users/ylapeyre/Documents/Shamwork/23_04/test_mass/"

plt.rc("text", usetex = True)
#plt.rcParams['text.usetex'] = True
my_parameters = {'legend.fontsize':'16','figure.figsize':(8,6),'axes.labelsize':'16','axes.titlesize':'16','xtick.labelsize':'16','ytick.labelsize':'16'}
pylab.rcParams.update(my_parameters)

def extract_data(filename, skip_header=3):
    
    file = np.genfromtxt(filename, dtype="float", skip_header=skip_header)
    # m, n = file.shape
    # I = []
    # X = []
    # shaped_file = np.reshape(file, (3, 100))
    
    return file.T


F = []
G = []
X = []
N = []

numbers = ["%04d" % t for t in range (bins +1)]

for i in range (0, N_time+1, 1):
    number = str(i).zfill(4)
    file = outputdir + 'dump_' + str(i).zfill(4)
    sdf = sarracen.read_phantom(file)
    unit_am = sarracen.disc.angular_momentum(sdf,r_in= 0.1, r_out= 2, bins=500,retbins=True, origin=[0.,0.,0.])
    tilt = np.arccos(unit_am[2])
    twist = np.arctan(unit_am[2] / unit_am[0])
    for j in range(500):
        F.append(twist[j])
        G.append(tilt[j])
        X.append(unit_am[3][j])
        N.append(int(number))
        
X = np.array(X)
F = np.array(F)
G = np.array(G)
N = np.array(N)

data = pd.DataFrame({'X': X,
                     'F': F,
                     'G': G,
                     'N': N
                    })

fig, (ax0, ax1) = plt.subplots(nrows=1, ncols=2, sharex=True)
#ax = plt.axes(xlim=(-0.1, 1.1), ylim=(-1, 1)) #F
#ax = plt.axes(xlim=(0, 2), ylim=(0, 2)) #density
ax0.set_xlim(0, 2)
ax0.set_ylim(0, 2)
ax0.set_xlabel("r")
ax0.set_ylabel("twist")

ax1.set_ylabel("tilt")
ax1.set_xlim(0, 2)
ax1.set_ylim(0, 2)

#x = data.loc[data['N'] == 0, 'X']
#y = data.loc[data['N'] == 0, 'F']
#ax.plot(x, y, color = "black")

scatf = ax0.scatter([], [], s=20)
linef, = ax0.plot([], [])
scatg = ax1.scatter([], [], s=20)
lineg, = ax1.plot([], [])

time_text = ax1.text(0.8, 0.001, "iteration = {}".format(""), fontsize=15)
def animate(i):
    
    x = data.loc[data['N'] == i, 'X']
    y = data.loc[data['N'] == i, 'F']
    g = data.loc[data['N'] == i, 'G']
    #scat.set_offsets(np.array([x, y]).T)
    
    X = x.to_numpy()
    Y = y.to_numpy()
    G = g.to_numpy()

    linef.set_data(X, Y)
    linef.set_label("t = {}".format(i))

    lineg.set_data(X, G)
    lineg.set_label("t = {}".format(i))
    time_text.set_text("t = {}".format(i))
    #scat.set_color('blue')
    ax0.legend()
    ax1.legend()
    
    return scatf,
    
anim = manimation.FuncAnimation(fig, animate, frames=N_time+1, interval=100000, repeat=False)

metadata = dict(title=title, artist='Y. LAPEYRE',
            comment='temporal evolution')

writervideo = manimation.FFMpegWriter(fps=2, metadata=metadata, bitrate= -1)
writergif = manimation.PillowWriter(fps=15, metadata=metadata)

anim.save(outputdir + "anim_500" +".mp4", writer=writervideo)
#plt.show()