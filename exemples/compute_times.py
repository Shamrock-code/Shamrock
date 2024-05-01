# -*- coding: utf-8 -*-
"""
Created on Thu Jun  2 16:16:25 2022

@author: yona

Compute precession time
"""

import numpy as np
from matplotlib import pyplot as plt
#plt.rcParams['animation.ffmpeg_path'] = r'C:\\ffmpeg\\bin\\ffmpeg.exe'
import matplotlib.pylab as pylab
import matplotlib.animation as manimation
import pandas as pd
import sarracen
from scipy.signal import savgol_filter
from scipy.ndimage import gaussian_filter1d
from scipy.fft import fft
from scipy.optimize import curve_fit

import os

plt.rc("text", usetex = True)
plt.rcParams['text.usetex'] = True
my_parameters = {'legend.fontsize':'16','figure.figsize':(8,6),'axes.labelsize':'16','axes.titlesize':'16','xtick.labelsize':'16','ytick.labelsize':'16'}
pylab.rcParams.update(my_parameters)

################################ INPUT PARAMETERS ##################################
a = 0.9
central_mass = 1e6
Rin = 0.2 #/ sicte.au() 
G = 40
orbital_period = 2 * np.pi * np.sqrt((Rin**3) / (central_mass * G)) #physical units
dt_dump = orbital_period / 3.
print("################## orbital period = {} ##################".format(orbital_period))

N_time = 1247
start = 0
bins = 50
outputdir = "/Users/ylapeyre/Documents/Shamwork/30_04/test1/"

TWIST = []
TIME = []
N = []

############################ BUILT DIC WITH ALL DATA ##################################
for i in range (start, N_time+1, 1):
    number = str(i).zfill(4)
    file = outputdir + 'dump_' + str(i).zfill(4)
    sdf = sarracen.read_phantom(file)
    unit_am = sarracen.disc.angular_momentum(sdf,r_in= 0.1, r_out= 2, bins=bins,retbins=True, origin=[0.,0.,0.])
    #tilt = np.arccos(unit_am[2])
    twist = np.arctan(unit_am[2] / unit_am[0])
    for j in range(bins):
        TWIST.append(twist[j])
        TIME.append(unit_am[3][j])
        N.append(int(number))
        
TIME = np.array(TIME)
TWIST = np.array(TWIST)
N = np.array(N)

######################### FIND PERIOD FOR EACH RADIUS #################################
################################## USING FFT ##########################################
PERIOD = []
RADIUS = []
RADIUS_ID = []

PERIOD_fit = []

chosen_id = 21
do_plot = True

for radius_id in range (bins):
    twist_at_id = []
    time = []
    for j in range (N_time +1 - start):
        print(j)
        twist_at_id.append(TWIST[j * bins +radius_id])
        time.append(j)

    smooth_twist = savgol_filter(twist_at_id, 10, 3)
   # smooth_twist_nonan = list(smooth_twist)
    #smooth_twist = gaussian_filter1d(twist_at_id, 2)
    #print(len(smooth_twist_nonan))
    #print(len(time))

    #inan = 0
    #for x in smooth_twist:
    #    if np.isnan(x):
    #        del time[inan]
    #        del smooth_twist_nonan[inan]
    #    inan +=1
    
    smooth_twist_nonan = [x for x in smooth_twist if not np.isnan(x)]
    time_nonan =[time[i] for i in range (len(time)) if not np.isnan(smooth_twist[i])]
    #print(len(smooth_twist_nonan))
    #print(len(time_nonan))
    time = np.array(time)

    if len(smooth_twist_nonan) < 10:
        print("[FFT] Cannot compute FFT on empty array. Mean radius at id {}".format(radius_id))
    else:
        fft_result = fft(smooth_twist_nonan)
        frequencies = np.fft.fftfreq(len(smooth_twist_nonan))
        # Find the peak frequency (excluding the DC component at index 0)
        peak_index = np.argmax(np.abs(fft_result[1:len(smooth_twist_nonan)//2])) + 1
        peak_frequency = frequencies[peak_index]
        # Calculate the period from the peak frequency
        period = 1 / abs(peak_frequency)

        RADIUS_ID.append(radius_id)
        RADIUS.append(unit_am[3][radius_id])
        PERIOD.append(period)

        def sinusoidal_function(t, A, omega, phi, offset):
            return A * np.sin(omega * t + phi) + offset 

        smooth_twist_nonan = np.array(smooth_twist_nonan)
        time_nonan = np.array(time_nonan)
        #popt, pcov = curve_fit(sinusoidal_function, time_nonan, smooth_twist_nonan, p0=(0.1, 0.6981317007977318, 0, 1.26))

        # Extract fitted parameters
        #fitted_A, fitted_omega, fitted_phi, fitted_offset = popt
        #print(popt)
        # Calculate period from angular frequency
        #fitted_period = 2 * np.pi / fitted_omega
        #PERIOD_fit.append(fitted_period)


    if do_plot and radius_id==chosen_id:
        fig, axs = plt.subplots(1, 2)
        axs[0].scatter(frequencies[:len(smooth_twist_nonan)//2], np.abs(fft_result[:len(smooth_twist_nonan)//2]), s=1, marker='.')
        axs[0].set_title('FFT')
        axs[0].set_xlabel('Frequency')
        axs[0].set_ylabel('Amplitude')

        axs[1].plot(time /20, twist_at_id, c='k', marker='x')
        #axs[1].plot(time, smooth_twist, c='r', label='Savitzky-Golay filter', marker='x')
        #axs[1].plot(time, smooth_twist, c='b', label='Gaussian 1D filter')
        axs[1].set_xlabel('time (orbits)')
        axs[1].set_ylabel('twist')
        axs[1].set_title('at radius R = {}'.format(unit_am[3][radius_id]))

        #auto_corr = np.correlate(smooth_twist_nonan, smooth_twist_nonan, mode='full')
        #axs[2].plot(auto_corr)
        #axs[2].set_title('autocorr')

        # Plot the signal and the fitted curve
        #axs[2].plot(time_nonan, smooth_twist_nonan, marker='x', label='Data', c='r')
        #axs[2].plot(time, sinusoidal_function(time, fitted_A, fitted_omega, fitted_phi, fitted_offset), c='b', label='Fitted curve')
        #axs[2].set_xlabel('time (/3 for approximately in orbits)')
        #axs[2].set_ylabel('twist')
        #axs[2].set_title('Curve Fitting for Period Estimation')

        # Compute the central differences
        dt = np.gradient(time)  # Assuming x is equally spaced
        dtwist_dt = np.gradient(smooth_twist_nonan)
        print('##########################################')
        print('############# dtwist = {} ##############'.format(dtwist_dt))
        print('##########################################')
        #ax[4].plot(dtwist_dt, time)
        plt.legend()
        plt.show()


######################## SHOW PERIOD EVOLUTION #################################

RADIUS = np.array(RADIUS)
print(np.log(PERIOD))
radius_th = np.linspace(0.2, 2)
period_th = [(r**3) / (2*a) for r in radius_th]
plt.plot(radius_th/Rin, period_th, c='r', label='theory')
print("$$$$$$$$$$$$$$$ period curvefitting = {}".format(PERIOD_fit))
plt.scatter(RADIUS/Rin, PERIOD, c='g', s=5, marker = "x", label='FFT')
#plt.scatter(RADIUS_ID, PERIOD_fit, c='orange', s=1, marker = ".", label='curve-fitting')
plt.xlabel('R / Rin')
plt.ylabel('precession time')
plt.legend()
plt.yscale('log')
plt.show()







 
