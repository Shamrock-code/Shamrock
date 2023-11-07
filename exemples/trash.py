import matplotlib.pyplot as plt
import numpy as np

outputdir = "/home/ylapeyre/track_bug1/KH_0sss/"
ph_dir = "/home/ylapeyre/track_bug1/KH_0/"
ph_file = ph_dir + "kh_00000"
ev_f = ph_dir + "kh01.ev"

ev_dic = {}
with open(ev_f, 'r') as phantom_ev:
    # read the col names
    #columns = phantom_ev.readline().strip().split()
    #print(columns)

    columns = ['time', 
               'ekin',
               'etherm',
               'emag',
               'epot',
               'etot',
               'erad',
               'totmom',
               'angtot',
               'rho_max',
               'rho_avg',
               'dt',
               'totentrop',
               'rmdmzch',
               'vrms',
               'xcom',
               'ycom',
               'zcom'
               'alpha_max']
    ev_data = np.genfromtxt(ev_f, skip_header=1)
    ev_data = ev_data.T

    i_dic = 0
    for column in columns:
        ev_dic[column] = ev_data[i_dic]
        i_dic +=1

print(ev_dic['dt'][490:])

current_dt = ev_dic['dt'][0]
t_sum = 0

for next_dt in ev_dic['dt'][1:]:
    t_sum += current_dt
    current_dt = next_dt

print(t_sum)