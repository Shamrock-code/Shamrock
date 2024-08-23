

dic = {}


'''
PSMN CascadeLake
s92node01
CPU : Platinum 9242 @ 2.3GHz (48 cores, 2 socket)
RAM : 384 GiB
interconnect 100 GiB/s
'''
#dic["CascadeLake1"] = {
#    "label" : "CascadeLake",
#    "X" : [2,4,8,16,128],
#    "rate" : [981011.0412950808, 2041550.4552234707, 3657283.5229988615, 7577602.18079052, 42975622.415431604],
#    "cnt" : [3900960, 7916016, 15835008, 31808472, 254467776],
#    "lb_pred" : [100,100,100,100, 100]
#}

dic["CascadeLake1"] = {
    "label" : "CascadeLake",
    "X" : [2, 4, 8, 16, 32,64,128,152],
    "rate" : [1705829.042131229, 3328780.0407591304, 6709830.7891846, 10790180.093741834, 20591599.941228684,40850996.51509906, 63314184.38101712,76556249.52454272],
    "cnt" : [11838528, 23793664, 47434464, 95496480, 191092864,381677472, 767134656,910455552],
    "lb_pred" : [100, 100, 100, 100,100,100,100, 100]
}



'''
grvingt
'''
dic["grvingt 1"] = {
    "label" : "grvingt ",
    "X" : [2,4,6,8,10,12,14,16,20,24,28,32,40,52,64,80,100,120],
    "rate" : [
        591498.6491027079,
        1181681.1136507655,
        1553104.688674408,
        2116310.7803585893,
        2593814.229834617,
        3168616.5402065823,
        2383392.539410924,
        4684412.440914724,
        4846178.7416838575,
        6477531.774039453,
        6464480.133897541,
        5853629.434544097,
        9719550.417095121,
        10949372.839548606,
        10947479.134140132,
        13665466.536836417,
        15941812.629143752,
        19827022.50752665
    ],
    "cnt" : [
        3900960,
        7916016,
        11838528,
        15835008,
        19749120,
        23793664,
        27799200,
        31808472,
        39580800,
        47434464,
        55444480,
        63638560,
        79356680,
        103155712,
        127635200,
        159221952,
        199056000,
        238775328
    ],
    "lb_pred" : [
        99.99,
        97.92,
        90.08,
        100,
        84.96,
        82.57,
        78.57,
        98.59,
        72.71,
        66.19,
        65.20,
        88.55,
        90.54,
        88.59,
        97.64,
        83.19,
        81.68,
        78.51
    ]
}

'''
grvingt 2
'''

dic["grvingt 2"] = {
    "label" : "grvingt 2",
    "X" : [2,4,6,8,10,12,14,16,20,24,28,32,40,52,64,80,100,120],
    "rate" : [
        628358.5779473973,
        1124054.859004319,
        1736449.7664354416,
        2380810.3909499245,
        1917293.5540002272,
        3382532.054667773,
        3894101.199054512,
        4504015.621482006,
        5211853.5672429325,
        4301093.504322428,
        7283223.907001199,
        8527158.81533037,
        11048942.191861987,
        10292658.651045002,
        17199615.303962123,
        21550971.728999294,
        26018819.0099095,
        31249213.904896423
    ],
    "cnt" : [
        7916016,
        15835008,
        23793664,
        31808472,
        39580800,
        47434464,
        55444480,
        63638560,
        79356680,
        95496480,
        111213232,
        127635200,
        159221952,
        206595968,
        254467776,
        318404736,
        399787392,
        477884736
    ],
    "lb_pred" : [
        98.88,
        100,
        90.73,
        100,
        85.07,
        83.24,
        79.11,
        93.87,
        95.08,
        94.33,
        93.31,
        98.35,
        91.49,
        88.90,
        78.70,
        84.19,
        81.81,
        78.99
    ]
}


'''
grvingt 3
'''
dic["grvingt 3"] = {
    "label" : "grvingt 3",
    "X" : [2,4,6,8,10,12,14,16,20,24,28,32,40,52,64,80,100,120],

    "rate" : [

        592141.0651991414,
        1239845.601482153,
        1858803.6876634469,
        2379322.316033665,
        2720184.7882375405,
        3328598.2052814085,
        3837423.3467434007,
        4464796.424425585,
        5782054.636144191,
        7018958.537090696,
        8070815.635627935,
        9457773.055821262,
        10776156.334684804,
        15829854.006017553,
        17850911.895728435,
        20357623.0642507,
        25717710.869283035,
        31210662.41445782

    ],
    "cnt" : [

        15835008,
        31808472,
        47434464,
        63638560,
        79356680,
        95496480,
        111213232,
        127635200,
        159221952,
        191092864,
        223067520,
        254467776,
        318404736,
        415100928,
        510350208,
        637955440,
        797460048,
        957954192

    ],
    "lb_pred" : [
        100,
        100,
        91,
        97.45,
        97.69,
        96.73,
        96.11,
        99.11,
        95.68,
        94.09,
        93.12,
        88.72,
        91.28,
        89.28,
        79.75,
        96.94,
        95.79,
        94.75
    ]

}


import matplotlib.pyplot as plt
import numpy as np
plt.style.use('custom_short_cycler.mplstyle')


plt.figure()
for k in dic.keys():
    plt.plot(dic[k]["X"]  , np.array(dic[k]["rate"])  /np.array(dic[k]["X"]  ), label = dic[k]["label"])
#plt.ylim(0,1e6)
#plt.xlim(1,200)
plt.xscale('log')
plt.ylabel(r"$N_{\rm part} / (N_{\rm cpu} t_{\rm step})$")
plt.xlabel(r"$N_{\rm cpu}$")
plt.legend()
plt.grid()
plt.savefig("sedov_scalling_div_cpu.svg")


plt.figure()
for k in dic.keys():
    plt.plot(dic[k]["X"]  , (np.array(dic[k]["rate"])  /np.array(dic[k]["X"]  ))/(np.array(dic[k]["rate"])  /np.array(dic[k]["X"]  ))[0], label =  dic[k]["label"])
#plt.ylim(0,1e6)
#plt.xlim(1,200)
plt.xscale('log')
plt.ylabel(r"$\chi$")
plt.xlabel(r"$N_{\rm cpu}$")
plt.legend()
plt.grid()
plt.savefig("sedov_scalling_eff_cpu.svg")


plt.figure()
for k in dic.keys():
    plt.plot(dic[k]["X"]  , np.array(dic[k]["rate"]), label = dic[k]["label"])
#plt.ylim(0,1e6)
#plt.xlim(1,200)
plt.xscale('log')
plt.yscale('log')
plt.ylabel(r"$N_{\rm part} / (t_{\rm step})$")
plt.xlabel(r"$N_{\rm cpu}$")
plt.legend()
plt.grid()
plt.savefig("sedov_scalling_cpu.svg")





dic = {}





dic = {}


dic["Adastra mi250X 1e6"] = {
    "label" : "Adastra mi250X 1e6",
    "X" : [1*4, 2*4, 4*4, 8*4, 16*4,32*4,64*4,128*4,256*4],
    "rate" : [
        10027327.390461681,
        28627147.672835156,
        67501397.63723984,
        68893852.40823084,
        181130613.04604933,
        467921137.5994176,
        194426012.21058697,
        403097423.35557157,
        864848615.6035736
        ],
    "cnt" : [
        8157600,
        16217760,
        32416896,
        64603248,
        129168000,
        256893816,
        514205664,
        1029488400,
        2052189216
        ],
    "lb_pred" : [100, 100, 100, 100,100,100,100]
}
'''v1
dic["Adastra mi250X 8e6"] = {
    "label" : "Adastra mi250X 8e6",
    "X" : [1*4, 2*4, 4*4, 8*4, 16*4,32*4,64*4,256*4],
    "rate" : [
        25848206.663575538,
        59237902.16775592,
        109234312.63323641,
        196113292.237535,
        452113937.04432863,
        849201205.388938,
        847490119.5221043,
        3279951826.7928686
        ],
    "cnt" : [
        64603248,
        129168000,
        256893816,
        514205664,
        1029488400,
        2052189216,
        4107883904,
        16404962112
        ],
    "lb_pred" : [100, 100, 100, 100,100,100,100]
}
'''
dic["Adastra mi250X 8e6"] = {
    "label" : "Adastra mi250X 8e6",
    "X" : [1*4, 2*4, 4*4, 8*4, 16*4,32*4,64*4,128*4,256*4],
    "rate" : [
        26008574.678514816,
        58880254.89086981,
        81988685.92629065,
        197070974.04349208,
        450267279.825966,
        420014612.6583188,
        885810397.4549791,
        1937228587.1181426,
        156993712.01623392
        ],
    "cnt" : [
        64603248,
        129168000,
        256893816,
        514205664,
        1029488400,
        2052189216,
        4107883904,
        8219290896,
        16404962112
        ],
    "lb_pred" : [100, 100, 100, 100,100,100,100]
}

dic["Adastra mi250X 16e6"] = {
    "label" : "Adastra mi250X 16e6",
    "X" : [1*4, 2*4, 4*4, 8*4, 16*4,32*4,64*4,128*4,256*4],
    "rate" : [
        30716696.37947792,
        55392681.947984114,
        102484035.0776653,
        238320708.63068068,
        431453021.27965516,
        606635005.5261021,
        1330179076.937796,
        2293882848.760971,
        294004163.7261851
        ],
    "cnt" : [
        129168000,
        256893816,
        514205664,
        1029488400,
        2052189216,
        4107883904,
        8219290896,
        16404962112,
        32824341072
        ],
    "lb_pred" : [100, 100, 100, 100,100,100,100]
}

'''
dic["Adastra mi250X 20e6"] = {
    "label" : "Adastra mi250X 20e6",
    "X" : [1*4, 2*4, 4*4, 8*4, 16*4,32*4,64*4,128*4,256*4],
    "rate" : [
        28162471.616545185,
        65985664.576028824,
        126697959.30386999,
        219791447.40876472,
        455775333.11801857,

        431453021.27965516,
        606635005.5261021,
        1330179076.937796,
        2293882848.760971,
        3279951826.7928686
        ],
    "cnt" : [
        403064480,
        802649952,
        1606415328,
        3211389720,
        6421199616,

        2052189216,
        4107883904,
        8219290896,
        16404962112,
        16404962112
        ],
    "lb_pred" : [100, 100, 100, 100,100,100,100]
}
'''

dic["Adastra mi250X 1"] = {
    "label" : "Adastra mi250X 50e6/GPU (low patch count)",
    "X" : [1*4, 2*4, 4*4, 8*4, 16*4,32*4,64*4],
    "rate" : [
        28144452.5247059,
        65301034.81380231,
        126258518.4939734,
        220797309.04850662,
        451664700.5182303,
        850424568.3261222,
        1548188962.967435],
    "cnt" : [
        403064480,
        802649952,
        1606415328,
        3211389720,
        6421199616,
        12828966752,
        25658246736],
    "lb_pred" : [100, 100, 100, 100,100,100,100]
}

'''
dic["Adastra mi250X 2"] = {
    "label" : "Adastra mi250X 20e6/GPU",
    "X" : [1*4, 2*4, 4*4, 8*4,32*4,64*4],
    "rate" : [
        32147676.539863713,
        54075174.88014797,
        112557582.75639057,
        247867549.4295095,
        684742741.7572424,
        1404448922.0424242],
    "cnt" : [
        160997760,
        321221120,
        642428928,
        1283517664,
        5128194048,
        10259483520],
    "lb_pred" : [100, 100, 100, 100,100,100,100]
}
'''
'''
dic["Adastra mi250X 3"] = {
    "label" : "Adastra mi250X 10e6/GPU",
    "X" : [1*4, 2*4, 4*4],
    "rate" : [
        28667177.60030524,
        61183024.82153851,
        105126282.02959135
        ],
    "cnt" : [
        80474112,
        160997760,
        321221120
        ],
    "lb_pred" : [100, 100, 100, 100,100,100,100]
}
'''

dic = {}


dic["Adastra mi250X 1e6"] = {
    "label" : "1e6 parts / GPUs",
    "X" : [1*4, 2*4, 4*4, 8*4, 16*4,32*4,64*4,128*4,256*4],
    "rate" : [
        10705666.585170781,
        30074005.10485011,
        24222645.776111893,
        77221476.61925279,
        215989609.6768017,
        179888954.78658682,
        589484332.7888682,
        1653007675.187291,
        1375007011.974956
        ],
    "cnt" : [
        8157600,
        16217760,
        32416896,
        64603248,
        129168000,
        256893816,
        514205664,
        1029488400,
        2052189216
        ],
    "lb_pred" : [100, 100, 100, 100,100,100,100]
}

dic["Adastra mi250X 8e6"] = {
    "label" : "8e6 parts / GPUs",
    "X" : [1*4, 2*4, 4*4, 8*4, 16*4,32*4,64*4,128*4,256*4],
    "rate" : [
        27925165.0051521,
        70589467.85605815,
        89884388.05374768,
        216719945.62412375,
        532159209.383341,
        709226620.418723,
        1700242124.593388,
        4046052257.6171503,
        5342278044.610529
        ],
    "cnt" : [
        64603248,
        129168000,
        256893816,
        514205664,
        1029488400,
        2052189216,
        4107883904,
        8219290896,
        16404962112
        ],
    "lb_pred" : [100, 100, 100, 100,100,100,100]
}

dic["Adastra mi250X 16e6"] = {
    "label" : "16e6 parts / GPUs",
    "X" : [1*4, 2*4, 4*4, 8*4, 16*4,32*4,64*4,128*4,256*4],
    "rate" : [
        37025358.19356769,
        70180062.59854978,
        112351276.67165202,
        289277358.35634714,
        547930990.8163583,
        888915130.1262488,
        2240839098.6210938,
        4111305966.830165,
        6757111637.777441
        ],
    "cnt" : [
        129168000,
        256893816,
        514205664,
        1029488400,
        2052189216,
        4107883904,
        8219290896,
        16404962112,
        32824341072
        ],
    "lb_pred" : [100, 100, 100, 100,100,100,100]
}



dic["Adastra mi250X 32e6"] = {
    "label" : "32e6 parts / GPUs",
    "X" : [1*4, 2*4, 4*4, 8*4, 16*4,32*4,64*4,128*4,256*4],
    "rate" : [
        37263623.46191787,
        74708547.71419899,
        149063290.90710586,
        285482136.6247805,
        554224717.891204,
        1143403466.2394433,
        2263933822.3465977,
        4263623185.706088,
        8729139669.248592
        ],
    "cnt" : [
        256893816,
        514205664,
        1029488400,
        2052189216,
        4107883904,
        8219290896,
        16404962112,
        32824341072,
        65656270200
        ],
    "lb_pred" : [100, 100, 100, 100,100,100,100]
}



from matplotlib.ticker import ScalarFormatter


fig = plt.figure()
ax = fig.add_subplot(111)
for k in dic.keys():
    X = dic[k]["X"]
    Y = np.array(dic[k]["rate"])  /np.array(dic[k]["X"]  )
    p, = plt.plot(X,Y, label = dic[k]["label"])
    plt.annotate(f'{Y[-1]:.2e}', xy=(1.01,Y[-1]), xycoords=('axes fraction', 'data'),
                     ha='left', va='center', color=p.get_color())
#plt.ylim(0,1e6)
#plt.xlim(1,200)
plt.xscale('log')
ax.xaxis.set_major_formatter(ScalarFormatter())

#plt.yscale('log')
#plt.ylabel(r"$N_{\rm part} / (N_{\rm GPU} t_{\rm step})$")
plt.ylabel(r"Particles / seconds / GPU")
plt.xlabel(r"GPUs")
plt.title("Adastra mi250x")
plt.legend()
plt.grid()
plt.savefig("sedov_scalling_div_GPU.svg")


fig = plt.figure()
ax = fig.add_subplot(111)
for k in dic.keys():
    X = dic[k]["X"]
    Y = np.array(dic[k]["rate"])  / ( 1900 * np.array(dic[k]["X"]) / 4)
    p, = plt.plot(X,Y, label = dic[k]["label"])
    plt.annotate(f'{Y[-1]:.2e}', xy=(1.01,Y[-1]), xycoords=('axes fraction', 'data'),
                     ha='left', va='center', color=p.get_color())
#plt.ylim(0,1e6)
#plt.xlim(1,200)
plt.xscale('log')
ax.xaxis.set_major_formatter(ScalarFormatter())

#plt.yscale('log')
#plt.ylabel(r"$N_{\rm part} / (N_{\rm GPU} t_{\rm step})$")
plt.ylabel(r"Particles / seconds / Watt")
plt.xlabel(r"GPUs")
plt.title("Adastra mi250x")
plt.legend()
plt.grid()
plt.savefig("sedov_scalling_energy_GPU.pdf")



fig = plt.figure()
ax = fig.add_subplot(111)
idx = 0
pos_anot = [0.5, 0.78, 0.70, 0.92]
for k in dic.keys():
    X = dic[k]["X"]
    Y = (np.array(dic[k]["rate"])  /np.array(dic[k]["X"]  ))/(np.array(dic[k]["rate"])  /np.array(dic[k]["X"]  ))[0]
    p, = plt.plot(X,Y, label = dic[k]["label"])
    plt.annotate(f'{Y[-1]:.2f}', xy=(1.01,pos_anot[idx]), xycoords=('axes fraction', 'data'),
                     ha='left', va='center', color=p.get_color())
    idx += 1
plt.ylim(0,1.5)
#plt.xlim(1,200)
plt.xscale('log')
ax.xaxis.set_major_formatter(ScalarFormatter())

plt.ylabel(r"Parallel efficiency ")
plt.xlabel(r"GPUs")
plt.title("Adastra mi250x")
plt.legend()
plt.grid()
plt.savefig("sedov_scalling_eff_GPU.svg")


fig = plt.figure()
ax = fig.add_subplot(111)
idx = 0
pos_anot = [1.38e9, 5.e9,7e9, 10e9]
for k in dic.keys():
    X = dic[k]["X"]
    Y = np.array(dic[k]["rate"])
    p, = plt.plot(X,Y, label = dic[k]["label"])
    plt.annotate(f'{Y[-1]:.2e}', xy=(1.01,pos_anot[idx]), xycoords=('axes fraction', 'data'),
                     ha='left', va='center', color=p.get_color())
    idx += 1

#plt.ylim(0,1e6)
#plt.xlim(1,200)
plt.xscale('log')
plt.yscale('log')
ax.xaxis.set_major_formatter(ScalarFormatter())

plt.ylabel(r"Particles / seconds")
plt.xlabel(r"GPUs")
plt.legend()
plt.title("Adastra mi250x")
plt.grid()
plt.savefig("sedov_scalling_GPU.svg")
