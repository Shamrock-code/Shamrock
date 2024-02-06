import json
import matplotlib.pyplot as plt 
import numpy as np
import random


files = ["sedov_scale_test_2_0", "sedov_scale_test_2_1","sedov_scale_test_init_2_0","sedov_scale_test_init_2_1"]

def load_dic(fname):
    fl = open(fname,'r') 
    str_f = fl.read()
    fl.close()

    arr = json.loads(str_f)

    dic = {}

    for entry in arr:
        if not (entry["name"] in dic.keys()):
            dic[entry["name"]] = []

        dic[entry["name"]].append(entry["tend"] - entry["tstart"])
    return dic

# load data
dic_lst = [load_dic(fname) for fname in files]

# make label map
counter = 0
dic_labels = {}

for dic in dic_lst:
    for k in dic.keys():
        if not(k in dic_labels.keys()):
            dic_labels[k] = counter
            counter += 1

# label list for the plot
labels = [k for k in dic_labels.keys()]

def rnd():
    return random.uniform(-1, 1)


fig,axs = plt.subplots(nrows=1,ncols=1,figsize=(20,10),dpi=90)

colors = ["tab:blue" ,
    "tab:orange" ,
    "tab:green",
    "tab:red",
    "tab:purple", 
    "tab:brown" ,
    "tab:pink" ,
    "tab:gray" ,
    "tab:olive",
    "tab:cyan"]

#colors = ['pink', 'lightblue', 'lightgreen']
cnt = 0
first = True
for dic in dic_lst:
    for k in dic.keys():
        xoff = dic_labels[k]
        X = [xoff + 0.2*rnd() for i in range(len(dic[k]))]

        if first:
            plt.scatter(X,dic[k], color = colors[cnt % len(colors)], label = files[cnt])
            first = False
        else:
            plt.scatter(X,dic[k], color = colors[cnt % len(colors)])


    first = True
    cnt += 1

plt.xticks([i for i in range(len(labels))], labels, rotation = "vertical")
plt.tight_layout()
plt.legend()
# fill with colors
#colors = ['pink', 'lightblue', 'lightgreen']
#for patch, color in zip(bplot1['boxes'], colors):
#    patch.set_facecolor(color)

plt.yscale('log')



plt.show()