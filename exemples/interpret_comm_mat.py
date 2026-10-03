out = r"""
0 # 1 # 2717888
0 # 2 # 2717888
0 # 3 # 2693072
1 # 0 # 2725152
1 # 2 # 2712656
1 # 3 # 2724976
2 # 0 # 2714592
2 # 1 # 2723040
2 # 3 # 2714768
3 # 0 # 2720752
3 # 1 # 2716704
3 # 2 # 2716880
"""

lines = out.split("\n")

lst_comm = {}

for l in lines[1:-1]:
    s = l.split("#")
    sender = int(s[0])
    receiver = int(s[1])
    size = int(s[2])

    lst_comm[(sender, receiver)] = size


max_rank = 0

for k in lst_comm.keys():
    max_rank = max(max_rank, max(k))

import numpy as np

mat_comm = np.zeros((max_rank + 1, max_rank + 1))

for k in lst_comm.keys():
    mat_comm[k[0], k[1]] = lst_comm[k]

print(mat_comm)

import matplotlib.pyplot as plt

plt.imshow(mat_comm, cmap="viridis", aspect="auto", extent=(0, max_rank + 1, max_rank + 1, 0))
plt.colorbar(label="Communication Size")
plt.xticks(range(max_rank + 2))
plt.yticks(range(max_rank + 2))
plt.show()
