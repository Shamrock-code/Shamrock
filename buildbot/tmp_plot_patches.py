from itertools import combinations, product

import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits import mplot3d
from mpl_toolkits.mplot3d.art3d import Line3DCollection, Poly3DCollection


def get_cube(xm, ym, zm, xp, yp, zp):
    Z = [
        [xm, ym, zm],
        [xm, ym, zp],
        [xm, yp, zm],
        [xm, yp, zp],
        [xp, ym, zm],
        [xp, ym, zp],
        [xp, yp, zm],
        [xp, yp, zp],
    ]

    verts = [
        [Z[0], Z[1], Z[3], Z[2]],
        [Z[4], Z[5], Z[7], Z[6]],
        [Z[0], Z[1], Z[5], Z[4]],
        [Z[2], Z[3], Z[7], Z[6]],
        [Z[1], Z[3], Z[7], Z[5]],
        [Z[4], Z[6], Z[2], Z[0]],
    ]
    return verts


def draw_cube(xm, ym, zm, xp, yp, zp, color, alpha):
    ax.add_collection3d(
        Poly3DCollection(
            get_cube(xm, ym, zm, xp, yp, zp),
            facecolors=color,
            linewidths=1,
            edgecolors="black",
            alpha=alpha,
        )
    )


import sys

prop_cycle = plt.rcParams["axes.prop_cycle"]
colors = prop_cycle.by_key()["color"]

colseq = ["cyan", "orange", "blue", "green", "grey", "magenta"]

patch_dic = {}


iteration = int(sys.argv[1])
node = int(sys.argv[2])

patch_filelist = ["patches_" + str(iteration) + "_node" + str(n) for n in range(node)]

it = 0
for a in patch_filelist:
    fil = open(a, "r")

    for l in fil.readlines():
        ll = l.split("|")[:-1]

        print(ll)

        xmin = float(ll[-6])
        xmax = float(ll[-5])

        ymin = float(ll[-4])
        ymax = float(ll[-3])

        zmin = float(ll[-2])
        zmax = float(ll[-1])

        # draw_cube(xmin, ymin, zmin, xmax, ymax, zmax, colseq[it])

        id = int(ll[0])

        patch_dic[id] = (xmin, ymin, zmin, xmax, ymax, zmax, it)

    fil.close()

    it = it + 1

# plt.show()


interfaces_dic = {}

for k in patch_dic.keys():
    interfaces_dic[k] = []

interf_filelist = ["interfaces_" + str(iteration) + "_node" + str(n) for n in range(4)]
it = 0
for a in interf_filelist:
    fil = open(a, "r")

    for l in fil.readlines():
        ll = l.split("|")[:-1]

        psend_id = int(ll[2])
        precv_id = int(ll[3])

        xmin = float(ll[-6]) - 1e-5
        xmax = float(ll[-5]) + 1e-5

        ymin = float(ll[-4]) - 1e-5
        ymax = float(ll[-3]) + 1e-5

        zmin = float(ll[-2]) - 1e-5
        zmax = float(ll[-1]) + 1e-5

        interfaces_dic[psend_id].append((psend_id, precv_id, xmin, ymin, zmin, xmax, ymax, zmax))

        print((psend_id, precv_id, xmin, ymin, zmin, xmax, ymax, zmax))

    fil.close()

    it = it + 1


def draw_patch(id):
    (xmin, ymin, zmin, xmax, ymax, zmax, it) = patch_dic[id]

    draw_cube(xmin, ymin, zmin, xmax, ymax, zmax, colors[it % len(colors)], 1)


def draw_interface_send(id):
    it = 0
    for interf in interfaces_dic[id]:
        (psend_id, precv_id, xmin, ymin, zmin, xmax, ymax, zmax) = interf

        print(
            (psend_id, precv_id, xmax - xmin, ymax - ymin, zmax - zmin),
            (xmax - xmin) * (ymax - ymin) * (zmax - zmin),
        )

        if (xmax - xmin) * (ymax - ymin) * (zmax - zmin) > 0.4**3:
            print("error :!!!!!")

        draw_cube(xmin, ymin, zmin, xmax, ymax, zmax, "grey", 0.1)

        it = it + 1


def draw_interface_send_list(idlist):
    vert_lst = []

    for id in idlist:
        it = 0
        for interf in interfaces_dic[id]:
            (psend_id, precv_id, xmin, ymin, zmin, xmax, ymax, zmax) = interf

            print(
                (psend_id, precv_id, xmax - xmin, ymax - ymin, zmax - zmin),
                (xmax - xmin) * (ymax - ymin) * (zmax - zmin),
            )

            if (xmax - xmin) * (ymax - ymin) * (zmax - zmin) > 0.4**3:
                print("error :!!!!!")

            vert_lst += get_cube(xmin, ymin, zmin, xmax, ymax, zmax)

            it = it + 1

    ax.add_collection3d(
        Poly3DCollection(vert_lst, facecolors="grey", linewidths=1, edgecolors="black", alpha=0.1)
    )


fig = plt.figure(figsize=(10, 7))
ax = plt.axes(projection="3d")
ax.scatter3D([-1, 1], [-1, 1], [-1, 1])

for k in patch_dic.keys():
    draw_patch(k)

# for k in patch_dic.keys():
#     draw_interface_send(k)

# draw_patch(15)
# draw_patch(17)
# draw_interface_send(15)
# draw_interface_send(17)


plt.show()
