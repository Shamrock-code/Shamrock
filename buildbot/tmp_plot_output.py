import glob
import struct
import sys

import matplotlib.pyplot as plt
import numpy as np
import pyvista as pv

# mayavi.mlab.options.backend = 'envisage'

# mayavi.mlab.figure(figure=None, bgcolor=None, fgcolor=None, engine=None, size=(1920, 1080))


frame = []


def get_plot_patchdata(filename):
    f = open(filename, "rb")

    header = f.read(24)
    header_unpacked = list(struct.unpack("IIIIII", header))

    dic = {}

    desc_nam = ["x", "y", "z"]
    desc_len = [4, 4, 4]
    tot_len = sum(desc_len)

    for a in desc_nam:
        dic[a] = []

    byte = "a"
    for i in range(header_unpacked[0]):
        byte = f.read(tot_len)
        x, y, z = struct.unpack("fff", byte)

        dic["x"].append(x)
        dic["y"].append(y)
        dic["z"].append(z)

    desc_nam = ["h", "omega"]
    desc_len = [4, 4]
    tot_len = sum(desc_len)

    for a in desc_nam:
        dic[a] = []

    byte = "a"
    for i in range(header_unpacked[0]):
        byte = f.read(tot_len)
        h, om = struct.unpack("ff", byte)

        dic["h"].append(h)
        dic["omega"].append(om)

    f.close()

    dic["x"] = dic["x"][::]
    dic["y"] = dic["y"][::]
    dic["z"] = dic["z"][::]
    dic["h"] = dic["h"][::]

    # print(dic["h"][:1000])

    # mayavi.mlab.points3d(dic["x"],dic["y"],dic["z"],dic["h"], scale_factor=0.5)

    dic_filtered = {
        "x": [],
        "y": [],
        "z": [],
        "h": [],
    }

    for i in range(len(dic["x"])):
        cd = True
        # cd = cd and dic["y"][i] < 0.1 and dic["y"][i] > -0.1
        # cd = cd and dic["z"][i] < 0.3e-2 and dic["z"][i] > -0.3e-2

        if cd:
            dic_filtered["x"].append(dic["x"][i])
            dic_filtered["y"].append(dic["y"][i])
            dic_filtered["z"].append(dic["z"][i])
            dic_filtered["h"].append(dic["h"][i])

    points = np.zeros((len(dic_filtered["x"]), 3))

    points[:, 0] = np.array(dic_filtered["x"])
    points[:, 1] = np.array(dic_filtered["y"])
    points[:, 2] = np.array(dic_filtered["z"])

    point_cloud = pv.PolyData(points)

    h = np.array(dic_filtered["h"])
    hfac = 1.2
    m = 1e-5

    point_cloud["hpart"] = h
    point_cloud["rho"] = m * (hfac / h) * (hfac / h) * (hfac / h)
    # point_cloud.plot(eye_dome_lighting=True)
    # plotter.add_mesh(point_cloud,scalars='hpart', cmap="viridis", render_points_as_spheres=True)

    return point_cloud

    # plt.scatter(dic_filtered["x"], dic_filtered["y"], c=dic_filtered["h"], cmap='nipy_spectral',vmin=0,vmax=0.025)


def loading_frames():
    idx = 0

    while True:
        file_list = glob.glob("./step" + str(idx) + "/patchdata*")
        print(file_list)

        if len(file_list) == 0:
            break

        tmp = []

        for i in file_list:
            print("plotting : {}".format(i))
            tmp.append(get_plot_patchdata(i))

        frame.append(tmp)

        idx = idx + 1


def plot_content():
    for f in frame:
        plotter = pv.Plotter(window_size=([1920, 1080]))
        for p in f:
            plotter.add_mesh(p, scalars="hpart", cmap="viridis", render_points_as_spheres=True)
        plotter.show_grid()
        plotter.show()


def make_gif():
    plotter = pv.Plotter(window_size=([1920, 1080]), notebook=False, off_screen=True)
    plotter.open_gif("out.gif")
    for f in frame:
        for p in f:
            plotter.add_mesh(
                p,
                scalars="hpart",
                cmap="viridis",
                clim=[0.0150, 0.03],
                render_points_as_spheres=True,
            )
        plotter.show_grid()
        plotter.write_frame()
        plotter.clear()
    plotter.close()


def make_gif_with_load():
    plotter = pv.Plotter(window_size=([1920, 1080]), notebook=False, off_screen=True)
    plotter.open_gif("out.gif")

    idx = 0

    while True:
        file_list = glob.glob("./step" + str(idx) + "/patchdata*")
        print(file_list)

        if len(file_list) == 0:
            break

        tmp = []

        for i in file_list:
            print("plotting : {}".format(i))
            tmp.append(get_plot_patchdata(i))

        idx = idx + 1

        for p in tmp:
            plotter.add_mesh(
                p, scalars="rho", cmap="viridis", render_points_as_spheres=True, clim=[1.5, 3]
            )
        plotter.show_grid()
        plotter.write_frame()
        plotter.clear()

    plotter.close()


def make_movie_with_load():
    plotter = pv.Plotter(window_size=([1920, 1080]), notebook=False, off_screen=True)
    plotter.open_movie("out.mp4")

    idx = 0

    while True:
        file_list = glob.glob("./step" + str(idx) + "/patchdata*")
        print(file_list)

        if len(file_list) == 0:
            break

        tmp = []

        for i in file_list:
            print("plotting : {}".format(i))
            tmp.append(get_plot_patchdata(i))

        idx = idx + 1

        for p in tmp:
            plotter.add_mesh(
                p, scalars="rho", cmap="viridis", render_points_as_spheres=True, clim=[1.5, 3]
            )
        plotter.show_grid()
        plotter.write_frame()
        plotter.clear()

    plotter.close()


def make_figs():
    plotter = pv.Plotter(window_size=([1920, 1080]), off_screen=True)
    idx = 0
    for f in frame:
        for p in f:
            plotter.add_mesh(p, scalars="hpart", cmap="viridis", render_points_as_spheres=True)
        plotter.show_grid()
        # plotter.save_graphic("step"+str(idx)+".pdf")
        plotter.show(screenshot="step" + str(idx) + ".png")
        plotter.clear()

        idx = idx + 1
    plotter.close()


# loading_frames()
# make_gif()
# plot_content()

make_gif_with_load()
# make_movie_with_load()

# plt.colorbar()

# plt.show()
# mayavi.mlab.show()
# print("saving ...")
# mayavi.mlab.savefig(sys.argv[1])
