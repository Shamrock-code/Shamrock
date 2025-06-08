import glob
import struct
import sys

import matplotlib.pyplot as plt
import numpy as np


def extract(header, data):
    tmp = data.split(header)
    return tmp[0], tmp[1]


def read_header_field(header_f, dic_res):
    tmp = header_f

    while len(tmp) > 0:
        tmp_in = tmp[:72]
        tmp = tmp[72:]

        nb = tmp_in[:64]
        nvarb = tmp_in[64:68]
        obj_cntb = tmp_in[68:72]

        name = str(nb.decode("utf-8")).rstrip("\x00")
        nvar = int.from_bytes(nvarb, "little")
        obj_cnt = int.from_bytes(obj_cntb, "little")

        dic_res.append({"name": name, "nvar": nvar, "obj_cnt": obj_cnt})


def get_plot_patchdata(filename):
    f = open(filename, "rb")

    data = f.read()

    data = data.split(b"##header end##\n\0")

    head = data[0].split(b"##header start##")[1]
    data = data[1]

    tmp, head = extract(b"#f32\0\0\0\0", head)
    head_f32, head = extract(b"#f32_2\0\0", head)
    head_f32_2, head = extract(b"#f32_3\0\0", head)
    head_f32_3, head = extract(b"#f32_4\0\0", head)
    head_f32_4, head = extract(b"#f32_8\0\0", head)
    head_f32_8, head = extract(b"#f32_16\0", head)
    head_f32_16, head = extract(b"#f64\0\0\0\0", head)
    head_f64, head = extract(b"#f64_2\0\0", head)
    head_f64_2, head = extract(b"#f64_3\0\0", head)
    head_f64_3, head = extract(b"#f64_4\0\0", head)
    head_f64_4, head = extract(b"#f64_8\0\0", head)
    head_f64_8, head = extract(b"#f64_16\0", head)
    head_f64_16, head = extract(b"#u32\0\0\0\0", head)
    head_u32, head_u64 = extract(b"#u64\0\0\0\0", head)

    # print(head_f32,head_f32_3)

    dic_fields = {
        "f32": [],
        "f32_2": [],
        "f32_3": [],
        "f32_4": [],
        "f32_8": [],
        "f32_16": [],
        "f64": [],
        "f64_2": [],
        "f64_3": [],
        "f64_4": [],
        "f64_8": [],
        "f64_16": [],
        "u32": [],
        "u64": [],
    }

    read_header_field(head_f32, dic_fields["f32"])
    read_header_field(head_f32_2, dic_fields["f32_2"])
    read_header_field(head_f32_3, dic_fields["f32_3"])
    read_header_field(head_f32_4, dic_fields["f32_4"])
    read_header_field(head_f32_8, dic_fields["f32_8"])
    read_header_field(head_f32_16, dic_fields["f32_16"])

    read_header_field(head_f64, dic_fields["f64"])
    read_header_field(head_f64_2, dic_fields["f64_2"])
    read_header_field(head_f64_3, dic_fields["f64_3"])
    read_header_field(head_f64_4, dic_fields["f64_4"])
    read_header_field(head_f64_8, dic_fields["f64_8"])
    read_header_field(head_f64_16, dic_fields["f64_16"])

    read_header_field(head_u32, dic_fields["u32"])
    read_header_field(head_u64, dic_fields["u64"])

    print(dic_fields)

    for field in dic_fields["f32"]:
        elements = field["obj_cnt"] * field["nvar"]
        off_data = elements * 4
        data_field = data[:off_data]

        # print(data_field)
        data = data[off_data:]

        field["field"] = []

        for i in range(elements):
            field["field"].append(struct.unpack("f", data_field[i * 4 : (i + 1) * 4]))

    for field in dic_fields["f32_3"]:
        elements = field["obj_cnt"] * field["nvar"]
        off_data = elements * 4 * 3
        data_field = data[:off_data]

        # print(data_field)
        data = data[off_data:]

        field["field"] = []

        for i in range(elements):
            field["field"].append(struct.unpack("fff", data_field[i * 12 : (i + 1) * 12]))

    # print(dic_fields)

    dic = {
        "x": [],
        "y": [],
        "z": [],
        "vx": [],
        "vy": [],
        "vz": [],
        "ax": [],
        "ay": [],
        "az": [],
        "ax_old": [],
        "ay_old": [],
        "az_old": [],
        "h": [],
    }

    for field in dic_fields["f32_3"]:
        if field["name"] == ("xyz"):
            for a in field["field"]:
                x, y, z = a
                dic["x"].append(x)
                dic["y"].append(y)
                dic["z"].append(z)

        if field["name"] == ("vxyz"):
            for a in field["field"]:
                x, y, z = a
                dic["vx"].append(x)
                dic["vy"].append(y)
                dic["vz"].append(z)

        if field["name"] == ("axyz"):
            for a in field["field"]:
                x, y, z = a
                dic["ax"].append(x)
                dic["ay"].append(y)
                dic["az"].append(z)

        if field["name"] == ("axyz_old"):
            for a in field["field"]:
                x, y, z = a
                dic["ax_old"].append(x)
                dic["ay_old"].append(y)
                dic["az_old"].append(z)

    for field in dic_fields["f32"]:
        if field["name"] == ("hpart"):
            for a in field["field"]:
                (x,) = a
                dic["h"].append(x)

    return dic


def plot_soundwave(idx: int, toff: float):
    file_list = glob.glob("./step" + str(idx) + "/patchdata*")

    f = open("./step" + str(idx) + "/timeval.bin", "rb")
    (tval,) = struct.unpack("d", f.read(8))
    f.close()

    tval -= toff

    dic = {}

    for fname in file_list:
        print("converting : {} t = {}".format(fname, tval))
        dic_tmp = get_plot_patchdata(fname)

        for k in dic_tmp.keys():
            if not k in dic.keys():
                dic[k] = []

            dic[k] += dic_tmp[k]

    fig, (ax1, ax2, ax3) = plt.subplots(nrows=3, sharex=True)  # frameon=False removes frames

    h_arr = np.array(dic["h"])

    # m = 1e-5
    m = 2e-4
    hfac = 1.2
    rho = m * (hfac / h_arr) * (hfac / h_arr) * (hfac / h_arr)

    # dv0 = 1e-2
    dv0 = 1e-2
    nmode = 2
    z_min = -1
    # z_max = 1.0249114036560059
    # z_max = 0.992251
    z_max = 0.9595917
    cs = 1  # /(1.694)*1.0362095232493131
    z = np.array(dic["z"])

    zs = z_max - z_min

    omega = 2 * np.pi * (2 / zs) * cs
    print("omega :", omega)
    print("wt/pi :", omega * tval / np.pi)
    anal_vz = (
        dv0 * np.cos(nmode * 2.0 * np.pi * (z - z_min) / (z_max - z_min)) * np.cos(omega * tval)
    )

    # rho_0 = 0.22097201794744842
    # rho_0 = 0.02684098807585487
    rho_0 = 0.55083
    # rho_0 = 1.7392566998603394
    # rho_0 = 0.2203
    drho_0 = rho_0 * (dv0 / cs)

    anal_rho = rho_0 + drho_0 * np.sin(
        nmode * 2.0 * np.pi * (z - z_min) / (z_max - z_min)
    ) * np.sin(omega * tval)

    anal_az = (
        -dv0
        * omega
        * np.cos(nmode * 2.0 * np.pi * (z - z_min) / (z_max - z_min))
        * np.sin(omega * tval)
    )

    ax1.scatter(z, dic["vz"], label="SPH")
    ax1.scatter(z, anal_vz, label="analytique")
    ax1.set_xlabel("$z$")
    ax1.set_ylabel("$v_z$")
    ax1.set_title("$t={}$".format(tval))

    ax2.scatter(z, rho, label="SPH")
    ax2.scatter(z, anal_rho, label="analytique")
    ax2.set_xlabel("$z$")
    ax2.set_ylabel("$\\rho$")

    ax3.scatter(z, dic["az"], label="SPH")
    ax3.scatter(z, anal_az, label="analytique")
    ax3.set_xlabel("$z$")
    ax3.set_ylabel("$a_z$")


f = open("./step" + str(4) + "/timeval.bin", "rb")
(toff,) = struct.unpack("d", f.read(8))
f.close()

plot_soundwave(5, toff)

plot_soundwave(25, toff)

plot_soundwave(50, toff)

plot_soundwave(75, toff)

plot_soundwave(100, toff)

plot_soundwave(150, toff)

# plot_soundwave(200,toff)


# plot_soundwave(250,toff)

# plot_soundwave(300,toff)
# plot_soundwave(400,toff)
# plot_soundwave(499,toff)

plt.show()
