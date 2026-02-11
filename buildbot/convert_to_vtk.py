import struct
import sys

import vtk
import vtkmodules


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
        "u": [],
        "du": [],
        "du_old": [],
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

        if field["name"] == ("u"):
            for a in field["field"]:
                (x,) = a
                dic["u"].append(x)

        if field["name"] == ("du"):
            for a in field["field"]:
                (x,) = a
                dic["du"].append(x)

        if field["name"] == ("du_old"):
            for a in field["field"]:
                (x,) = a
                dic["du_old"].append(x)

    return dic


def write_dic_to_vtk(dic, filename):
    points = vtk.vtkPoints()
    points.SetNumberOfPoints(len(dic["x"]))
    for i in range(len(dic["x"])):
        points.SetPoint(i, (dic["x"][i], dic["y"][i], dic["z"][i]))

    ugrid = vtkmodules.vtkCommonDataModel.vtkUnstructuredGrid()

    ugrid.SetPoints(points)

    for k in ["h", "u", "du", "du_old"]:
        if not k in ["x", "y", "z"]:
            values = vtk.vtkDoubleArray()
            values.SetName(k)
            values.SetNumberOfValues(len(dic["x"]))
            for i in range(len(dic["x"])):
                values.SetValue(i, dic[k][i])

            ugrid.GetPointData().AddArray(values)

    for desc in [["v", "vx", "vy", "vz"]]:
        vect = vtk.vtkDoubleArray()
        vect.SetName(desc[0])
        vect.SetNumberOfComponents(3)
        vect.SetNumberOfValues(len(dic["x"]) * 3)

        for i in range(len(dic["x"])):
            vect.SetTuple(i, (dic[desc[1]][i], dic[desc[2]][i], dic[desc[3]][i]))

            ugrid.GetPointData().AddArray(vect)

    for desc in [["a", "ax", "ay", "az"]]:
        vect = vtk.vtkDoubleArray()
        vect.SetName(desc[0])
        vect.SetNumberOfComponents(3)
        vect.SetNumberOfValues(len(dic["x"]) * 3)

        for i in range(len(dic["x"])):
            vect.SetTuple(i, (dic[desc[1]][i], dic[desc[2]][i], dic[desc[3]][i]))

            ugrid.GetPointData().AddArray(vect)

    for desc in [["a_old", "ax_old", "ay_old", "az_old"]]:
        vect = vtk.vtkDoubleArray()
        vect.SetName(desc[0])
        vect.SetNumberOfComponents(3)
        vect.SetNumberOfValues(len(dic["x"]) * 3)

        for i in range(len(dic["x"])):
            vect.SetTuple(i, (dic[desc[1]][i], dic[desc[2]][i], dic[desc[3]][i]))

            ugrid.GetPointData().AddArray(vect)

    fn = filename + ".vtu"
    writer = vtk.vtkUnstructuredGridWriter()
    writer.SetFileName(fn)
    writer.SetFileTypeToBinary()
    writer.SetInputData(ugrid)
    writer.Write()


import glob

for idx in range(170, 1000):
    file_list = glob.glob("./step" + str(idx) + "/patchdata*")

    print(file_list)

    if len(file_list) == 0:
        break

    f = open("./step" + str(idx) + "/timeval.bin", "rb")
    (tval,) = struct.unpack("d", f.read(8))
    f.close()

    dic = {}

    for fname in file_list:
        print("converting : {} t = {}".format(fname, tval))
        dic_tmp = get_plot_patchdata(fname)

        for k in dic_tmp.keys():
            if not k in dic.keys():
                dic[k] = []

            dic[k] += dic_tmp[k]

    write_dic_to_vtk(dic, "step" + str(idx))

    # lst.append({"timestep" : tval, "file" : "step"+str(idx)+'.vtu'})
exit()

import glob

for idx in range(1, 1000):
    file_list = glob.glob("./step" + str(idx) + "/merged0_patchdata*")

    print(file_list)

    if len(file_list) == 0:
        break

    f = open("./step" + str(idx) + "/timeval.bin", "rb")
    (tval,) = struct.unpack("d", f.read(8))
    f.close()

    dic = {}

    for fname in file_list:
        print("converting : {} t = {}".format(fname, tval))
        dic_tmp = get_plot_patchdata(fname)

        for k in dic_tmp.keys():
            if not k in dic.keys():
                dic[k] = []

            dic[k] += dic_tmp[k]

    write_dic_to_vtk(dic, "merged_step" + str(idx))


import glob

for idx in range(1, 1000):
    file_list = glob.glob("./step_before_reatrib" + str(idx) + "/patchdata*")

    print(file_list)

    if len(file_list) == 0:
        break

    f = open("./step_before_reatrib" + str(idx) + "/timeval.bin", "rb")
    (tval,) = struct.unpack("d", f.read(8))
    f.close()

    dic = {}

    for fname in file_list:
        print("converting : {} t = {}".format(fname, tval))
        dic_tmp = get_plot_patchdata(fname)

        for k in dic_tmp.keys():
            if not k in dic.keys():
                dic[k] = []

            dic[k] += dic_tmp[k]

    write_dic_to_vtk(dic, "step_before_reatrib" + str(idx))

import glob

for idx in range(1, 1000):
    file_list = glob.glob("./step_after_reatrib" + str(idx) + "/patchdata*")

    print(file_list)

    if len(file_list) == 0:
        break

    f = open("./step_after_reatrib" + str(idx) + "/timeval.bin", "rb")
    (tval,) = struct.unpack("d", f.read(8))
    f.close()

    dic = {}

    for fname in file_list:
        print("converting : {} t = {}".format(fname, tval))
        dic_tmp = get_plot_patchdata(fname)

        for k in dic_tmp.keys():
            if not k in dic.keys():
                dic[k] = []

            dic[k] += dic_tmp[k]

    write_dic_to_vtk(dic, "step_after_reatrib" + str(idx))
