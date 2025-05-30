import glob

from lib.buildbot import *

print_buildbot_info("make status file table")

import json
import os


def get_new_state(state1: str, state2: str):
    dic = {
        "?": -1000,
        "Deprecated": -1,
        "Should rewrite": 0,
        "Need cleaning": 1,
        "Clean unfinished": 2,
        "Good": 3,
        "Clean": 4,
    }

    dic_inv = {}
    for k in dic.keys():
        dic_inv[dic[k]] = k

    st1 = dic[state1]
    st2 = dic[state2]

    st = [st1, st2]
    st.sort()

    if -1000 in st:
        return dic_inv[-1000]

    elif st == [-1, -1]:
        return dic_inv[-1]

    elif -1 in st:
        return dic_inv[0]

    else:
        return dic_inv[min(st[0], st[1])]


def path_to_dict(path):
    d = {"name": os.path.basename(path)}
    if os.path.isdir(path):
        d["type"] = "directory"
        d["children"] = []

        for x in os.listdir(path):
            if not x.endswith(".txt"):
                d["children"].append(path_to_dict(os.path.join(path, x)))

        status = ""

        for c in d["children"]:
            if "implstatus" in c.keys():
                if status == "":
                    status = c["implstatus"]
                else:
                    status = get_new_state(c["implstatus"], status)

            else:
                status = "?"

        if not (status == ""):
            d["implstatus"] = status

    else:
        d["type"] = "file"

        f = open(path, "r")

        flag = "//%Impl status : "

        for l in f.readlines():
            if l.startswith(flag):
                d["implstatus"] = l[len(flag) :][:-1]

        f.close()

    return d


j_out = path_to_dict(abs_src_dir)

# print(j_out)


def print_node(prefix, node):
    color = ""
    spacing = ""

    impl_str = ""

    if "implstatus" in node.keys():
        if node["implstatus"] == "Good":
            color = "\x1b[32m"
            spacing = "     "
        if node["implstatus"] == "Clean":
            color = "\x1b[34m"
            spacing = "     "
        if node["implstatus"] == "Clean unfinished":
            color = "\x1b[36m"
            spacing = "     "
        if node["implstatus"] == "Need cleaning":
            color = "\x1b[35m"
            spacing = "     "
        if node["implstatus"] == "Should rewrite":
            color = "\x1b[33m"
            spacing = "     "
        if node["implstatus"] == "Deprecated":
            color = "\x1b[31m"
            spacing = "     "

        impl_str = "(" + color + node["implstatus"] + "\x1b[0m" + ")"

    str = prefix + " " + color + node["name"] + "\x1b[0m"

    print(str.ljust(60) + spacing + impl_str)

    if "children" in node.keys():
        for c in node["children"]:
            print_node(prefix + " | ", c)


print_node("", j_out)
