import os
import sys

comp_db = open("build/compile_commands.json", "r")
db = comp_db.read()

db = db.replace("--acpp-targets='omp'", "")
# print(db)


def remove_plugin_flags(cmd):
    new_cmd = ""

    for a in cmd.split():
        if not (a.startswith("-fplugin=") or a.startswith("-fpass-plugin")):
            new_cmd += a
            new_cmd += " "

    # print(new_cmd)

    return new_cmd


def remove_external_files():
    global db

    import json

    dic = json.loads(db)

    ret_dic = []

    for a in dic:
        if not ("Shamrock/external" in a["file"]):
            cmd = a["command"] + " --acpp-dryrun"
            # print("--->",cmd)
            new_cmd = os.popen(cmd).readlines()[0][:-1]
            a["command"] = remove_plugin_flags(new_cmd)
            ret_dic.append(a)

    db = json.dumps(ret_dic, indent=4)


remove_external_files()

try:
    os.mkdir("build/clang-tidy.mod")
except:
    pass

comp_db = open("build/clang-tidy.mod/compile_commands.json", "w")
comp_db.write(db)
