import os
import sys

print("Current working directory:", os.getcwd())
comp_db = open("compile_commands.json", "r")
db = comp_db.read()

print("Database opened, replacing non standard flags ...")

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


print("Removing external files ...")
remove_external_files()

print("Creating clang-tidy.mod directory ...")
try:
    os.mkdir("clang-tidy.mod")
except:
    pass

print("Writing compile_commands.json to clang-tidy.mod directory ...")
comp_db = open("clang-tidy.mod/compile_commands.json", "w")
comp_db.write(db)

print("Done !")
