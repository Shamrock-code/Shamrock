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


def remove_pch_flags(cmd):
    # clang-tidy runs against source files directly, without ever building the actual
    # .pch object (the tidy CI job only configures + generates version.cpp, it never
    # runs a full ninja build), so "-Xclang -include-pch -Xclang <file>.pch" points at a
    # file that doesn't exist: "PCH file ... not found: module file not found". Drop
    # that pair (and the now-pointless -Winvalid-pch) and keep the plain
    # "-Xclang -include -Xclang <file>.hxx", which still exists and just gets parsed
    # normally like any other header.
    tokens = cmd.split()
    new_tokens = []
    i = 0
    while i < len(tokens):
        tok = tokens[i]
        if tok == "-Winvalid-pch":
            i += 1
            continue
        if (
            tok == "-Xclang"
            and i + 3 < len(tokens)
            and tokens[i + 1] == "-include-pch"
            and tokens[i + 2] == "-Xclang"
        ):
            i += 4
            continue
        new_tokens.append(tok)
        i += 1

    return " ".join(new_tokens) + " "


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
            a["command"] = remove_pch_flags(remove_plugin_flags(new_cmd))
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
