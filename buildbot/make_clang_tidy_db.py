import os

comp_db = open("build/compile_commands.json", "r")
db = comp_db.read()

db = db.replace("--acpp-targets='omp'","")
#print(db)


def remove_external_files():
    global db

    import json
    dic = json.loads(db)

    ret_dic = []

    for a in dic:
        if not ("Shamrock/external" in a["file"]):
            ret_dic.append(a)

    db = json.dumps(ret_dic)

remove_external_files()

try:
    os.mkdir("build/clang-tidy.mod")
except:
    pass

comp_db = open("build/clang-tidy.mod/compile_commands.json", "w")
comp_db.write(db)
