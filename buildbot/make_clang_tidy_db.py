import os 

comp_db = open("build/compile_commands.json", "r")
db = comp_db.read()

db = db.replace("--acpp-targets='omp'","")
print(db)

try:
    os.mkdir("build/clang-tidy.mod")
except:
    pass

comp_db = open("build/clang-tidy.mod/compile_commands.json", "w")
comp_db.write(db)