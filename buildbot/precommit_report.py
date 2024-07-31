import glob
import sys

file_list = glob.glob("log_precommit_*",recursive=True)

def load_file(fname):
    f = open(fname,'r')
    source = f.read()
    f.close()
    return source 

dic = []

for f in file_list:
    #print(f)
    log_f = load_file(f)
    #print(log_f)

    if f == "log_precommit_check_sycl_include":
        dic.append({
            "checkname" : ""
        })


print("## Resulting diff")
print("<details>")
print("<summary>")
print("Detailed changes :")
print("</summary>")
print(f" ")
print("```diff")
print(load_file("diff-pre-commit"))
print("```")
print("")
print("</details>")