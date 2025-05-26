# Take two doxygen warning output and compare them to generate a diff in markdown (printed to out)

import os
import sys

# Get current file path
cur_file = os.path.realpath(os.path.expanduser(__file__))

# Get project directory
abs_proj_dir = os.path.abspath(os.path.join(cur_file, "../.."))


def make_sorted_file(filename, sorted_file):
    os.system(f"cat {filename} | sort > {sorted_file}")


def make_diff(file1, file2):
    os.system(f"diff {file1} {file2} > out_diff")


def line_count_file(file):
    return sum(1 for _ in open(file))


def shorten_file_path(l):
    l = l.replace(abs_proj_dir + "/", "")
    l = l.replace("/__w/Shamrock/Shamrock/", "")
    l = l.replace("/home/docker/actions-runner/_work/Shamrock/Shamrock/", "")
    return l


def load_diff(file):
    f = open(file, "r")
    lines = f.readlines()
    f.close()

    line_add = []
    line_del = []

    for l in lines:
        l = shorten_file_path(l)
        if l.startswith(">"):
            line_add.append(l[2:-1])
        if l.startswith("<"):
            line_del.append(l[2:-1])

    return line_add, line_del


file1 = sys.argv[1]
file2 = sys.argv[2]

make_sorted_file(file1, file1 + ".sort")
make_sorted_file(file2, file2 + ".sort")
make_diff(file1 + ".sort", file2 + ".sort")

line_add, line_del = load_diff("out_diff")

before_count = line_count_file(file1)
after_count = line_count_file(file2)


def format_delta_pourcent():
    delt = after_count - before_count
    div = delt / before_count

    ret = ""
    if before_count == 0:
        if after_count == 0:
            return ""
        else:
            return "(+∞)"
    else:
        return "({:.1%})".format(div)


print("# Doxygen diff with `main`")
print(f"Removed warnings : {len(line_del)}")
print(f"New warnings : {len(line_add)}")
print(f"Warnings count : {before_count} → {after_count} {format_delta_pourcent()}")
print("<details>")
print("<summary>")
print("Detailed changes :")
print("</summary>")
print(" ")
print("```diff")

lst = []

for l in line_add:
    lst.append([l, "+"])
for l in line_del:
    lst.append([l, "-"])

lst.sort()
for [a, b] in lst:
    if a.startswith("src"):
        if b == "+":
            print(f"+ {a}")
        if b == "-":
            print(f"- {a}")

print("```")
print("")
print("</details>")
