# Take two doxygen warning output and compare them to generate a diff in markdown (printed to out)

import os
import sys

def make_sorted_file(filename, sorted_file):
    os.system(f"cat {filename} | sort > {sorted_file}")

def make_diff(file1, file2):
    os.system(f"diff {file1} {file2} > out_diff")

def load_diff(file):
    f = open(file, 'r')
    lines = f.readlines()
    f.close()

    line_add = []
    line_del = []

    for l in lines:
        if l.startswith(">"):
            line_add.append(l[2:-1])
        if l.startswith("<"):
            line_del.append(l[2:-1])

    return line_add, line_del

file1 = sys.argv[1]
file2 = sys.argv[2]

make_sorted_file(file1, file1+".sort")
make_sorted_file(file2, file2+".sort")
make_diff(file1+".sort", file2+".sort")

line_add, line_del = load_diff("out_diff")

print(f"Removed warnings : {len(line_del)}")
print(f"New warnings : {len(line_add)}")
print(f"Changes : ")
print("```diff")
for l in line_add:
    print(f"+ {l}")
for l in line_del:
    print(f"- {l}")
print("```")