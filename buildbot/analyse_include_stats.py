import argparse
import hashlib
import json
import multiprocessing
import os
import re

parser = argparse.ArgumentParser(description="Configure utility for Shamrock")
parser.add_argument("filename")  # positional argument


args = parser.parse_args()


def modify_command(cmd, output):
    return re.sub(r"(-o [^ ]+) ", "-E -o {} ".format(output), cmd)


def get_file_line_count(fname):
    count = 0
    with open(fname) as f:
        for line in f:
            count += 1
    return count


print_db = []


def analyse_file(f):
    fname = f["file"]
    fname_relat = fname[1 + fname.index("/src/") :]

    md5 = hashlib.md5()
    md5.update(fname.encode())
    hash_ = "{0}".format(md5.hexdigest())

    fout = "test" + hash_

    # print("processing :",fname_relat)

    line_src = get_file_line_count(fname)

    mod_cmd = modify_command(f["command"], fout)

    # print("run : ",mod_cmd)
    os.system(mod_cmd)

    line_src_preproc = get_file_line_count(fout)

    os.system("rm " + fout)

    print(fname_relat, ": src : {} -E {}".format(line_src, line_src_preproc))

    return [fname_relat, line_src, line_src_preproc]


with open(args.filename) as cmd_comp:
    database = json.load(cmd_comp)

    pool_obj = multiprocessing.Pool()
    print_db = pool_obj.map(analyse_file, database)


sum_l = 0
sum_E = 0

print_db.sort()

print(
    "                      filename                                           line count     -E     "
)
print(
    "----------------------------------------------------------------------    ---------   ---------"
)
for [name, src, preproc] in print_db:
    tmp = "{:70s} {:10} {:10}".format(name, src, preproc)
    print(tmp)

    sum_l += src
    sum_E += preproc

print(
    "----------------------------------------------------------------------    ---------   ---------"
)
tmp = "{:70s} {:10} {:10}".format("SUM", sum_l, sum_E)
print(tmp)
