import argparse

from lib.buildbot import *

print_buildbot_info("compile stats")


parser = argparse.ArgumentParser(description="Configure utility for Shamrock")
parser.add_argument("path")  # positional argument
args = parser.parse_args()


arr = []

f = open(args.path + "/.ninja_log", "r")
for l in f.readlines():
    if len(l.split()) == 5:
        arr.append(l.split())
f.close()


end_time = 0

names = []
comp_time = []


for [tstart, tend, hsh, name, hsh2] in arr:
    tstart = float(tstart)
    tend = float(tend)

    names.append(name)
    comp_time.append((tend - tstart) / 1000)

    end_time = max(end_time, tend)

sum_time = sum(comp_time)

prc = [100 * (c / sum_time) for c in comp_time]

arr2 = []

for n, t, p in zip(names, comp_time, prc):
    arr2.append((p, n, t))

arr2.sort()

arr2 = arr2[::-1]

for p, n, t in arr2:
    print("{:>7.2f}% {:>7.2f}s  {:}".format(p, t, n))
