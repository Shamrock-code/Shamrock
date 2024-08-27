from lib.buildbot import *
import json
import sys

print_buildbot_info("mrege profiling files")


outfile = sys.argv[-1]
flist = sys.argv[1:-1]

str_lst = []

for f in flist:

    print(f)
    str_f = ""

    fl = open(f,'r')
    str_f = fl.read()
    fl.close()

    if(str_f.endswith(",")):
        str_f = str_f[:-1] + "]"

    if(str_f.endswith("}")):
        str_f = str_f + "]"

    str_lst.append(str_f)



sumator = []

for st in str_lst:
    sumator += json.loads(st)

with open(outfile,'w') as outf :
    json.dump(sumator, outf)
