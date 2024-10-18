from lib.buildbot import *
import json
import sys

print_buildbot_info("mrege profiling files")


flist = sys.argv[1:]


with open("merged_profile.json",'w') as outf :
    outf.write("[\n")
    for f in flist:

        print(f)
        str_f = "[\n"

        fl = open(f,'r')
        str_f = fl.read()
        fl.close()

        outf.write(str_f)

    outf.write("]\n")
