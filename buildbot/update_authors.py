from lib.buildbot import * 
import glob
import sys
import subprocess

print_buildbot_info("Authors check tool")

file_list = glob.glob(str(abs_src_dir)+"/**",recursive=True)

file_list.sort()

missing_doxygenfilehead = []

def get_doxstring(path, filename):
    tmp = " * @file "+ filename+"\n"
    try:
        tmp+= (subprocess.check_output(R'git log --pretty=format:" * @author %aN (%aE)" '+path+' |sort |uniq',shell=True).decode())[:-1]
    except subprocess.CalledProcessError as err:
        print(err)

    return tmp

import re
import difflib

def print_diff(before, after, beforename, aftername):
    sys.stdout.writelines(difflib.context_diff(before.split("\n"), after.split("\n"), fromfile=beforename, tofile=aftername))


def autocorect(source, filename, path):

    l_start = 0
    l_end = 0
    i = 0

    splt = source.split("\n")
    for l in splt:
        if(l_start > 0):
            if not("@author" in l):
                break
        if("@file" in l):
            l_start = i
        if("@author" in l):
            l_end = i
        i += 1

    if l_end == 0:
        l_end = l_start

    new_splt = splt[:l_start] 
    new_splt.append(get_doxstring(path, filename))
    new_splt += splt[l_end+1:]

    new_src = ""
    for l in new_splt:
        new_src += l + "\n"
    new_src = new_src[:-1]


    do_replace = not (new_src == source)

    if do_replace:
        print_diff(source, new_src, filename, filename + " (corec)")

    return do_replace,new_src

def run_autocorect():

    errors = []

    for fname in file_list:

        if (not fname.endswith(".cpp")) and (not fname.endswith(".hpp")):
            continue

        if fname.endswith("version.cpp"):
            continue

        if "/src/tests/" in fname:
            continue
        if "exemple.cpp" in fname:
            continue
        if "godbolt.cpp" in fname:
            continue

        f = open(fname,'r')
        source = f.read()
        f.close()


        change, source = autocorect(source,os.path.basename(fname),fname)

        if change: 
            print("autocorect : ",fname.split(abs_proj_dir)[-1])
            f = open(fname,'w')
            f.write(source)
            f.close()
            errors.append(fname.split(abs_proj_dir)[-1])

    return errors



missing_doxygenfilehead = run_autocorect()
