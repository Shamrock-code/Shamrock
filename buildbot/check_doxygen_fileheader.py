from lib.buildbot import * 
import glob
import sys

print_buildbot_info("licence check tool")

file_list = glob.glob(str(abs_src_dir)+"/**",recursive=True)

file_list.sort()

missing_doxygenfilehead = []

def has_header(filedata, filename):

    has_file_tag = ("@file "+filename) in filedata
    has_author_tag =  ("@author ") in filedata

    return has_file_tag and has_author_tag

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
    res = has_header(f.read(), os.path.basename(fname))
    f.close()

    if not res : 
        missing_doxygenfilehead.append(fname)


if len(missing_doxygenfilehead) > 0:
    print(" => \033[1;34mDoxygen header missing in \033[0;0m: ")

    for i in missing_doxygenfilehead:
        print(" -",i.split(abs_proj_dir)[-1])

    print(r"""
    
    Please add a doxygen header in the file above, similar to this : 

    /**
     * @file {filename}
     * @author {name} (mail@mail.com)
     * @brief ...
     */
    
    """)

    sys.exit("Missing doxygen header for some source files")
else : 
    print(" => \033[1;34mLicense status \033[0;0m: OK !")