from lib.buildbot import *
import glob
import sys

print_buildbot_info("licence check tool")

licence = R'''// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright(C) 2021-2023 Timothée David--Cléris <timothee.david--cleris@ens-lyon.fr>
// Licensed under CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//'''

file_list = glob.glob(str(abs_src_dir)+"/**",recursive=True)

file_list.sort()

missing_licence = []

for fname in file_list:

    if (not fname.endswith(".cpp")) and (not fname.endswith(".hpp")):
        continue

    if fname.endswith("version.cpp"):
        continue

    f = open(fname,'r')
    res = f.read().startswith(licence)
    f.close()

    if not res :
        missing_licence.append(fname)



def write_file(fname, source):
    f = open(fname,'w')
    f.write(source)
    f.close()

def make_check_pr_report():
    rep = ""
    rep += ("## ❌ Check license headers")
    rep += ("""

The pre-commit checks have found some missing or ill formed license header.
All C++ files (headers or sources) should start with :
```
// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright(C) 2021-2023 Timothée David--Cléris <timothee.david--cleris@ens-lyon.fr>
// Licensed under CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//
```
Any line break before this header or change in its formatting will trigger the fail of this test
        """)

    rep += "List of files with errors :\n\n"
    for i in missing_licence:
        rep += (" - `"+i.split(abs_proj_dir)[-1]+"`\n")

    write_file("log_precommit_license_check", rep)


if len(missing_licence) > 0:
    make_check_pr_report()
    print(" => \033[1;34mlicence missing in \033[0;0m: ")

    for i in missing_licence:
        print(" -",i.split(abs_proj_dir)[-1])

    sys.exit("Missing liscence for some source files")
else :
    print(" => \033[1;34mLicense status \033[0;0m: OK !")
