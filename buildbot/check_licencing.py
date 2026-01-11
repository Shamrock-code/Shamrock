import glob
import sys

from lib.buildbot import *

print_buildbot_info("licence check tool")


####################################################################################################
# Licenses
####################################################################################################

licence_cpp = R"""// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright (c) 2021-2025 Timothée David--Cléris <tim.shamrock@proton.me>
// SPDX-License-Identifier: CeCILL Free Software License Agreement v2.1
// Shamrock is licensed under the CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//"""


def selector_cpp(fname):
    import re

    pattern = re.compile(r".*\.(cpp|hpp)$")
    return bool(pattern.match(fname))


licence_cmake = R"""## -------------------------------------------------------
##
## SHAMROCK code for hydrodynamics
## Copyright (c) 2021-2025 Timothée David--Cléris <tim.shamrock@proton.me>
## SPDX-License-Identifier: CeCILL Free Software License Agreement v2.1
## Shamrock is licensed under the CeCILL 2.1 License, see LICENSE for more information
##
## -------------------------------------------------------"""


def selector_cmake(fname):
    t1 = fname.endswith("/CMakeLists.txt")
    import re

    pattern = re.compile(r".*\.cmake$")
    return bool(pattern.match(fname)) or t1


checkers = [(selector_cpp, licence_cpp), (selector_cmake, licence_cmake)]


####################################################################################################
# File listing
####################################################################################################


file_list = glob.glob(str(abs_proj_dir) + "/cmake/**", recursive=True)
file_list += glob.glob(str(abs_src_dir) + "/**", recursive=True)

file_list.sort()


####################################################################################################
# Check part
####################################################################################################


missing_licence = []

for fname in file_list:
    print("checking", fname)
    for selector, licence in checkers:
        if selector(fname):
            print(" -", fname)

            f = open(fname, "r")
            res = f.read().startswith(licence)
            f.close()

            if not res:
                missing_licence.append(fname)


def write_file(fname, source):
    f = open(fname, "w")
    f.write(source)
    f.close()


def make_check_pr_report():
    rep = ""
    # start allow utf-8
    rep += "## ❌ Check license headers"
    # end allow utf-8
    rep += """

The pre-commit checks have found some missing or ill formed license header.
All C++ files (headers or sources) should start with :
```
// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright (c) 2021-2025 Timothée David--Cléris <tim.shamrock@proton.me>
// SPDX-License-Identifier: CeCILL Free Software License Agreement v2.1
// Shamrock is licensed under the CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//
```
Any line break before this header or change in its formatting will trigger the fail of this test
        """

    rep += "List of files with errors :\n\n"
    for i in missing_licence:
        rep += " - `" + i.split(abs_proj_dir)[-1] + "`\n"

    write_file("log_precommit_license_check", rep)


if len(missing_licence) > 0:
    make_check_pr_report()
    print(" => \033[1;34mlicence missing in \033[0;0m: ")

    for i in missing_licence:
        print(" -", i.split(abs_proj_dir)[-1])

    sys.exit("Missing liscence for some source files")
else:
    print(" => \033[1;34mLicense status \033[0;0m: OK !")
