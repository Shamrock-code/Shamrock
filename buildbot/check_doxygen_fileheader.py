import glob
import sys

from lib.buildbot import *

print_buildbot_info("licence check tool")

file_list = glob.glob(str(abs_src_dir) + "/**", recursive=True)

file_list.sort()

missing_doxygenfilehead = []


def has_header(filedata, filename):
    has_file_tag = ("@file " + filename) in filedata
    has_author_tag = ("@author ") in filedata

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

    f = open(fname, "r")
    res = has_header(f.read(), os.path.basename(fname))
    f.close()

    if not res:
        missing_doxygenfilehead.append(fname)


import re


def autocorect(source, filename):
    do_replace = not (("@file " + filename) in source) and (" * @file " in source)

    source = re.sub(r" \* @file (.+)\n", r" * @file " + filename + "\n", source)

    return do_replace, source


def run_autocorect():
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

        f = open(fname, "r")
        source = f.read()
        f.close()

        res = has_header(source, os.path.basename(fname))

        if not res:
            change, source = autocorect(source, os.path.basename(fname))

            if change:
                print("autocorect : ", fname.split(abs_proj_dir)[-1])
                f = open(fname, "w")
                f.write(source)
                f.close()


def write_file(fname, source):
    f = open(fname, "w")
    f.write(source)
    f.close()


def make_check_pr_report():
    rep = ""
    # start allow utf-8
    rep += "## ❌ Check doxygen headers"
    # end allow utf-8
    rep += """

The pre-commit checks have found some files without or with a wrong doxygen file header

A properly formed doxygen header should be like and placed just below `#pragma once` in headers or below the license in source files:
```
/**
 * @file <filename>
 * @author <Author name> (<email>)
 * @brief <Description of the file content>
 *
 */
```

This test is triggered if the doxygen header is missing or with the wrong file name

At some point we will refer to a guide in the doc about this
\n"""

    rep += "List of files with errors :\n\n"

    for i in missing_doxygenfilehead:
        rep += " - `" + i.split(abs_proj_dir)[-1] + "`\n"

    write_file("log_precommit_doxygen_header", rep)


if len(missing_doxygenfilehead) > 0:
    print(" => \033[1;34mDoxygen header missing in \033[0;0m: ")

    for i in missing_doxygenfilehead:
        print(" -", i.split(abs_proj_dir)[-1])

    print(
        r"""

    Please add a doxygen header in the file above, similar to this :

    /**
     * @file {filename}
     * @author {name} (mail@mail.com)
     * @brief ...
     */

    """
    )

    run_autocorect()
    make_check_pr_report()

    sys.exit("Missing doxygen header for some source files")
else:
    print(" => \033[1;34mLicense status \033[0;0m: OK !")
