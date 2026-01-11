import glob
import sys

from lib.buildbot import *

print_buildbot_info("include guard check tool")

check_line = R"""#pragma once"""

file_list = glob.glob(str(abs_src_dir) + "/**", recursive=True)

file_list.sort()

pragma_once_missing = []

for fname in file_list:
    if not fname.endswith(".hpp"):
        continue

    if fname.endswith("version.cpp"):
        continue

    f = open(fname, "r")
    res = f.readlines()

    has_pragma = False
    has_line_before_guard = False
    for l in res:
        is_pragma = l.startswith(check_line)
        if is_pragma:
            has_pragma = True
            break
        else:
            if not (
                l.startswith(r"//")
                or l.startswith(r"/*")
                or l.startswith(r"/*")
                or l.startswith("\n")
            ):
                # print(l)
                has_line_before_guard = True

    f.close()

    if has_line_before_guard:
        pragma_once_missing.append(fname)


def write_file(fname, source):
    f = open(fname, "w")
    f.write(source)
    f.close()


def make_check_pr_report():
    rep = ""
    # start allow utf-8
    rep += "## ❌ Check #pragma once"
    # end allow utf-8
    rep += """

The pre-commit checks have found some headers that are not starting with `#pragma once`.
This indicates to the compiler that this header should only be included once per source files avoid double definitions of function or variables

All headers files should have, just below the license header the following line :
```
#pragma once
```

At some point we will refer to a guide in the doc about this
"""

    rep += "List of files with errors :\n\n"

    for i in pragma_once_missing:
        rep += " - `" + i.split(abs_proj_dir)[-1] + "`\n"

    write_file("log_precommit_pragma_once_check", rep)


if len(pragma_once_missing) > 0:
    make_check_pr_report()
    print(" => \033[1;34m#pragma once missing in \033[0;0m: ")

    for i in pragma_once_missing:
        print(" -", i.split(abs_proj_dir)[-1])

    sys.exit("Missing include guard for some source files")
else:
    print(" => \033[1;34mInclude guard status \033[0;0m: OK !")
