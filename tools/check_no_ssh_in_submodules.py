import os
import subprocess
import sys

import shamrock_tool_banner

shamrock_tool_banner.print_tool_info("No SSH in git submodules remote")
abs_proj_dir = os.path.join(os.path.dirname(__file__), "..")
abs_src_dir = os.path.join(abs_proj_dir, "src")


check_line = R"""No ssh in git submodules remote"""


def check_no_ssh_in_submodules():
    offenses = []

    with open(os.path.join(abs_proj_dir, ".gitmodules")) as file:
        for line in file:
            if "git@" in line:
                offenses.append(line)

    if len(offenses) > 0:
        print(" => \033[1;31mSSH remote found in .gitmodules\033[0;0m :")
        for line in offenses:
            print(line)

    return len(offenses) == 0


if check_no_ssh_in_submodules():
    print(" => \033[1;34mSSH remote status \033[0;0m: OK !")
else:
    sys.exit(" => \033[1;31mSSH remote found in .gitmodules\033[0;0m")
