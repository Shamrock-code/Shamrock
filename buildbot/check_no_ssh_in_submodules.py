import os
import subprocess
import sys

from lib.buildbot import *

print_buildbot_info("include guard check tool")

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
