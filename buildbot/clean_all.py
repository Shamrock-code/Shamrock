import os
import argparse

import colorama
from colorama import Fore
from colorama import Style
import pathlib



colorama.init()
print("\n"+Fore.BLUE + Style.BRIGHT + "   >>> Compilation utility for SPHIVE <<<   "+ Style.RESET_ALL + "\n")


abs_proj_dir = os.path.abspath(os.path.join(__file__, "../.."))
abs_src_dir = os.path.join(abs_proj_dir,"src")
abs_build_dir = os.path.join(abs_proj_dir,"build")


print(Fore.BLUE + Style.BRIGHT + "Project directory : "+ Style.RESET_ALL + abs_proj_dir)
print(Fore.BLUE + Style.BRIGHT + " Source directory : "+ Style.RESET_ALL + abs_src_dir)
print(Fore.BLUE + Style.BRIGHT + " Build  directory : "+ Style.RESET_ALL + abs_build_dir)
print(Fore.BLUE + Style.BRIGHT + " Build pipe directory : "+ Style.RESET_ALL + abs_build_dir + "_pipe")
print()

os.system("rm -r "+ abs_build_dir + " " + abs_build_dir + "_pipe")

os.system("rm -r " + abs_proj_dir + "/buildbot/coverage_* " + abs_proj_dir + "/buildbot/test_report ")

os.system("rm -r " + abs_proj_dir + "/buildbot/log_full_*")

os.system("rm *.dot *.llvm *.llvm.off")
