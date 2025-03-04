from lib.buildbot import *

print_buildbot_info("compile tool")

abs_build_dir = os.path.join(abs_proj_dir, "build")

compile_prog(abs_build_dir)
