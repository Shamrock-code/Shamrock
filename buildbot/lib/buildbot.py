import os
import re
import subprocess
from enum import Enum
from pathlib import Path

# start allow utf-8
title_wide = """
  █████████  █████   █████   █████████   ██████   ██████ ███████████      ███████      █████████  █████   ████
 ███░░░░░███░░███   ░░███   ███░░░░░███ ░░██████ ██████ ░░███░░░░░███   ███░░░░░███   ███░░░░░███░░███   ███░
░███    ░░░  ░███    ░███  ░███    ░███  ░███░█████░███  ░███    ░███  ███     ░░███ ███     ░░░  ░███  ███
░░█████████  ░███████████  ░███████████  ░███░░███ ░███  ░██████████  ░███      ░███░███          ░███████
 ░░░░░░░░███ ░███░░░░░███  ░███░░░░░███  ░███ ░░░  ░███  ░███░░░░░███ ░███      ░███░███          ░███░░███
 ███    ░███ ░███    ░███  ░███    ░███  ░███      ░███  ░███    ░███ ░░███     ███ ░░███     ███ ░███ ░░███
░░█████████  █████   █████ █████   █████ █████     █████ █████   █████ ░░░███████░   ░░█████████  █████ ░░████
 ░░░░░░░░░  ░░░░░   ░░░░░ ░░░░░   ░░░░░ ░░░░░     ░░░░░ ░░░░░   ░░░░░    ░░░░░░░      ░░░░░░░░░  ░░░░░   ░░░░
"""

title_normal = """
███████╗██╗  ██╗ █████╗ ███╗   ███╗██████╗  ██████╗  ██████╗██╗  ██╗
██╔════╝██║  ██║██╔══██╗████╗ ████║██╔══██╗██╔═══██╗██╔════╝██║ ██╔╝
███████╗███████║███████║██╔████╔██║██████╔╝██║   ██║██║     █████╔╝
╚════██║██╔══██║██╔══██║██║╚██╔╝██║██╔══██╗██║   ██║██║     ██╔═██╗
███████║██║  ██║██║  ██║██║ ╚═╝ ██║██║  ██║╚██████╔╝╚██████╗██║  ██╗
╚══════╝╚═╝  ╚═╝╚═╝  ╚═╝╚═╝     ╚═╝╚═╝  ╚═╝ ╚═════╝  ╚═════╝╚═╝  ╚═╝
"""

title_small = """
╔═╗╦ ╦╔═╗╔╦╗╦═╗╔═╗╔═╗╦╔═
╚═╗╠═╣╠═╣║║║╠╦╝║ ║║  ╠╩╗
╚═╝╩ ╩╩ ╩╩ ╩╩╚═╚═╝╚═╝╩ ╩
"""
# end allow utf-8

abs_proj_dir = os.path.abspath(os.path.join(__file__, "../../.."))
abs_src_dir = os.path.join(abs_proj_dir, "src")


def print_buildbot_info(utility_name):

    col_cnt = 100

    try:
        col_cnt = os.get_terminal_size().columns
    except:
        print("Warn : couldn't get terminal size")

    if col_cnt > 112:
        print(title_wide)
    elif col_cnt > 69:  # nice
        print(title_normal)
    else:
        print(title_small)

    print("\033[1;90m" + "-" * col_cnt + "\033[0;0m\n")

    print("\033[1;34mCurrent tool      \033[0;0m: ", utility_name)
    print("\033[1;34mProject directory \033[0;0m: ", abs_proj_dir)
    print("\033[1;34mSource  directory \033[0;0m: ", abs_src_dir)

    print()

    str_git = os.popen("git log -n 1 --decorate=full").read()

    git_hash = str_git.split()[1]
    git_head = str_git[str_git.find("HEAD -> ") + 8 : str_git.find(")")]

    git_head = git_head.split(",")

    if len(git_head) == 1:
        git_head = "\033[1;92m" + git_head[0] + "\033[0;0m"
    else:
        git_head = "\033[1;92m" + git_head[0] + "\033[0;0m , \033[1;91m" + git_head[0] + "\033[0;0m"

    print("\033[1;34mGit status \033[0;0m: ")
    print("     \033[1;93mcommit \033[0;0m: ", git_hash)
    print("     \033[1;36mHEAD   \033[0;0m: ", git_head)
    print("     \033[1;31mmodified files\033[0;0m (since last commit):")
    print(os.popen('git diff-index --name-only HEAD -- | sed "s/^/        /g"').read())
    print("\033[1;90m" + "-" * col_cnt + "\033[0;0m\n")


def run_cmd(str):

    col_cnt = 64

    try:
        col_cnt = os.get_terminal_size().columns
    except:
        print("Warn : couldn't get terminal size")

    print("\033[1;34mRunning \033[0;0m: " + str)
    print("\033[1;90mcmd out " + "-" * (col_cnt - 8) + "\033[0;0m")
    ret = os.system(str)
    print("\033[1;90m" + "-" * col_cnt + "\033[0;0m")
    if not (ret == 0):
        exit(ret)


def chdir(path):
    print("\033[1;34mcd \033[0;0m: " + path)
    os.chdir(path)


class BuildSystem(Enum):
    Makefiles = 1
    Ninja = 2


class SyCLBE(Enum):
    Host = 1
    OpenMP = 2
    OpenCL = 3
    CUDA = 4
    HIP = 5


class SyclCompiler:
    DPCPP = 1
    DPCPP_SUPPORT = [SyCLBE.Host, SyCLBE.CUDA]

    HipSYCL = 2
    HipSYCL_SUPPORT = [SyCLBE.OpenMP]


class Targets(Enum):
    SHAMROCK = 1
    Test = 2
    Visu = 3


class BuildMode(Enum):
    Normal = 0
    Release = 1
    Debug = 2


class PrecisionMode(Enum):
    Single = 1
    Mixed = 2
    Double = 3


def is_ninja_available() -> bool:

    try:
        # pipe output to /dev/null for silence
        null = open("/dev/null", "w")
        subprocess.Popen(["ninja", "--version"], stdout=null, stderr=null)
        null.close()
        return True
    except OSError:
        return False


def get_default_build_system() -> BuildSystem:
    if is_ninja_available():
        return BuildSystem.Ninja
    else:
        return BuildSystem.Makefiles


def compile_prog(abs_build_dir):
    cmake_cmd = "cmake"
    cmake_cmd += " --build"
    cmake_cmd += " " + abs_build_dir

    run_cmd(cmake_cmd)


def get_current_buildsystem(abs_build_dir) -> BuildSystem:

    if os.path.isfile(abs_build_dir + "/build.ninja"):
        return BuildSystem.Ninja
    if os.path.isfile(abs_build_dir + "/Makefile"):
        return BuildSystem.Makefiles

    raise "buildsystem not recognized"


def clean_build_dir(abs_build_dir):
    current_build_sys = get_current_buildsystem(abs_build_dir)
    current_dir = os.getcwd()
    chdir(abs_build_dir)

    if current_build_sys == BuildSystem.Ninja:
        run_cmd("ninja clean")

    if current_build_sys == BuildSystem.Makefiles:
        run_cmd("make clean")

    chdir(current_dir)


# cmake -S src -B build -DSyCL_Compiler=HIPSYCL -DSyCL_Compiler_BE=OMP -DCOMP_ROOT_DIR=/home/tim/Documents/these/codes/shamrock_workspace/sycl_cpl/hipSYCL -G Ninja -DBUILD_SIM=true -DBUILD_VISU=true -DBUILD_TEST=true -DMorton_precision=single -DPhysics_precision=single
# cmake -S src -B build -DSyCL_Compiler=DPCPP -DSyCL_Compiler_BE=CUDA -DCOMP_ROOT_DIR=/home/tim/Documents/these/codes/shamrock_workspace/sycl_cpl/dpcpp -G Ninja -DBUILD_SIM=true -DBUILD_VISU=true -DBUILD_TEST=true -DMorton_precision=single -DPhysics_precision=single


def configure(
    src_dir: str,
    build_dir: str,
    compiler: SyclCompiler,
    backend: SyCLBE,
    compiler_dir: str,
    target_build_mode,
    build_sys,
    target_lst,
):
    print("\033[1;34mConfiguring SHAMROCK\033[0;0m")

    enabled_targets_str = ""
    if Targets.Test in target_lst:
        enabled_targets_str += "test "

    if Targets.SHAMROCK in target_lst:
        enabled_targets_str += "shamrock "

    if Targets.Visu in target_lst:
        enabled_targets_str += "visu"

    print("  -> \033[1;34mEnabled targets   : \033[0;0m" + enabled_targets_str)

    cmake_cmd = "cmake"
    cmake_cmd += " -S " + src_dir + "/.."
    cmake_cmd += " -B " + build_dir

    if target_build_mode == BuildMode.Normal:
        cmake_cmd += ""
    elif target_build_mode == BuildMode.Release:
        cmake_cmd += " -DCMAKE_BUILD_TYPE=Release"
    elif target_build_mode == BuildMode.Debug:
        cmake_cmd += " -DCMAKE_BUILD_TYPE=Debug"

    if build_sys == BuildSystem.Ninja:
        cmake_cmd += " -G Ninja"

    if build_sys == BuildSystem.Makefiles:
        cmake_cmd += ' -G "Unix Makefiles"'

    if compiler == SyclCompiler.DPCPP:
        cmake_cmd += " -DSyCL_Compiler=DPCPP"
        if not backend in SyclCompiler.DPCPP_SUPPORT:
            raise "error backend not supported by dpcpp"

        if backend == SyCLBE.CUDA:
            cmake_cmd += " -DSyCL_Compiler_BE=CUDA"
        cmake_cmd += " -DCMAKE_CXX_COMPILER=" + str(os.path.abspath(compiler_dir)) + "/bin/clang++"

    elif compiler == SyclCompiler.HipSYCL:
        cmake_cmd += " -DSyCL_Compiler=HIPSYCL"
        if not backend in SyclCompiler.HipSYCL_SUPPORT:
            raise "error backend not supported by hipsycl"

        if backend == SyCLBE.OpenMP:
            cmake_cmd += " -DSyCL_Compiler_BE=OMP"

        cmake_cmd += " -DCMAKE_CXX_COMPILER=" + str(os.path.abspath(compiler_dir)) + "/bin/syclcc"
        compiler_arg = (
            "--hipsycl-cpu-cxx=g++ --hipsycl-config-file="
            + str(os.path.abspath(compiler_dir))
            + "/etc/hipSYCL/syclcc.json --hipsycl-targets='omp' --hipsycl-platform=cpu"
        )
        cmake_cmd += ' -DCMAKE_CXX_FLAGS="' + compiler_arg + '"'

    cmake_cmd += " -DCOMP_ROOT_DIR=" + str(os.path.abspath(compiler_dir))

    if Targets.Test in target_lst:
        cmake_cmd += " -DBUILD_TEST=true"

    if Targets.SHAMROCK in target_lst:
        cmake_cmd += " -DBUILD_SIM=true"

    if Targets.Visu in target_lst:
        cmake_cmd += " -DBUILD_VISU=true"

    cmake_cmd += " -DMorton_precision=single"
    cmake_cmd += " -DPhysics_precision=single"
    run_cmd(cmake_cmd)


# configure_dpcpp("../src", "../build2","../../llvm",BuildSystem.Ninja,SyCLBE.CUDA,[Targets.Test],PrecisionMode.Single,PrecisionMode.Single)
def configure_dpcpp(
    src_dir,
    build_dir,
    llvm_root,
    target_build_mode,
    build_sys,
    sycl_BE,
    target_lst,
    morton_prec,
    phy_prec,
):

    print("\033[1;34mConfiguring SHAMROCK with DPC++\033[0;0m")

    enabled_targets_str = ""
    if Targets.Test in target_lst:
        enabled_targets_str += "test "

    if Targets.SHAMROCK in target_lst:
        enabled_targets_str += "shamrock "

    if Targets.Visu in target_lst:
        enabled_targets_str += "visu"

    print("  -> \033[1;34mEnabled targets   : \033[0;0m" + enabled_targets_str)

    cmake_cmd = "cmake"
    cmake_cmd += " -S " + src_dir
    cmake_cmd += " -B " + build_dir

    if target_build_mode == BuildMode.Normal:
        cmake_cmd += ""
    elif target_build_mode == BuildMode.Release:
        cmake_cmd += " -DCMAKE_BUILD_TYPE=Release"
    elif target_build_mode == BuildMode.Debug:
        cmake_cmd += " -DCMAKE_BUILD_TYPE=Debug"

    if build_sys == BuildSystem.Ninja:
        cmake_cmd += " -G Ninja"

    if build_sys == BuildSystem.Makefiles:
        cmake_cmd += ' -G "Unix Makefiles"'

    cmake_cmd += " -DSyCL_Compiler=DPC++"
    cmake_cmd += " -DDPCPP_INSTALL_LOC=" + os.path.join(llvm_root, "build")

    if sycl_BE == SyCLBE.CUDA:
        cmake_cmd += " -DSyCL_Compiler_BE=CUDA"

    if Targets.Test in target_lst:
        cmake_cmd += " -DBUILD_TEST=true"

    if Targets.SHAMROCK in target_lst:
        cmake_cmd += " -DBUILD_SIM=true"

    if Targets.Visu in target_lst:
        cmake_cmd += " -DBUILD_VISU=true"

    if morton_prec == PrecisionMode.Single:
        print("  -> \033[1;34mMorton mode       : \033[0;0msingle")
        cmake_cmd += " -DMorton_precision=single"
    elif morton_prec == PrecisionMode.Double:
        print("  -> \033[1;34mMorton mode       : \033[0;0mdouble")
        cmake_cmd += " -DMorton_precision=double"
    else:
        raise "unknown morton precision mode : " + str(morton_prec)

    if phy_prec == PrecisionMode.Single:
        print("  -> \033[1;34mPrecision mode    : \033[0;0msingle")
        cmake_cmd += " -DPhysics_precision=single"
    elif phy_prec == PrecisionMode.Mixed:
        print("  -> \033[1;34mPrecision mode    : \033[0;0mmixed")
        cmake_cmd += " -DPhysics_precision=mixed"
    elif phy_prec == PrecisionMode.Double:
        print("  -> \033[1;34mPrecision mode    : \033[0;0mdouble")
        cmake_cmd += " -DPhysics_precision=double"
    else:
        raise "unknown phy precision mode : " + str(phy_prec)

    # if args.xray:
    #    cmake_cmd += " -DXRAY_INSTRUMENTATION=true"

    run_cmd(cmake_cmd)


# regex to fin all new
# /\s+([^ ]+)\s*(=\s*new [^;]+;)/g
# replace by
#  $1 $2 \n      log_new($1,log_alloc_ln);\n

# regex to find all delete []
# /delete\s*\[\]\s*([^ ]+)\s*;/g
# replace by
# delete [] $1; log_delete($1);

# regex to find all delete
# /delete\s*([^ ]+)\s*;/g
# replace by
# delete $1; log_delete($1);


def patch_file(file, header_loc):

    incl_loc_head = str(os.path.relpath(header_loc, Path(file).parent))
    str_incl = '#include "' + incl_loc_head + '"\n\n'

    lines_in = ""
    with open(file, "r") as f_in:
        lines_in = f_in.read()

    # lines_in = re.sub(r"//[^\n]+",r"", lines_in)
    # lines_in = re.sub(r"\A(?s).*?\*\/(?-s)",r"", lines_in)

    lines_in = re.sub(
        r"(?<!_)delete\s*([^ ]+)\s*;", "{log_delete(\g<1>,log_alloc_ln);delete \g<1>;}", lines_in
    )

    lines_in = re.sub(
        r"(?<!_)delete\s*\[\]\s*([^ ]+)\s*;",
        "{log_delete(\g<1>,log_alloc_ln);delete[] \g<1>;}",
        lines_in,
    )

    lines_in = re.sub(
        r"=\s*new\s+([^\[(]+)(.*?);", r"= (\g<1> *) log_new(new \g<1>\g<2>,log_alloc_ln);", lines_in
    )

    splt_lnin = lines_in.split("\n")

    rel_file_loc = os.path.relpath(file, Path(header_loc).parent)

    splt_lnin2 = []

    line_cnt = 1
    for a in splt_lnin:
        splt_lnin2.append(a.replace("log_alloc_ln", '"' + rel_file_loc + ":" + str(line_cnt) + '"'))
        line_cnt += 1

    lines_in = ""
    for a in splt_lnin2:
        lines_in += a + "\n"

    if "#pragma once" in lines_in:
        dtt = lines_in.split("#pragma once")
        lines_in = dtt[0] + "\n" + "#pragma once\n" + str_incl + dtt[1]
    else:
        lines_in = str_incl + lines_in

    with open(file, "w") as f_out:
        f_out.write(lines_in)


def gen_mem_patched_dir(abs_src_dir, abs_patchedsrc_dir):
    run_cmd("rm -r " + abs_patchedsrc_dir)
    run_cmd("mkdir " + abs_patchedsrc_dir)
    run_cmd("cp -r " + abs_src_dir + "/* " + abs_patchedsrc_dir)

    lst = [path for path in Path(abs_patchedsrc_dir).rglob("*.cpp")] + [
        path for path in Path(abs_patchedsrc_dir).rglob("*.hpp")
    ]

    for path in lst:

        if not (str(path.name) == "mem_track.hpp"):

            abs_path_in = str(path.absolute())

            obj = os.stat(abs_path_in)

            print(
                "patching : " + os.path.relpath(abs_path_in)
            )  # + "modified time: {}".format(obj.st_mtime))
            patch_file(abs_path_in, abs_patchedsrc_dir + "/mem_track.hpp")


def run_test(node_cnt, run_only="", oversubscribe=False, supargs=""):

    args = " --run-only " + run_only

    if run_only == "":
        args = ""

    args += " -o " + "test_res_" + str(node_cnt) + " " + supargs

    compile_prog("../build")

    if node_cnt == 1:
        run_cmd("../build/shamrock_test" + args)
    else:

        str_node = ""
        for i in range(node_cnt - 1):
            str_node += str(i) + ","
        str_node += str(node_cnt - 1)

        if oversubscribe:
            run_cmd(
                "mpirun --oversubscribe -n "
                + str(node_cnt)
                + " -xterm "
                + str_node
                + "! ../build/shamrock_test"
                + args
            )
        else:
            run_cmd(
                "mpirun -n "
                + str(node_cnt)
                + " -xterm "
                + str_node
                + "! ../build/shamrock_test"
                + args
            )


def run_test_mempatch(node_cnt, run_only="", oversubscribe=False, supargs=""):

    args = " --run-only " + run_only

    if run_only == "":
        args = ""

    args += " -o " + "test_res_" + str(node_cnt) + " " + supargs

    gen_mem_patched_dir("../src", "../src_patched")

    configure_dpcpp(
        "../src_patched",
        "../build_patched",
        "../../llvm",
        BuildMode.Normal,
        BuildSystem.Ninja,
        SyCLBE.CUDA,
        [Targets.Test],
        PrecisionMode.Single,
        PrecisionMode.Single,
    )

    compile_prog("../build_patched")

    if node_cnt == 1:
        run_cmd("../build_patched/shamrock_test" + args)
    else:

        str_node = ""
        for i in range(node_cnt - 1):
            str_node += str(i) + ","
        str_node += str(node_cnt - 1)

        if oversubscribe:
            run_cmd(
                "mpirun --oversubscribe -n "
                + str(node_cnt)
                + " -xterm "
                + str_node
                + "! ../build_patched/shamrock_test"
                + args
            )
        else:
            run_cmd(
                "mpirun -n "
                + str(node_cnt)
                + " -xterm "
                + str_node
                + "! ../build_patched/shamrock_test"
                + args
            )
