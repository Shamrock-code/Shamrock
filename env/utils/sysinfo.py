import importlib.util
import os
import shutil

psutil_spec = importlib.util.find_spec("psutil")
psutil_found = psutil_spec is not None

if psutil_found:
    import psutil


def is_ninja_available():
    return not (shutil.which("ninja") == None)


def get_avail_mem():
    import subprocess

    if psutil_found:
        return (psutil.virtual_memory().available) / 1e6

    try:
        free_available = not (os.popen("free -m -t").read() == "")

        if free_available:
            free_res = os.popen("free -m -t").readlines()[1:]
            out = 0
            for l in free_res:
                l = l.split()[1:]
                if len(l) < 6:
                    tot_m, used_m, free_m = map(int, l)
                    out = max(out, free_m)
                else:
                    tot_m, used_m, free_m, sharedmem, bufcache, avail = map(int, l)
                    out = max(out, free_m)
                    out = max(out, avail)

            return out
        # else:
        #    print(os.popen('vm_stat | grep page size').readlines())
    except:
        print("Available memory can not be detected using free")
        print("Error was :")
        import traceback

        print(traceback.format_exc())

    print("Available memory can not be detected -> assuming 16Go")
    return 1e9 * 16


def should_limit_comp_cores():
    MAX_COMP_SZ = 1e9
    avail = get_avail_mem() * 1e6

    limit = False
    cnt = os.cpu_count()

    avail_per_cores = avail / os.cpu_count()
    if avail_per_cores < MAX_COMP_SZ:
        print("-- low memory per cores, limitting number of thread for compilation")
        print("   ->  free memory /cores :", avail / os.cpu_count())
        cnt = int(avail / MAX_COMP_SZ)
        limit = True
        if cnt < 1:
            cnt = 1
        print("   ->  limiting to", cnt, "cores")

    return limit, cnt


def select_generator(args, buildtype):
    limit_cores, cores = should_limit_comp_cores()

    gen = "make"
    gen_opt = ""

    if args.gen == None:
        if is_ninja_available():
            gen = "ninja"
    else:
        gen = args.gen

    cmake_gen = ""
    if gen == "make":
        cmake_gen = '"Unix Makefiles"'
        gen_opt = " -j " + str(cores)
    elif gen == "ninja":
        cmake_gen = '"Ninja"'
        if limit_cores:
            gen_opt = " -j " + str(cores)
        else:
            gen_opt = ""
    else:
        raise "unknown generator " + gen

    if args.gen == None:
        print("-- generator not specified, defaulting to :", gen)

    cmake_buildt = "Release"
    if buildtype == "release":
        cmake_buildt = "Release"
    elif buildtype == "debug":
        cmake_buildt = "Debug"
    elif buildtype == "asan":
        cmake_buildt = "ASAN"
    elif buildtype == "ubsan":
        cmake_buildt = "UBSAN"
    elif buildtype == "coverage":
        cmake_buildt = "COVERAGE"
    else:
        raise "Unknown build type"

    return gen, gen_opt, cmake_gen, cmake_buildt
