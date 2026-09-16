import importlib.util
import os
import shutil
import subprocess
import sys

psutil_spec = importlib.util.find_spec("psutil")
psutil_found = psutil_spec is not None

if psutil_found:
    import psutil


def is_ninja_available():
    return not (shutil.which("ninja") == None)


def get_available_ram_macos():
    page_size = int(subprocess.check_output(["sysctl", "-n", "hw.pagesize"]).strip())
    vm_stat = subprocess.check_output(["vm_stat"]).decode("utf-8")

    stats = {}
    for line in vm_stat.splitlines()[1:]:
        if ":" in line:
            key, val = line.split(":")
            stats[key.strip()] = int(val.strip().rstrip("."))

    # Available = Free + Inactive + Speculative pages
    free_pages = stats.get("Pages free", 0)
    inactive_pages = stats.get("Pages inactive", 0)
    speculative_pages = stats.get("Pages speculative", 0)

    available_bytes = (free_pages + inactive_pages + speculative_pages) * page_size
    return available_bytes


def get_avail_mem():
    if psutil_found:
        return (psutil.virtual_memory().available) / 1e6

    try:
        if sys.platform == "darwin":
            return get_available_ram_macos()

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
    except (OSError, subprocess.SubprocessError, ValueError, KeyError):
        tool = "vm_stat/sysctl" if sys.platform == "darwin" else "free"
        print(f"Available memory can not be detected using {tool}")
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
        print("   -> available memory:", avail)
        print("   -> core count:", os.cpu_count())
        print("   -> free memory /cores :", avail / os.cpu_count())
        cnt = int(avail / MAX_COMP_SZ)
        limit = True
        cnt = max(cnt, 1)
        print("   -> limiting to", cnt, "cores")

    env_nproc = os.environ.get("SHAMROCK_BUILD_NPROC")
    if env_nproc:
        try:
            nproc = int(env_nproc)
        except ValueError:
            print(f"-- invalid SHAMROCK_BUILD_NPROC={env_nproc!r}, ignoring")
        else:
            if nproc < 1:
                print(f"-- SHAMROCK_BUILD_NPROC={nproc} must be >= 1, ignoring")
            else:
                if nproc < cnt:
                    print(f"-- SHAMROCK_BUILD_NPROC={nproc}, limiting compilation to {nproc} cores")
                    cnt = nproc
                    limit = True
                elif not limit:
                    # Explicit request: force -j even when memory did not limit
                    print(f"-- SHAMROCK_BUILD_NPROC={nproc}, using {nproc} compile jobs")
                    cnt = nproc
                    limit = True

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
    elif buildtype == "msan":
        cmake_buildt = "MSAN"
    elif buildtype == "tsan":
        cmake_buildt = "TSAN"
    elif buildtype == "coverage":
        cmake_buildt = "COVERAGE"
    else:
        raise ValueError("Unknown build type")

    return gen, gen_opt, cmake_gen, cmake_buildt
