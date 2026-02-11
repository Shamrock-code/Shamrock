import argparse
import json

from lib.buildbot import *

parser = argparse.ArgumentParser(description="Test pipeline utility for the code")
parser.add_argument("llvm_root", help="llvm location", type=str)
args = parser.parse_args()

abs_llvm_dir = os.path.abspath(os.path.join(os.getcwd(), args.llvm_root))


print_buildbot_info("Test pipeline")

abs_test_pipeline_dir = os.path.join(abs_proj_dir, "test_pipeline")

run_cmd("rm -r " + abs_test_pipeline_dir)
run_cmd("mkdir " + abs_test_pipeline_dir)

abs_test_pipeline_dir_src = os.path.join(abs_test_pipeline_dir, "src_patched")

gen_mem_patched_dir(abs_src_dir, abs_test_pipeline_dir_src)


configure_dpcpp(
    abs_test_pipeline_dir_src,
    abs_test_pipeline_dir + "/build_ss",
    abs_llvm_dir,
    BuildMode.Release,
    BuildSystem.Ninja,
    SyCLBE.CUDA,
    [Targets.Test],
    PrecisionMode.Single,
    PrecisionMode.Single,
)
configure_dpcpp(
    abs_test_pipeline_dir_src,
    abs_test_pipeline_dir + "/build_sm",
    abs_llvm_dir,
    BuildMode.Release,
    BuildSystem.Ninja,
    SyCLBE.CUDA,
    [Targets.Test],
    PrecisionMode.Single,
    PrecisionMode.Mixed,
)
configure_dpcpp(
    abs_test_pipeline_dir_src,
    abs_test_pipeline_dir + "/build_sd",
    abs_llvm_dir,
    BuildMode.Release,
    BuildSystem.Ninja,
    SyCLBE.CUDA,
    [Targets.Test],
    PrecisionMode.Single,
    PrecisionMode.Double,
)
configure_dpcpp(
    abs_test_pipeline_dir_src,
    abs_test_pipeline_dir + "/build_ds",
    abs_llvm_dir,
    BuildMode.Release,
    BuildSystem.Ninja,
    SyCLBE.CUDA,
    [Targets.Test],
    PrecisionMode.Double,
    PrecisionMode.Single,
)
configure_dpcpp(
    abs_test_pipeline_dir_src,
    abs_test_pipeline_dir + "/build_dm",
    abs_llvm_dir,
    BuildMode.Release,
    BuildSystem.Ninja,
    SyCLBE.CUDA,
    [Targets.Test],
    PrecisionMode.Double,
    PrecisionMode.Mixed,
)
configure_dpcpp(
    abs_test_pipeline_dir_src,
    abs_test_pipeline_dir + "/build_dd",
    abs_llvm_dir,
    BuildMode.Release,
    BuildSystem.Ninja,
    SyCLBE.CUDA,
    [Targets.Test],
    PrecisionMode.Double,
    PrecisionMode.Double,
)

node_cnt = [1, 2, 3, 4, 8, 16, 32]

conf_dir_lst = [
    abs_test_pipeline_dir + "/build_ss",
    abs_test_pipeline_dir + "/build_sm",
    abs_test_pipeline_dir + "/build_sd",
    abs_test_pipeline_dir + "/build_ds",
    abs_test_pipeline_dir + "/build_dm",
    abs_test_pipeline_dir + "/build_dd",
]

conf_dir_desc = [
    "Morton = single Physical precision = single",
    "Morton = single Physical precision = mixed",
    "Morton = single Physical precision = double",
    "Morton = double Physical precision = single",
    "Morton = double Physical precision = mixed",
    "Morton = double Physical precision = double",
]

for dir_ in conf_dir_lst:
    compile_prog(dir_)

current_wdir = os.getcwd()

for dir_ in conf_dir_lst:
    os.chdir(dir_)

    for cnt in node_cnt:
        run_cmd(
            "mpirun --oversubscribe -n "
            + str(cnt)
            + " ./shamrock_test -o test_res_"
            + str(cnt)
            + ".sutest"
        )

os.chdir(current_wdir)

dic_test_res_load = {}

for dir_, desc in zip(conf_dir_lst, conf_dir_desc):
    conf_dic = {}

    for cnt in node_cnt:
        file_name = dir_ + "/test_res_" + str(cnt) + ".sutest"

        conf_dic["description"] = desc
        conf_dic["world_size=" + str(cnt)] = file_name

    dic_test_res_load[dir_] = conf_dic

print(dic_test_res_load)

out_file = open(abs_test_pipeline_dir + "/test_result_list.json", "w")
json.dump(dic_test_res_load, out_file, indent=6)
out_file.close()

from lib.make_report import *

make_report(ReportFormat.Tex, abs_test_pipeline_dir + "/test_result_list.json")

# TODO compile and show pdf report
