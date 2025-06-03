import argparse
import os
import pathlib

import colorama
from colorama import Fore, Style

colorama.init()
print(
    "\n" + Fore.BLUE + Style.BRIGHT + "   >>> Test Pipeline SPHIVE <<<   " + Style.RESET_ALL + "\n"
)


parser = argparse.ArgumentParser(description="Configure utility for the code")

parser.add_argument("--ninja", action="store_true", help="use NINJA build system instead of Make")
parser.add_argument("--cuda", action="store_true", help="use CUDA instead of OPENCL")
parser.add_argument("llvm_root", help="llvm location", type=str)

args = parser.parse_args()


abs_proj_dir = os.path.abspath(os.path.join(__file__, "../.."))
abs_src_dir = os.path.join(abs_proj_dir, "src")
abs_build_dir_src = os.path.join(abs_proj_dir, "build_pipe")

abs_llvm_dir = os.path.abspath(os.path.join(os.getcwd(), args.llvm_root))

sycl_comp = "DPC++"


print(Fore.BLUE + Style.BRIGHT + "Project directory : " + Style.RESET_ALL + abs_proj_dir)
print(Fore.BLUE + Style.BRIGHT + " Source directory : " + Style.RESET_ALL + abs_src_dir)
print(
    Fore.BLUE + Style.BRIGHT + " Build  directory prefix : " + Style.RESET_ALL + abs_build_dir_src
)
print()
print(Fore.BLUE + Style.BRIGHT + "LLVM directory    : " + Style.RESET_ALL + abs_llvm_dir)
print(Fore.BLUE + Style.BRIGHT + "SyCL compiler     : " + Style.RESET_ALL + sycl_comp)


def generate_cmake_conf(build_dir):
    cmake_conf_cmd = "cmake"
    cmake_conf_cmd += " -S " + abs_src_dir
    cmake_conf_cmd += " -B " + build_dir

    if args.ninja:
        cmake_conf_cmd += " -G Ninja"

    if sycl_comp == "DPC++":
        cmake_conf_cmd += " -DSyCL_Compiler=DPC++"
        cmake_conf_cmd += " -DDPCPP_INSTALL_LOC=" + os.path.join(abs_llvm_dir, "build")
        if args.cuda:
            cmake_conf_cmd += " -DSyCL_Compiler_BE=CUDA"

    cmake_conf_cmd += " -DBUILD_TEST=true"

    return cmake_conf_cmd


def generate_cmake_comp(build_dir):
    cmake_comp_cmd = "cmake"
    cmake_comp_cmd += " --build"
    cmake_comp_cmd += " " + build_dir

    return cmake_comp_cmd


test_exe_filename = "shamrock_test"


config_names = ["ss", "ds", "sm", "dm", "sd", "dd"]

flags_list = [
    "-DMorton_precision=single -DPhysics_precision=single",
    "-DMorton_precision=double -DPhysics_precision=single",
    "-DMorton_precision=single -DPhysics_precision=mixed",
    "-DMorton_precision=double -DPhysics_precision=mixed",
    "-DMorton_precision=single -DPhysics_precision=double",
    "-DMorton_precision=double -DPhysics_precision=double",
]

process_cnt = [4 for i in range(len(config_names))]


# prepare tmp folders
for cid in range(len(config_names)):
    os.system("mkdir coverage_src_" + config_names[cid])


# configure step
for cid in range(len(config_names)):
    cmake_cmd_cov = generate_cmake_conf(abs_build_dir_src + "/conf_" + config_names[cid])
    cmake_cmd_cov += " -DCMAKE_BUILD_TYPE=COVERAGE_MAP " + flags_list[cid]

    print("\n" + Fore.BLUE + Style.BRIGHT + "Running : " + Style.RESET_ALL + cmake_cmd_cov + "\n")
    os.system(cmake_cmd_cov)


# compilation step
for cid in range(len(config_names)):
    cmake_comp_cmd = generate_cmake_comp(abs_build_dir_src + "/conf_" + config_names[cid])

    print("\n" + Fore.BLUE + Style.BRIGHT + "Running : " + Style.RESET_ALL + cmake_comp_cmd + "\n")
    os.system(cmake_comp_cmd)


print(
    "\n"
    + Fore.BLUE
    + Style.BRIGHT
    + "Running : "
    + Style.RESET_ALL
    + "find "
    + abs_build_dir_src
    + ' -iname "*.gcda" -exec rm {} \;'
    + "\n"
)
os.system("find " + abs_build_dir_src + ' -iname "*.gcda" -exec rm {} \;')

# running step
for cid in range(len(config_names)):
    abs_build_dir = abs_build_dir_src + "/conf_" + config_names[cid]

    runcmd = "mpirun --show-progress "

    for i in range(process_cnt[cid]):
        runcmd += (
            '-n 1 -x LLVM_PROFILE_FILE="'
            + abs_build_dir
            + "/program"
            + str(i)
            + '.profraw" '
            + abs_build_dir
            + "/"
            + test_exe_filename
        )

        if i < process_cnt[cid] - 1:
            runcmd += " : "

    print("\n" + Fore.BLUE + Style.BRIGHT + "Running : " + Style.RESET_ALL + runcmd + "\n")
    os.system(runcmd)

    runcmd = "mv unit_test_report.json coverage_src_" + config_names[cid]
    print("\n" + Fore.BLUE + Style.BRIGHT + "Running : " + Style.RESET_ALL + runcmd + "\n")
    os.system(runcmd)

# merge profiling data

for cid in range(len(config_names)):
    llvm_profdata_cmd = abs_llvm_dir + "/build/bin/llvm-profdata merge "

    abs_build_dir = abs_build_dir_src + "/conf_" + config_names[cid]

    for i in range(process_cnt[cid]):
        llvm_profdata_cmd += abs_build_dir + "/program" + str(i) + ".profraw "

    llvm_profdata_cmd += "-o " + abs_build_dir + "/program.profdata"

    print(
        "\n" + Fore.BLUE + Style.BRIGHT + "Running : " + Style.RESET_ALL + llvm_profdata_cmd + "\n"
    )
    os.system(llvm_profdata_cmd)


for cid in range(len(config_names)):
    llvmcovshow = abs_llvm_dir + "/build/bin/llvm-cov show "

    abs_build_dir = abs_build_dir_src + "/conf_" + config_names[cid]

    llvmcovshow += " " + abs_build_dir + "/" + test_exe_filename + " "

    llvmcovshow += (
        " -instr-profile="
        + abs_build_dir
        + "/program.profdata -use-color --format html --ignore-filename-regex=/tmp/* -output-dir=coverage_src_"
        + config_names[cid]
    )

    print("\n" + Fore.BLUE + Style.BRIGHT + "Running : " + Style.RESET_ALL + llvmcovshow + "\n")
    os.system(llvmcovshow)


# generate_report

folder_report = "coverage_report"

from lxml import etree

print("making " + folder_report)

try:
    os.mkdir(folder_report)
except:
    os.system("rm -r " + folder_report)
    os.mkdir(folder_report)

os.system("cp " + "coverage_src_" + config_names[0] + "/style.css " + folder_report)


report_html_main = ""
head = ""
header_info_clang = ""
version_clang_info = ""
html_tables = []

for cid in range(len(config_names)):
    f_in_html = open("coverage_src_" + config_names[cid] + "/index.html", "r")

    str_in_html = f_in_html.read()

    html_in = etree.HTML(str_in_html)

    head = (etree.tostring(html_in[0], pretty_print=True, method="html")).decode("ASCII")
    # print(head)

    body_result = html_in[1]

    header_info_clang = (etree.tostring(body_result[0], pretty_print=True, method="html")).decode(
        "ASCII"
    )
    header_info_clang += (etree.tostring(body_result[1], pretty_print=True, method="html")).decode(
        "ASCII"
    )
    header_info_clang += (etree.tostring(body_result[2], pretty_print=True, method="html")).decode(
        "ASCII"
    )

    html_tables.append(
        (
            "<h4>"
            + flags_list[cid]
            + "</h4>\n"
            + (etree.tostring(body_result[3], pretty_print=True, method="html")).decode("ASCII")
        ).replace(str(abs_src_dir), "_" + config_names[cid])
    )

    version_clang_info = (etree.tostring(body_result[4], pretty_print=True, method="html")).decode(
        "ASCII"
    )

    result = etree.tostring(html_in, pretty_print=True, method="html")
    # print(result)

    f_in_html.close()

f_out_html = open(folder_report + "/index.html", "w")


final_str_html = ""

final_str_html += "<!doctype html><html>\n" + head + "<body>"


# add test_result info

import json

final_str_html += "<h2>Test Report</h2>"


for cid in config_names:
    fold_n = folder_report + "/test_" + cid
    try:
        os.mkdir(fold_n)
    except:
        os.system("rm -r " + fold_n)
        os.mkdir(fold_n)

    os.system("cp " + folder_report + "/style.css " + fold_n)

for cid in range(len(config_names)):
    final_str_html += "<h4>" + flags_list[cid] + "</h4>\n"

    final_str_html += """
    <div class="centered"><table>
    <tbody><tr>
    <td class="column-entry-bold">Test name</td>
    <td class="column-entry-bold">MPI</td>
    <td class="column-entry-bold">Assert count</td>
    <td class="column-entry-bold">Assert succes</td>
    <td class="column-entry-bold">Succes ratio</td>
    </tr>
    """

    jsn_file = open("coverage_src_" + config_names[cid] + "/unit_test_report.json", "r")

    jsn = json.load(jsn_file)

    for k in jsn.keys():
        final_str_html += '<tr class="light-row">'

        pref = ""

        # print(k , jsn[k])

        try:
            if jsn[k]["succes_rate"] == 1.0:
                pref = '<td class="column-entry-green">'
            elif jsn[k]["succes_rate"] > 0.5:
                pref = '<td class="column-entry-yellow">'
            else:
                pref = '<td class="column-entry-red">'
        except:
            pref = '<td class="column-entry-red">'

        sub_file_assert_log = (
            folder_report + "/test_" + config_names[cid] + "/" + k.replace("/", "_") + ".html"
        )

        final_str_html += (
            pref
            + "<pre>"
            + '<a href="test_'
            + config_names[cid]
            + "/"
            + k.replace("/", "_")
            + ".html"
            + '">'
            + k
            + "</a></pre></td>"
        )
        final_str_html += pref + "<pre>" + str(jsn[k]["mpi"]) + "</pre></td>"
        final_str_html += pref + "<pre>" + str(jsn[k]["total_assert_cnt"]) + "</pre></td>"
        final_str_html += pref + "<pre>" + str(jsn[k]["succes_assert_cnt"]) + "</pre></td>"

        try:
            final_str_html += (
                pref + "<pre>" + str(int(jsn[k]["succes_rate"] * 100)) + "%</pre></td>"
            )
        except:
            final_str_html += pref + "<pre>None</pre></td>"

        final_str_html += "</tr>"

        try:
            asserts_ = jsn[k]["asserts"]

            f_subfile_assert_log = open(sub_file_assert_log, "w")

            subfile_assert_html = "<!doctype html><html>\n" + head + "<body>"

            subfile_assert_html += """
                <div class="centered"><table>
                <tbody><tr>
                <td class="column-entry-bold">Assert desc</td>
                <td class="column-entry-bold">node id</td>
                <td class="column-entry-bold">succes</td>
                </tr>
                """

            for ass in asserts_:
                for jj in range(len(jsn[k]["asserts"][ass])):
                    subfile_assert_html += '<tr class="light-row">'

                    pref = ""

                    if jsn[k]["asserts"][ass][jj]:
                        pref = '<td class="column-entry-green">'
                    else:
                        pref = '<td class="column-entry-red">'

                    subfile_assert_html += pref + "<pre>" + str(ass) + "</pre></td>"
                    subfile_assert_html += pref + "<pre>" + str(jj) + "</pre></td>"
                    subfile_assert_html += (
                        pref + "<pre>" + str(jsn[k]["asserts"][ass][jj]) + "</pre></td>"
                    )

                    subfile_assert_html += "</tr>"

            subfile_assert_html += "</tbody></table></div></body>" + "\n</html>"
            f_subfile_assert_log.write(subfile_assert_html)
        except:
            print("no asserts for", k)

    # <tr class="light-row">
    # <td><pre><a href="coverage_ss/io/logger.hpp.html">io/logger.hpp</a></pre></td>
    # <td class="column-entry-green"><pre> 100.00% (3/3)</pre></td>
    # <td class="column-entry-green"><pre> 100.00% (13/13)</pre></td>
    # <td class="column-entry-green"><pre> 100.00% (3/3)</pre></td>
    # <td class="column-entry-green"><pre>- (0/0)</pre></td>
    # </tr>

    final_str_html += "</tbody></table></div>"


final_str_html += header_info_clang

for sttr in html_tables:
    final_str_html += sttr

final_str_html += version_clang_info + "</body>" + "\n</html>"

# print(str(abs_src_dir))

final_str_html = final_str_html.replace(str(abs_src_dir), "")

for cid in range(len(config_names)):
    os.system(
        "mv "
        + "coverage_src_"
        + config_names[cid]
        + "/coverage"
        + abs_src_dir
        + " "
        + folder_report
        + "/coverage_"
        + config_names[cid]
    )


f_out_html.write(final_str_html)


# fix paths from coverage subfolders
import glob
import re

for root, dirs, files in os.walk(folder_report):
    for file in files:
        if file.endswith(".html"):
            if not "index.html" in file:
                str_file = open(root + "/" + str(file), "r").read()

                relat_path_stylecss = str(os.path.relpath(folder_report, root + "/" + str(file)))

                str_out = re.sub(
                    "href='?(.*?)/style.css'",
                    "href='" + relat_path_stylecss + "/" + folder_report + "/style.css'",
                    str_file,
                    flags=re.DOTALL,
                )
                str_out = str_out.replace(abs_src_dir, "src")

                str_file = open(root + "/" + str(file), "w").write(str_out)


exit()
# python test_pipeline.py --ninja --cuda ../../llvm
