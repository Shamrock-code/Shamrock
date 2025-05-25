import json
import math
from enum import Enum

Tex_template = r"""

\documentclass{article}

\usepackage[a4paper,total={170mm,260mm},left=20mm,top=20mm,]{geometry}



\usepackage{fancyhdr} % entêtes et pieds de pages personnalisés

\pagestyle{fancy}
\fancyhead[L]{\scriptsize \textsc{Test suite report}} % À changer
\fancyhead[R]{\scriptsize \textsc{\textsc{SHAMROCK}}} % À changer
\fancyfoot[C]{ \thepage}


\usepackage{titling}

\setlength{\droptitle}{-4\baselineskip} % Move the title up

\pretitle{\begin{center}\Huge\bfseries} % Article title formatting
\posttitle{\end{center}} % Article title closing formatting
\title{\textsc{SHAMROCK} test suite report} % Article title
\author{%
\textsc{Timothée David--Cléris}\thanks{timothee.david--cleris@ens-lyon.fr} \\[1ex] % Your name
\normalsize CRAL ENS de Lyon \\ % Your institution
}
\date{\today}

\usepackage{xcolor}
\definecolor{linkcolor}{rgb}{0,0,0.6}


\usepackage[ pdftex,colorlinks=true,
pdfstartview=ajustementV,
linkcolor= linkcolor,
citecolor= linkcolor,
urlcolor= linkcolor,
hyperindex=true,
hyperfigures=false]
{hyperref}



\usepackage{color}


\definecolor{GREEN}{rgb}{0,.7,0}
\definecolor{RED}{rgb}{.8,0,0}


\def\OK{\textcolor{GREEN}{OK}}
\def\FAIL{\textcolor{RED}{FAIL}}

\begin{document}
\maketitle

%%tabl_world_sz_res%%


\tableofcontents


%%content%%
\end{document}
"""


class ReportFormat(Enum):
    Tex = 1
    HTML = 1
    Txt = 1


def load_test_report(file):
    print(file)

    out_file = open(file, "r")
    lst_ln = out_file.readlines()
    out_file.close()

    cur_test = ""
    cur_world_sz = -1
    cur_world_rk = -1

    cur_assert = ""
    cur_assert_log = ""
    cur_assert_result = -1

    cur_assert_log_state = False

    dic_loaded = {}

    for l in lst_ln:
        if l.startswith(r"%test_name = "):
            test_name = l[l.find('"') + 1 : l.find('"', l.find('"') + 1)]
            cur_test = test_name
            # print(" -> starting_test", test_name)

            if not (cur_test in dic_loaded.keys()):
                dic_loaded[cur_test] = {}

        elif l.startswith(r"%end_test"):
            # print(" -> end_test", test_name)

            cur_test = ""
            cur_world_sz = -1
            cur_world_rk = -1

        # elif l.startswith(r"%world_size = "):
        #     cur_world_sz = int(l[len("%world_size = "):])
        #     print("     -> world size",cur_world_sz)

        #     dic_loaded[cur_test]["world_size"] = cur_world_sz

        elif l.startswith(r"%world_rank = "):
            cur_world_rk = int(l[len("%world_rank = ") :])
            # print("     -> world rank",cur_world_rk)

            dic_loaded[cur_test][cur_world_rk] = []

        elif l.startswith(r"%start_assert"):
            assert_name = l[l.find('"') + 1 : l.find('"', l.find('"') + 1)]
            cur_assert = assert_name
            # print("         -> start_assert",assert_name)

        elif l.startswith(r"%end_assert"):
            # print("         -> end_assert",cur_assert)

            dic_loaded[cur_test][cur_world_rk].append(
                {"name": cur_assert, "log": cur_assert_log, "result": cur_assert_result}
            )

            cur_assert = ""
            cur_assert_log = ""
            cur_assert_result = -1

        elif l.startswith(r"%result = "):
            cur_assert_result = int(l[len("%result = ") :])
            # print("             -> assert result",assert_name,cur_assert_result)

        elif l.startswith(r"%startlog"):
            cur_assert_log_state = True
            # print("             -> start log")
        elif l.startswith(r"%endlog"):
            cur_assert_log_state = False

            # print("             -> end log content : \n",cur_assert_log)

        elif cur_assert_log_state:
            cur_assert_log += l + "\n"

    return dic_loaded


def get_succes_count_data(dt):
    out_dic = {}
    for k_cur_test in dt.keys():
        tmp = {}

        sum_cnt_assert = 0
        sum_cnt_succes = 0

        for k_cur_wrk in dt[k_cur_test].keys():
            cnt_assert = 0
            cnt_succes = 0

            for asserts in dt[k_cur_test][k_cur_wrk]:
                cnt_assert += 1
                cnt_succes += asserts["result"]

            sum_cnt_assert += cnt_assert
            sum_cnt_succes += cnt_succes

            # print("test ",k_cur_test, "world size =",k_cur_wrk,"| succes rate =",cnt_succes,"/",len(dt[k_cur_test][k_cur_wrk]))
            tmp[k_cur_wrk] = {"suc_cnt": cnt_succes, "assert_cnt": cnt_assert}

        tmp["suc_cnt"] = sum_cnt_succes
        tmp["assert_cnt"] = sum_cnt_assert

        out_dic[k_cur_test] = tmp
    return out_dic


def make_tex_repport(dat):
    dic_int = {}

    for config_k in dat.keys():
        conf_dic = dat[config_k]

        for k in conf_dic.keys():
            if k.startswith("world_size="):
                wsz = int(k[len("world_size=") :])

                dic_int["world size = " + str(wsz)] = {}

    for config_k in dat.keys():
        conf_dic = dat[config_k]

        for k in conf_dic.keys():
            if k.startswith("world_size="):
                wsz = int(k[len("world_size=") :])

                dic_res = load_test_report(dat[config_k][k])
                dic_suc_cnt = get_succes_count_data(dic_res)

                cnt_test = 0
                cnt_succes = 0

                for ktest in dic_suc_cnt.keys():
                    # cnt_assert += dic_suc_cnt[ktest]["assert_cnt"]
                    # cnt_succes += dic_suc_cnt[ktest]["suc_cnt"]

                    cnt_test += 1
                    cnt_succes += dic_suc_cnt[ktest]["suc_cnt"] == dic_suc_cnt[ktest]["assert_cnt"]

                dic_int["world size = " + str(wsz)][dat[config_k]["description"]] = {
                    "results": dic_res,
                    "succes_cnt": dic_suc_cnt,
                    "global_suc_cnt": cnt_succes,
                    "global_test_cnt": cnt_test,
                }

    out_file = open("tmp.json", "w")
    json.dump(dic_int, out_file, indent=6)
    out_file.close()

    dic_suc_cnt_global = {}

    for kworldsz in dic_int.keys():
        cnt_config = 0
        cnt_succes = 0

        for kconfig in dic_int[kworldsz].keys():
            cnt_config += 1
            cnt_succes += (
                dic_int[kworldsz][kconfig]["global_suc_cnt"]
                == dic_int[kworldsz][kconfig]["global_test_cnt"]
            )

        dic_suc_cnt_global[kworldsz] = {
            "global_suc_cnt": cnt_succes,
            "global_config_cnt": cnt_config,
        }

    tabl_world_sz_res = ""

    tabl_world_sz_res += r""" \begin{center}
        \begin{tabular}{|c|c|c|}
        \hline
        World size & Status & Succesfull config / total number of config \\  \hline \hline
    """
    for kworldsz in dic_int.keys():
        config_suc_cnt = dic_suc_cnt_global[kworldsz]["global_suc_cnt"]
        config_cnt = dic_suc_cnt_global[kworldsz]["global_config_cnt"]

        succes = config_suc_cnt == config_cnt

        tabl_world_sz_res += kworldsz + " & "

        if succes:
            tabl_world_sz_res += "\OK & "
        else:
            tabl_world_sz_res += "\FAIL & "

        tabl_world_sz_res += (
            "$" + str(config_suc_cnt) + "/" + str(config_cnt) + r"$\\ \hline" + "\n"
        )

    tabl_world_sz_res += r"""
        \end{tabular}\end{center}
    """

    str_file = ""

    for kworldsz in dic_int.keys():
        str_file += (
            r"""
            \newpage
             \begin{center}
            \section{"""
            + kworldsz
            + r"""}
        """
        )

        str_file += r"""
            \begin{tabular}{|c|c|c|}
            \hline
            Config & Status & Succesfull tests / total number of tests \\  \hline \hline
        """
        for kconfig in dic_int[kworldsz].keys():
            test_suc_cnt = dic_int[kworldsz][kconfig]["global_suc_cnt"]
            test_cnt = dic_int[kworldsz][kconfig]["global_test_cnt"]

            succes = test_suc_cnt == test_cnt

            str_file += kconfig + " & "

            if succes:
                str_file += "\OK & "
            else:
                str_file += "\FAIL & "

            str_file += "$" + str(test_suc_cnt) + "/" + str(test_cnt) + r"$\\ \hline" + "\n"

        str_file += r"""
            \end{tabular}\end{center}
        """

        for kconfig in dic_int[kworldsz].keys():
            str_file += (
                r"""
                \subsection{"""
                + kconfig
                + r"""}
            """
            )

            str_file += r"""
            \begin{center}
                \begin{tabular}{|c|c|c|}
                \hline
                Test name & Status & Succesfull asserts / total number of asserts \\  \hline \hline
            """
            for ktest in dic_int[kworldsz][kconfig]["succes_cnt"].keys():
                assert_suc_cnt = dic_int[kworldsz][kconfig]["succes_cnt"][ktest]["suc_cnt"]
                assert_cnt = dic_int[kworldsz][kconfig]["succes_cnt"][ktest]["assert_cnt"]

                succes = assert_suc_cnt == assert_cnt

                str_file += r"\verb|" + ktest + "| & "

                if succes:
                    str_file += "\OK & "
                else:
                    str_file += "\FAIL & "

                str_file += "$" + str(assert_suc_cnt) + "/" + str(assert_cnt) + r"$\\ \hline" + "\n"

            str_file += r"""
                \end{tabular}\end{center}
            """

    out_tex = Tex_template.replace(r"%%tabl_world_sz_res%%", tabl_world_sz_res).replace(
        r"%%content%%", str_file
    )

    print(out_tex)

    out_file = open("test_repport.tex", "w")
    out_file.write(out_tex)
    out_file.close()


def make_report(format, out_res_map_file):
    out_file = open(out_res_map_file, "r")
    data = json.load(out_file)
    out_file.close()

    # print(data)

    if format == ReportFormat.Tex:
        make_tex_repport(data)


if __name__ == "__main__":
    make_report(ReportFormat.Tex, "../../test_pipeline/test_result_list.json")

    # dat_ld = load_test_report("/home/tim/Documents/these/codes/sycl_workspace/shamrock/test_pipeline/build_ss/test_res_2.sutest")
    # print(dat_ld)
    # print(get_succes_count_dat(dat_ld))
