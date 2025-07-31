// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright (c) 2021-2025 Timothée David--Cléris <tim.shamrock@proton.me>
// SPDX-License-Identifier: CeCILL Free Software License Agreement v2.1
// Shamrock is licensed under the CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

/**
 * @file texTestReport.cpp
 * @author Timothée David--Cléris (tim.shamrock@proton.me)
 * @brief implementation of the Tex test report generation
 */

#include "shambase/string.hpp"
#include "shambackends/sycl.hpp"
#include "shamrock/version.hpp"
#include "shamsys/NodeInstance.hpp"
#include "shamtest/details/TestResult.hpp"
#include "texTestReport.hpp"
#include <algorithm>
#include <sstream>
#include <string>
#include <vector>

/// Tex report template to be used
std::string tex_template = R"==(

\documentclass[10pt]{article}

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

\usepackage{graphicx}
\usepackage{cprotect}

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



)==";

/// Footer of the document
std::string tex_template_end = R"(
    \end{document}
)";

namespace shamtest::details {

    /// Add the unittest section to the Tex report
    void add_unittest_section(std::stringstream &output, std::vector<TestResult> &results) {

        output << R"(\section{Unittests})"
               << "\n\n";

        using namespace details;

        std::unordered_map<std::string, u32> assert_test_count;
        std::unordered_map<std::string, u32> assert_test_count_success;

        for (TestResult &res : results) {
            if (res.type == Unittest) {
                assert_test_count[res.name] = 0;
            }
        }

        for (TestResult &res : results) {
            if (res.type == Unittest) {
                assert_test_count[res.name] += res.asserts.get_assert_count();
                assert_test_count_success[res.name] += res.asserts.get_assert_success_count();
            }
        }

        std::vector<std::string> strings;
        for (auto &[key, value] : assert_test_count) {
            strings.push_back(key);
        }
        std::sort(strings.begin(), strings.end());

        std::string table_header = R"==(
            \begin{center}
            \begin{tabular}{|l|c|c|}
            \hline
            Test name & Status & Asserts \\  \hline \hline
        )==";

        std::string table_footer = R"==(
            \hline
            \end{tabular}\end{center}
        )==";

        output << table_header;

        u32 table_cnt = 0;
        for (std::string key : strings) {
            u32 cnt     = assert_test_count[key];
            u32 cnt_suc = assert_test_count_success[key];

            if (table_cnt > 50) {
                output << table_footer << "\n\n" << table_header;
                table_cnt = 0;
            }

            output << R"(\verb|)" << shambase::trunc_str_start(key, 64) << "| & ";

            if (cnt == cnt_suc) {
                output << R"(\OK & )";
            } else {
                output << R"(\FAIL & )";
            }

            output << shambase::format(R"({}/{} \\)", cnt_suc, cnt);
            output << "\n";

            table_cnt++;
        }

        output << table_footer;

        output << "\n";

        for (TestResult &res : results) {
            if (res.asserts.get_assert_success_count() != res.asserts.get_assert_count()) {
                output << "Test : " << res.name << "\n\n";

                for (unsigned int j = 0; j < res.asserts.asserts.size(); j++) {

                    output << shambase::format_printf(
                        "     - Assert [%d/%zu] \n\n", j + 1, res.asserts.asserts.size());

                    output << R"(\verb|)" << res.asserts.asserts[j].name << "| : ";

                    if (res.asserts.asserts[j].value) {
                        output << R"(\OK)";
                    } else {
                        output << R"(\FAIL)";
                    }

                    if ((!res.asserts.asserts[j].value)
                        && !res.asserts.asserts[j].comment.empty()) {
                        output << "\n\n $\\rightarrow$ failed assert logs :\n\n";
                        output << R"(\begin{verbatim})" << res.asserts.asserts[j].comment
                               << R"(\end{verbatim})";
                        output << "\n";
                    }

                    output << "\n\n";
                }
            }
        }
    }

    /// Add tex outputs from the tests to the report
    void add_tex_output_section(std::stringstream &output, std::vector<TestResult> &results) {

        output << R"(\section{Tex report})";

        for (TestResult &res : results) {
            std::string &tex_out = res.tex_output;

            if (!tex_out.empty()) {

                output << R"==(
                \cprotect\subsection{ \verb+)=="
                              + res.name + R"==(+}
                )==";

                output << tex_out;
            }
        }
    }

    /// Document details about the SYCL queues status in the report
    void add_queue_section(std::stringstream &output, sycl::queue &q) {

        sycl::device d = q.get_device();
        output << "device name : " << d.get_info<sycl::info::device::name>() << "\n";
        output << "platform name : " << d.get_platform().get_info<sycl::info::platform::name>()
               << "\n";
        output << "device property : \n";
        output << "global_mem_size : "
               << shambase::readable_sizeof(d.get_info<sycl::info::device::global_mem_size>())
               << "\n";
        output << "global_mem_cache_size : "
               << shambase::readable_sizeof(d.get_info<sycl::info::device::global_mem_cache_size>())
               << "\n";
        output << "global_mem_cache_line_size : "
               << shambase::readable_sizeof(
                      d.get_info<sycl::info::device::global_mem_cache_line_size>())
               << "\n";
        output << "local_mem_size : "
               << shambase::readable_sizeof(d.get_info<sycl::info::device::local_mem_size>())
               << "\n";
        output << "is_endian_little : " << d.get_info<sycl::info::device::is_endian_little>()
               << "\n";
    }

    /// Document details about the Shamrock status in the report
    void add_config_section(std::stringstream &output) {

        std::string cxxflags = compile_arg;
        shambase::replace_all(cxxflags, " ", "\n");

        output << R"(\section{Shamrock config})";
        output << "\n\n";
        output << R"(\subsection{Git info})"
               << "\n";
        output << R"(\begin{verbatim})"
               << "\n";
        output << git_info_str;
        output << R"(\end{verbatim})"
               << "\n";
        output << R"(\subsection{c++ flags})"
               << "\n";
        output << R"(\begin{verbatim})"
               << "\n";
        output << cxxflags;
        output << R"(\end{verbatim})"
               << "\n";

        output << R"(\subsection{MPI status})"
               << "\n";
        output << R"(\begin{verbatim})"
               << "\n";
        output << "world size = " << shamcomm::world_size();
        output << R"(\end{verbatim})"
               << "\n";

        output << R"(\subsection{NodeInstance Status})"
               << "\n";
        output << "compute queue : "
               << "\n";
        output << R"(\begin{verbatim})"
               << "\n";
        add_queue_section(output, shamsys::instance::get_compute_queue());
        output << R"(\end{verbatim})"
               << "\n";
        output << "alt queue : "
               << "\n";
        output << R"(\begin{verbatim})"
               << "\n";
        add_queue_section(output, shamsys::instance::get_alt_queue());
        output << R"(\end{verbatim})"
               << "\n";

        output << "\n\n";
    }

    /// Make the tex report
    std::string make_test_report_tex(std::vector<TestResult> &results, bool mark_fail) {

        std::stringstream output;

        output << tex_template;

        output << "Global status : ";

        if (mark_fail) {
            output << R"(\FAIL)";
        } else {
            output << R"(\OK)";
        }

        output << "\n\n";
        output << R"(\tableofcontents)";
        output << "\n\n";

        add_config_section(output);

        add_unittest_section(output, results);

        add_tex_output_section(output, results);

        output << tex_template_end;

        return output.str();
    }
} // namespace shamtest::details
