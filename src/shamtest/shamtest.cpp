// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright (c) 2021-2025 Timothée David--Cléris <tim.shamrock@proton.me>
// SPDX-License-Identifier: CeCILL Free Software License Agreement v2.1
// Shamrock is licensed under the CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

/**
 * @file shamtest.cpp
 * @author Timothée David--Cléris (tim.shamrock@proton.me)
 * @brief
 */

#include "shambase/bytestream.hpp"
#include "shambase/exception.hpp"
#include "shambase/stacktrace.hpp"
#include "shambase/string.hpp"
#include "shambase/term_colors.hpp"
#include "shambase/time.hpp"
#include "shambindings/pybindaliases.hpp"
#include "shambindings/start_python.hpp"
#include "shamcmdopt/ci_env.hpp"
#include "shamcmdopt/env.hpp"
#include "shamcomm/collectives.hpp"
#include "shamcomm/logs.hpp"
#include "shamcomm/worldInfo.hpp"
#include "shamrock/version.hpp"
#include "shamsys/MpiWrapper.hpp"
#include "shamsys/NodeInstance.hpp"
#include "shamsys/legacy/log.hpp"
#include "shamsys/legacy/sycl_handler.hpp"
#include "shamtest.hpp"
#include "shamtest/details/TestResult.hpp"
#include "shamtest/details/reporters/texTestReport.hpp"
#include <pybind11/embed.h>
#include <unordered_map>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <sstream>
#include <string>
#include <vector>

namespace shamtest {

    bool is_run_only         = false; ///< Is in run only mode
    bool is_full_output_mode = false; ///< Is in full output mode

    /**
     * @brief print the line in terminal when a test start
     *
     * @param test
     * @param test_num
     * @param test_count
     */
    void _start_test_print(details::Test &test, u32 test_num, u32 test_count) {

        std::string output;
        if (is_run_only) {
            output += ("- : ");
        } else {
            output += shambase::format("- [{}/{}] :", test_num + 1, test_count);
        }

        bool any_node_cnt = test.node_count == -1;
        if (any_node_cnt) {
            output += (" [any] ");
        } else {
            output += shambase::format(" [{:03}] ", test.node_count);
        }

        output += "\033[;34m" + test.name + "\033[0m\n";
        ON_RANK_0(printf("%s", output.c_str()));
    }

    /**
     * @brief print the line in terminal when a test ends
     *
     * @param res
     * @param timer
     */
    void _end_test_print(std::vector<details::TestResult> &rank_results, shambase::Timer &timer) {

        for (int rank = 0; rank < rank_results.size(); rank++) {
            auto &res = rank_results[rank];

            for (unsigned int j = 0; j < res.asserts.asserts.size(); j++) {

                if (is_full_output_mode || (!res.asserts.asserts[j].value)) {
                    printf("       Rank %3d [%d/%zu] : ", rank, j + 1, res.asserts.asserts.size());
                    printf("%-20s", res.asserts.asserts[j].name.c_str());

                    if (res.asserts.asserts[j].value) {
                        std::cout << "  (\033[;32mSucces\033[0m)\n";
                    } else {
                        std::cout << "  (\033[1;31m Fail \033[0m)\n";
                        if (!res.asserts.asserts[j].comment.empty()) {
                            std::cout << "----- logs : \n"
                                      << res.asserts.asserts[j].comment << "\n-----" << std::endl;
                        }
                    }
                }
            }
        }

        u32 assert_count = 0;
        u32 succes_cnt   = 0;
        for (int rank = 0; rank < rank_results.size(); rank++) {
            auto &res = rank_results[rank];
            for (unsigned int j = 0; j < res.asserts.asserts.size(); j++) {
                if (res.asserts.asserts[j].value) {
                    succes_cnt++;
                }
                assert_count++;
            }
        }

        if (succes_cnt == assert_count) {
            std::cout << "   -> Result : \033[;32mSucces\033[0m";
        } else {
            std::cout << "   -> Result : \033[1;31m Fail \033[0m";
        }

        std::string s_assert = shambase::format(" [{}/{}] ", succes_cnt, assert_count);
        printf("%-15s", s_assert.c_str());
        std::cout << " (" << timer.get_time_str() << ")" << std::endl;

        if (shamcmdopt::is_ci_github_actions()) {
            if (succes_cnt != assert_count) {
                logger::raw_ln(shambase::format("##[error]Test {} failed", rank_results[0].name));
            }
        }

        std::cout << std::endl;
    }

    /**
     * @brief check is a test is failed
     *
     * @param res
     * @return true
     * @return false
     */
    bool is_test_failed(details::TestResult &res) {
        for (unsigned int j = 0; j < res.asserts.asserts.size(); j++) {

            if (!res.asserts.asserts[j].value) {
                return true;
            }
        }

        return false;
    }

    /**
     * @brief Generate a failure log at the end of the tests
     *
     * @param res
     * @return std::string
     */
    std::string gen_fail_log(details::TestResult &res) {
        std::string out = "";

        std::string sep = "\n-------------------------------------\n";

        out += " - Test : \033[;34m" + res.name
               + "\033[0m world_rank = " + std::to_string(res.world_rank);
        out += "\n    Assertion list :\n";

        for (unsigned int j = 0; j < res.asserts.asserts.size(); j++) {

            out += shambase::format_printf("     - [%d/%zu] ", j + 1, res.asserts.asserts.size());
            out += shambase::format_printf("%-20s", res.asserts.asserts[j].name.c_str());

            if (res.asserts.asserts[j].value) {
                out += "  (\033[;32mSucces\033[0m)\n";
            } else {
                out += "  (\033[1;31m Fail \033[0m)\n";
            }

            if ((!res.asserts.asserts[j].value) && !res.asserts.asserts[j].comment.empty()) {
                out += " -> failed assert logs : " + sep + res.asserts.asserts[j].comment + sep;
            }
        }

        out += "\n\n";

        return out;
    }

    /**
     * @brief Print summary of the test run
     *
     * @param results
     */
    void _print_summary(std::vector<details::TestResult> &results) {
        if (shamcomm::world_rank() > 0) {
            return;
        }

        logger::print_faint_row();
        logger::print_faint_row();
        logger::print_faint_row();
        logger::raw_ln(
            shambase::term_colors::bold() + "Test Report :" + shambase::term_colors::reset());
        logger::raw_ln();

        u32 test_count  = results.size();
        u32 succ_count  = 0;
        u32 fail_count  = 0;
        std::string log = "";
        for (details::TestResult &res : results) {
            if (!is_test_failed(res)) {
                succ_count++;
            } else {
                fail_count++;
                log += gen_fail_log(res);
            }
        }

        std::cout << "Test suite status : ";
        if (fail_count == 0) {
            std::cout << "  (\033[;32mSucces\033[0m)";
            printf(" [%d/%d] \n", succ_count, test_count);
        } else {
            std::cout << "  (\033[1;31m Fail \033[0m)";
            printf(" [%d/%d] \n", succ_count, test_count);
            std::cout << "\nFailed tests : \n\n" << log;
        }

        logger::print_faint_row();
        logger::print_faint_row();
        logger::print_faint_row();
    }

    /// Gather test results from all MPI ranks
    std::vector<details::TestResult> gather_tests(
        std::vector<details::TestResult> rank_result, usize &gather_bytecount) {
        if (shamcomm::world_size() == 1) {
            return rank_result;
        }

        // generate payload
        std::basic_stringstream<byte> outrank;

        shambase::stream_write_vector(outrank, rank_result);

        std::basic_string<byte> gathered;
        shamcomm::gather_basic_str(outrank.str(), gathered);

        if (shamcomm::world_rank() != 0) {
            return {};
        }

        gather_bytecount = gathered.size();

        std::basic_stringstream<byte> reader(gathered);

        std::vector<details::TestResult> out;

        for (u32 i = 0; i < shamcomm::world_size(); i++) {
            shambase::stream_read_vector(reader, out);
        }

        return out;
    }

    /**
     * @brief print all the available tests
     *
     */
    void print_test_list() {

        if (shamcomm::world_rank() > 0) {
            return;
        }

        using namespace shamtest::details;

        auto print_list = [&](TestType t) {
            for (auto test : static_init_vec_tests) {
                if (test.type == t) {
                    if (test.node_count == -1) {
                        printf("- [any] %-15s\n", test.name.c_str());
                    } else {
                        printf("- [%03d] %-15s\n", test.node_count, test.name.c_str());
                    }
                }
            }
        };

        printf("--- Benchmark ---\n");

        print_list(Benchmark);

        printf("--- LongBenchmark ---\n");

        print_list(LongBenchmark);

        printf("--- ValidationTest  ---\n");

        print_list(ValidationTest);

        printf("--- LongValidationTest  ---\n");

        print_list(LongValidationTest);

        printf("--- Unittest  ---\n");

        print_list(Unittest);
    }

    /// Write the JSON report
    void write_json_report(std::vector<details::TestResult> &results, std::string outfile) {
        if (shamcomm::world_rank() > 0) {
            return;
        }

        std::stringstream rank_test_res_out;
        for (details::TestResult &res : results) {
            rank_test_res_out << res.serialize_json() << ",";
        }

        std::string out_res_string = rank_test_res_out.str();

        // generate json output and write it into the specified file

        if (out_res_string.back() == ',') {
            out_res_string = out_res_string.substr(0, out_res_string.size() - 1);
        }

        std::string s_out;

        s_out = "{\n";

        s_out += R"(    "commit_hash" : ")" + git_commit_hash + "\",\n";
        s_out += R"(    "world_size" : ")" + std::to_string(shamcomm::world_size()) + "\",\n";

#if defined(SYCL_COMP_INTEL_LLVM)
        s_out += R"(    "compiler" : "DPCPP",)"
                 "\n";
#elif defined(SYCL_COMP_HIPSYCL)
        s_out += R"(    "compiler" : "HipSYCL",)"
                 "\n";
#else
        s_out += R"(    "compiler" : "Unknown",)"
                 "\n";
#endif

        s_out += R"(    "comp_args" : ")" + compile_arg + "\",\n";

        s_out += R"(    "results" : )"
                 "[\n\n";
        s_out += shambase::increase_indent(out_res_string);
        s_out += "\n    ]\n}";

        // printf("%s\n",s_out.c_str());

        shambase::write_string_to_file(outfile, s_out);
    }

    /// Write the tex report
    void write_tex_report(std::vector<details::TestResult> &results, bool mark_fail) {
        if (shamcomm::world_rank() > 0) {
            return;
        }

        logger::raw("write report Tex : ");

        shambase::write_string_to_file(
            "tests/report.tex", details::make_test_report_tex(results, mark_fail));

        logger::raw_ln("Done (tests/report.tex)");
    }

    std::vector<u32> select_print_tests(TestConfig cfg) {

        bool run_unit_test           = cfg.run_unittest;
        bool run_validation_test     = cfg.run_validation;
        bool run_longvalidation_test = cfg.run_validation && cfg.run_long_tests;
        bool run_benchmark_test      = cfg.run_benchmark;
        bool run_longbenchmark_test  = cfg.run_benchmark && cfg.run_long_tests;

        auto can_run = [&](shamtest::details::Test &t) -> bool {
            bool any_node_cnt  = (t.node_count == -1);
            bool world_size_ok = t.node_count == shamcomm::world_size();

            bool can_run_type = false;

            auto test_type = t.type;
            can_run_type |= (run_unit_test && (Unittest == test_type));
            can_run_type |= (run_validation_test && (ValidationTest == test_type));
            can_run_type |= (run_longvalidation_test && (LongValidationTest == test_type));
            can_run_type |= (run_benchmark_test && (Benchmark == test_type));
            can_run_type |= (run_longbenchmark_test && (LongBenchmark == test_type));

            return can_run_type && (any_node_cnt || world_size_ok);
        };

        auto print_test = [&](shamtest::details::Test &t, bool enabled) {
            bool any_node_cnt = (t.node_count == -1);

            std::string output = "";

            if (enabled) {

                if (any_node_cnt) {
                    output += (" - [\033[;32many\033[0m] ");
                } else {
                    output += shambase::format(" - [\033[;32m{:03}\033[0m] ", t.node_count);
                }
                output += "\033[;32m" + t.name + "\033[0m\n";

            } else {
                if (any_node_cnt) {
                    output += (" - [\033[;31many\033[0m] ");
                } else {
                    output += shambase::format(" - [\033[;31m{:03}\033[0m] ", t.node_count);
                }
                output += "\033[;31m" + t.name + "\033[0m\n";
            }

            printf("%s", output.c_str());
        };

        using namespace shamtest::details;

        std::vector<u32> selected_tests;

        auto run_only_check = [&](std::string test_name) -> bool {
            if (cfg.run_only) {
                return *cfg.run_only == test_name;
            } else {
                return true;
            }
        };

        auto test_loop = [&](TestType t) {
            for (u32 i = 0; i < static_init_vec_tests.size(); i++) {
                if (static_init_vec_tests[i].type == t) {

                    bool run_test = can_run(static_init_vec_tests[i])
                                    && run_only_check(static_init_vec_tests[i].name);

                    ON_RANK_0(print_test(static_init_vec_tests[i], run_test));

                    if (run_test) {
                        selected_tests.push_back(i);
                    }
                }
            }
        };

        ON_RANK_0(printf("\n------------ Tests list --------------\n"));
        if (run_benchmark_test) {
            ON_RANK_0(printf("--- Benchmark ---\n"));
            test_loop(Benchmark);
        }

        if (run_benchmark_test) {
            ON_RANK_0(printf("--- LongBenchmark ---\n"));
            test_loop(LongBenchmark);
        }

        if (run_validation_test) {
            ON_RANK_0(printf("--- ValidationTest ---\n"));
            test_loop(ValidationTest);
        }

        if (run_longvalidation_test) {
            ON_RANK_0(printf("--- LongValidationTest ---\n"));
            test_loop(LongValidationTest);
        }

        if (run_unit_test) {
            ON_RANK_0(printf("--- Unittest  ---\n"));
            test_loop(Unittest);
        }
        ON_RANK_0(printf("--------------------------------------\n\n"));

        return selected_tests;
    }

    int run_all_tests(int argc, char *argv[], TestConfig cfg) {
        StackEntry stack{};

        is_full_output_mode = cfg.full_output;

        mpi::barrier(MPI_COMM_WORLD);
        std::vector<u32> selected_tests = select_print_tests(cfg);
        mpi::barrier(MPI_COMM_WORLD);

        u32 test_loc_cnt = 0;

        bool has_error = false;

        logger::info_ln("Test", "start python interpreter");
        py::initialize_interpreter();

        ON_RANK_0(shamcomm::logs::print_faint_row());
        shambindings::modify_py_sys_path(shamcomm::world_rank() == 0);
        ON_RANK_0(shamcomm::logs::print_faint_row());

        // import shamrock in pybind
        py::exec(R"(
            import shamrock
        )");

        std::filesystem::create_directories("tests/figures");

        using namespace shamtest::details;

        ON_RANK_0(logger::raw_ln("Running tests : "));
        ON_RANK_0(shamcomm::logs::print_faint_row());

        std::vector<TestResult> results;
        for (u32 i : selected_tests) {

            shamtest::details::Test &test = static_init_vec_tests[i];

            _start_test_print(test, test_loc_cnt, selected_tests.size());

            [[maybe_unused]] shambase::scoped_exception_gen_callback scoped_callback(nullptr);

            mpi::barrier(MPI_COMM_WORLD);
            shambase::Timer timer;
            timer.start();
            TestResult res = test.run();
            timer.end();
            mpi::barrier(MPI_COMM_WORLD);

            usize gather_bytecount           = 0;
            std::vector<TestResult> gathered = gather_tests({res}, gather_bytecount);
            if (shamcomm::world_rank() == 0) {
                logger::raw_ln("Test result gathered :", gather_bytecount, "bytes");
                _end_test_print(gathered, timer);
            }

            results.push_back(std::move(res));

            test_loc_cnt++;
        }

        logger::info_ln("Test", "close python interpreter");
        py::finalize_interpreter();

        usize gather_bytecount = 0;
        results                = gather_tests(std::move(results), gather_bytecount);

        if (shamcomm::world_rank() == 0) {
            logger::print_faint_row();
            logger::raw_ln("Test result gathered :", gather_bytecount, "bytes");
        }

        for (TestResult &res : results) {
            has_error = has_error || is_test_failed(res);
        }

        _print_summary(results);

        if (cfg.json_output) {
            write_json_report(results, *cfg.json_output);
        }

        write_tex_report(results, has_error);

        i32 errcode;
        if (has_error) {
            errcode = 255;
        } else {
            errcode = 0;
        }

        mpi::barrier(MPI_COMM_WORLD);

        if (shamcomm::world_rank() == 0) {
            logger::raw_ln("Tests done exiting ... exitcode =", errcode);
        }
        mpi::barrier(MPI_COMM_WORLD);
        shamsys::instance::close();

        return errcode;
    }

    void gen_test_list(std::string_view outfile) {
        // logger::raw_ln("Test list ...", outfile);

        using namespace details;

        std::array rank_list{1, 2, 3, 4};

        auto get_pref_type = [](TestType t) -> std::string {
            switch (t) {
            case Benchmark         : return "Benchmark";
            case LongBenchmark     : return "LongBenchmark";
            case ValidationTest    : return "ValidationTest";
            case LongValidationTest: return "LongValidationTest";
            case Unittest          : return "Unittest";
            }
        };

        auto get_arg = [](TestType t) -> std::string {
            switch (t) {
            case Benchmark         : return "--benchmark";
            case LongBenchmark     : return "--long-test --benchmark";
            case ValidationTest    : return "--validation";
            case LongValidationTest: return "--long-test --validation";
            case Unittest          : return "--unittest";
            }
        };

        auto get_test_name = [&](Test t, int ranks) -> std::string {
            std::string name = get_pref_type(t.type) + "/" + t.name
                               + shambase::format(
                                   "(ranks={})"
                                   //"{}"
                                   ,
                                   ranks);
            // shambase::replace_all(name, "/", "");
            return name;
        };

        std::ofstream filestream;
        filestream.open(std::string(outfile));

        std::vector<std::string> cmake_test_list;

        auto add_test = [&](Test t, int ranks) {
            std::string tname = get_test_name(t, ranks);
            cmake_test_list.push_back(tname);

            std::string ret = "add_test(\"";
            ret += tname;
            ret += "\"";
            if (ranks > 1) {
                ret += " mpirun -n " + std::to_string(ranks) + " ../shamrock_test --sycl-cfg 0:0";
            } else {
                ret += " ../shamrock_test --sycl-cfg 0:0";
            }
            ret += " --run-only \"" + std::string(t.name) + "\"";
            ret += " " + get_arg(t.type);
            ret += ")\n";
            filestream << ret;
        };

        for (const Test &t : static_init_vec_tests) {
            if (t.type == Benchmark || t.type == LongBenchmark)
                continue;
            if (t.node_count == -1) {
                for (int ncount : rank_list) {
                    add_test(t, ncount);
                }
            } else {
                add_test(t, t.node_count);
            }
        }

        filestream << "\n";

        auto REF_FILES_PATH = shamcmdopt::getenv_str("REF_FILES_PATH");

        if (REF_FILES_PATH) {
            filestream << "set_tests_properties(\n";
            for (auto tname : cmake_test_list) {
                filestream << "    \"" << tname << "\"\n";
            }
            filestream << "  PROPERTIES\n";
            filestream << "    ENVIRONMENT \"REF_FILES_PATH=" + *REF_FILES_PATH << "\"\n";
            filestream << ")\n";
        }

        filestream.close();
    }

} // namespace shamtest
