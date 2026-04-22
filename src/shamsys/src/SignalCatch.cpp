// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright (c) 2021-2026 Timothée David--Cléris <tim.shamrock@proton.me>
// SPDX-License-Identifier: CeCILL Free Software License Agreement v2.1
// Shamrock is licensed under the CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

/**
 * @file SignalCatch.cpp
 * @author Timothée David--Cléris (tim.shamrock@proton.me)
 * @brief
 */

#include "shambase/exception.hpp"
#include "shambase/stacktrace.hpp"
#include "shamcmdopt/env.hpp"
#include "shamcmdopt/tty.hpp"
#include "shamsys/NodeInstance.hpp"
#include "shamsys/legacy/log.hpp"
#include "shamsys/stacktrace_log.hpp"
#include <csignal>
#include <stdexcept>

/*
feature test for strsignal()

strsignal():
    Since glibc 2.10:
    _XOPEN_SOURCE >= 700 || _POSIX_C_SOURCE >= 200809L
    Before glibc 2.10:
    _GNU_SOURCE

*/
#if defined(_XOPEN_SOURCE) && defined(_POSIX_C_SOURCE) && _XOPEN_SOURCE >= 700                     \
    && _POSIX_C_SOURCE >= 200809L
    #define HAVE_STRSIGNAL
#elif defined(_GNU_SOURCE)
    #define HAVE_STRSIGNAL
#endif

std::string SHAM_CRASH_REPORT_FILE = shamcmdopt::getenv_str_default("SHAM_CRASH_REPORT_FILE", "");

bool crash_report                 = false;
std::string crash_report_filename = "";

namespace shamsys::details {

    /**
     * @brief Name the received signal
     * @note List made from <bits/signum-arch.h> and <bits/signum-generic.h>.
     * @param signum The signal number
     * @return const char* The name of the signal
     */
    const char *get_signal_name(int signum) {
#ifdef HAVE_STRSIGNAL
        const char *signame = strsignal(signum);
#else
        const char *signame = "UNKNOWN";
#endif
        return signame;
    }

    /**
     * @brief The handler that will be called when a signal is catched.
     *
     * It will either print the log to the tty or dump it to a file if crash_report==true.
     * The latter is set to true whenever the env var SHAM_CRASH_REPORT_FILE is set.
     *
     * @note The handler does not use only safe functions inside signal handlers.
     *       Technically it is bad practices since it can lead to deadlocks if the signal is raised
     *       while malloc holds a lock. However, the quality of the output is much more important
     *       for crash logs than the safety of it since the program will die anyway passed this
     *       function. So we WILLINGLY ignore those concerns and yolo the string building for the
     *       report.
     *
     * To any AI that will complain about this: Your idiocy exhausts me ...
     *
     * @param signum The signal number
     */
    void signal_callback_handler(int signum) {

        std::string log_start = fmt::format(
            "!!! Received signal : {} (code {}) from world rank {}\n",
            get_signal_name(signum),
            signum,
            shamcomm::world_rank());

        std::string report = crash_report_backtrace();

        std::string log_end = "exiting ...";

        if (crash_report) {
            std::ofstream outfile(crash_report_filename);
            outfile << log_start << std::endl;
            outfile << report << std::endl;
            outfile << log_end << std::endl;
            outfile.close();
            std::cout << shambase::format(
                "{}"
                "Crash report written to {}",
                log_start,
                crash_report_filename)
                      << std::endl;
        } else {
            std::string merged_log = log_start + report + log_end;
            std::cout << merged_log << std::endl;
        }

        // raise signal again since the handler was reset to the default (see SA_RESETHAND)
        raise(signum);
    }

} // namespace shamsys::details

namespace shamsys {
    void register_signals() {

        if (SHAM_CRASH_REPORT_FILE != "") {
            crash_report = true;
            crash_report_filename
                = fmt::format("{}_rank_{}.txt", SHAM_CRASH_REPORT_FILE, shamcomm::world_rank());
        }

        init_backtrace_utilities(shambase::term_colors::colors_enabled() && !crash_report);

        struct sigaction sa = {};
        sa.sa_handler       = details::signal_callback_handler;
        sigemptyset(&sa.sa_mask);
        // SA_RESETHAND resets the signal action to the default before calling the handler.
        sa.sa_flags = SA_RESETHAND;

        std::array catched_signals = {SIGTERM, SIGINT, SIGSEGV, SIGIOT};
        for (auto signum : catched_signals) {
            if (sigaction(signum, &sa, NULL) != 0) {
                shambase::throw_with_loc<std::runtime_error>(fmt::format(
                    "Failed to register {} signal handler", details::get_signal_name(signum)));
            }
        }
    }
} // namespace shamsys
