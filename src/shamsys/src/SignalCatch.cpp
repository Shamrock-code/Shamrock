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
#include "shamsys/NodeInstance.hpp"
#include "shamsys/legacy/log.hpp"
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

    void signal_callback_handler(int signum) {

        // ensure that we print in one block to avoid interleaving
        std::string log = fmt::format(
            "!!! Received signal : {} (code {}) from world rank {}\n"
            "Current stacktrace : \n"
            "{}\n"
            "exiting ...",
            get_signal_name(signum),
            signum,
            shamcomm::world_rank(),
            shambase::fmt_callstack());

        std::cout << log << std::endl;

        // raise signal again since the handler was reset to the default (see SA_RESETHAND)
        raise(signum);
    }
} // namespace shamsys::details

namespace shamsys {
    void register_signals() {
        struct sigaction sa = {};

        sa.sa_handler = details::signal_callback_handler;
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
