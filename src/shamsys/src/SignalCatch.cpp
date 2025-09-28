// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright (c) 2021-2025 Timothée David--Cléris <tim.shamrock@proton.me>
// SPDX-License-Identifier: CeCILL Free Software License Agreement v2.1
// Shamrock is licensed under the CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

/**
 * @file SignalCatch.cpp
 * @author Timothée David--Cléris (tim.shamrock@proton.me)
 * @brief
 */

#include "shambase/stacktrace.hpp"
#include "shamsys/NodeInstance.hpp"
#include "shamsys/legacy/log.hpp"
#include <csignal>

namespace shamsys::details {
    void signal_callback_handler(int signum) {

        const char *signame = nullptr;
        switch (signum) {
        case SIGTERM: signame = "SIGTERM"; break;
        case SIGINT : signame = "SIGINT"; break;
        case SIGSEGV: signame = "SIGSEGV"; break;
        default     : signame = "UNKNOWN"; break;
        }

        // ensure that we print in one block to avoid interleaving
        std::string log = fmt::format(
            "!!! Received signal : {} (code {}) from world rank {}\n"
            "Current stacktrace : \n"
            "{}\n"
            "exiting ...",
            signame,
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

        sigaction(SIGTERM, &sa, NULL);
        sigaction(SIGINT, &sa, NULL);
        sigaction(SIGSEGV, &sa, NULL);
    }
} // namespace shamsys
