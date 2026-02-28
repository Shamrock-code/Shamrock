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

namespace shamsys::details {

    /**
     * @brief Name the received signal
     * @note List made from <bits/signum-arch.h> and <bits/signum-generic.h>.
     * @param signum The signal number
     * @return const char* The name of the signal
     */
    const char *get_signal_name(int signum) {
        const char *signame = nullptr;
        switch (signum) {

        // ISO C99 signals.
        case SIGINT : signame = "SIGINT"; break;
        case SIGILL : signame = "SIGILL"; break;
        case SIGABRT: signame = "SIGABRT"; break;
        case SIGFPE : signame = "SIGFPE"; break;
        case SIGSEGV: signame = "SIGSEGV"; break;
        case SIGTERM: signame = "SIGTERM"; break;

        // Historical signals specified by POSIX.
        case SIGHUP : signame = "SIGHUP"; break;
        case SIGQUIT: signame = "SIGQUIT"; break;
        case SIGTRAP: signame = "SIGTRAP"; break;
        case SIGKILL: signame = "SIGKILL"; break;
        case SIGPIPE: signame = "SIGPIPE"; break;
        case SIGALRM: signame = "SIGALRM"; break;

        // Adjustments and additions to the signal number constants for  most Linux systems.
        case SIGSTKFLT: signame = "SIGSTKFLT"; break;
        case SIGPWR   : signame = "SIGPWR"; break;

        // Historical signals specified by POSIX.
        case SIGBUS: signame = "SIGBUS"; break;
        case SIGSYS: signame = "SIGSYS"; break;

        // New(er) POSIX signals (1003.1-2008, 1003.1-2013).
        case SIGURG   : signame = "SIGURG"; break;
        case SIGSTOP  : signame = "SIGSTOP"; break;
        case SIGTSTP  : signame = "SIGTSTP"; break;
        case SIGCONT  : signame = "SIGCONT"; break;
        case SIGCHLD  : signame = "SIGCHLD"; break;
        case SIGTTIN  : signame = "SIGTTIN"; break;
        case SIGTTOU  : signame = "SIGTTOU"; break;
        case SIGPOLL  : signame = "SIGPOLL"; break;
        case SIGXFSZ  : signame = "SIGXFSZ"; break;
        case SIGXCPU  : signame = "SIGXCPU"; break;
        case SIGVTALRM: signame = "SIGVTALRM"; break;
        case SIGPROF  : signame = "SIGPROF"; break;
        case SIGUSR1  : signame = "SIGUSR1"; break;
        case SIGUSR2  : signame = "SIGUSR2"; break;

        // Nonstandard signals found in all modern POSIX systems (including both BSD and Linux).
        case SIGWINCH: signame = "SIGWINCH"; break;

        default: signame = "UNKNOWN"; break;
        }
        return signame;
    }

    void signal_callback_handler(int signum) {

        const char *signame = get_signal_name(signum);

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

        if (sigaction(SIGTERM, &sa, NULL) != 0) {
            shambase::throw_with_loc<std::runtime_error>(
                "Failed to register SIGTERM signal handler");
        }
        if (sigaction(SIGINT, &sa, NULL) != 0) {
            shambase::throw_with_loc<std::runtime_error>(
                "Failed to register SIGINT signal handler");
        }
        if (sigaction(SIGSEGV, &sa, NULL) != 0) {
            shambase::throw_with_loc<std::runtime_error>(
                "Failed to register SIGSEGV signal handler");
        }
        if (sigaction(SIGIOT, &sa, NULL) != 0) {
            shambase::throw_with_loc<std::runtime_error>(
                "Failed to register SIGIOT signal handler");
        }
    }
} // namespace shamsys
