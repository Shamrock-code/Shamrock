// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright (c) 2021-2026 Timothée David--Cléris <tim.shamrock@proton.me>
// SPDX-License-Identifier: CeCILL Free Software License Agreement v2.1
// Shamrock is licensed under the CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

/**
 * @file timeout_detect.cpp
 * @author Timothée David--Cléris (tim.shamrock@proton.me)
 * @brief Watchdog thread that aborts the run if it is not woken up regularly
 */

#include "shamcomm/timeout_detect.hpp"
#include "mpi.h"
#include "shamcomm/worldInfo.hpp"
#include <atomic>
#include <chrono>
#include <csignal>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <pthread.h>
#include <thread>

namespace {

    // Trivially destructible state only: the thread is detached and can outlive static destruction
    std::atomic<f64> timeout_seconds{std::numeric_limits<f64>::infinity()};
    std::atomic<bool> wakeup_flag{false};
    std::atomic<bool> started{false};
    pthread_t main_thread;

    constexpr auto check_period      = std::chrono::seconds(1);
    constexpr auto abort_grace_time  = std::chrono::seconds(30);
    constexpr auto abort_poll_period = std::chrono::milliseconds(100);
    constexpr int abort_error_code   = 30;

    /// True if something (typically shamsys::register_signals) installed a SIGABRT handler
    bool has_sigabrt_handler() {
        struct sigaction current = {};
        if (sigaction(SIGABRT, nullptr, &current) != 0) {
            return false;
        }
        if (current.sa_flags & SA_SIGINFO) {
            return current.sa_sigaction != nullptr;
        }
        return current.sa_handler != SIG_DFL && current.sa_handler != SIG_IGN;
    }

    [[noreturn]] void on_timeout(f64 elapsed_sec, f64 timeout_sec) {
        std::fprintf(
            stderr,
            "[rank %d] timeout detected: no wakeup for %.1f s (limit %.1f s), aborting\n",
            shamcomm::world_rank(),
            elapsed_sec,
            timeout_sec);

        if (has_sigabrt_handler()) {
            // The main thread is the one hanging: raise the signal there so that the crash
            // handler reports its backtrace, then it re-raises with the default action and dies.
            pthread_kill(main_thread, SIGABRT);

            // Safety net in case the handler itself is stuck
            for (auto waited = std::chrono::milliseconds(0); waited < abort_grace_time;
                 waited += abort_poll_period) {
                std::this_thread::sleep_for(abort_poll_period);
            }
        }

        MPI_Abort(MPI_COMM_WORLD, abort_error_code);
        // MPI_Abort should not return
        std::abort();
    }

    void timeout_thread_loop() {
        using clock = std::chrono::steady_clock;

        auto last_wakeup = clock::now();

        while (true) {
            std::this_thread::sleep_for(check_period);

            auto now = clock::now();

            if (wakeup_flag.exchange(false)) {
                last_wakeup = now;
                continue;
            }

            f64 elapsed = std::chrono::duration<f64>(now - last_wakeup).count();
            f64 timeout = timeout_seconds.load();

            if (elapsed > timeout) {
                on_timeout(elapsed, timeout);
            }
        }
    }

} // namespace

namespace shamcomm {

    void set_timeout(f64 seconds) { timeout_seconds.store(seconds); }

    void start_timeout_thread() {
        if (started.exchange(true)) {
            return;
        }

        main_thread = pthread_self();
        std::thread(timeout_thread_loop).detach();
    }

    void wakeup_timeout_thread() { wakeup_flag.store(true); }

} // namespace shamcomm
