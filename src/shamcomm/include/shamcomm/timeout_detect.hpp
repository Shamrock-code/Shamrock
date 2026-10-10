// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright (c) 2021-2026 Timothée David--Cléris <tim.shamrock@proton.me>
// SPDX-License-Identifier: CeCILL Free Software License Agreement v2.1
// Shamrock is licensed under the CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

#pragma once

/**
 * @file timeout_detect.hpp
 * @author Timothée David--Cléris (tim.shamrock@proton.me)
 * @brief Watchdog thread that aborts the run if it is not woken up regularly
 *
 * A background thread checks every second that wakeup_timeout_thread() was called at least once
 * within the timeout. If not, the run is considered hung and is aborted. If the shamsys signal
 * handlers are installed (shamsys::register_signals), the main thread receives a SIGABRT so that
 * its backtrace is reported, otherwise MPI_Abort is called directly.
 */

#include "shambase/aliases_float.hpp"

namespace shamcomm {

    /**
     * @brief Set the timeout after which the run is aborted if the thread was not woken up
     *
     * The default is no timeout (infinite). Can be called before or after start_timeout_thread().
     *
     * @param seconds The timeout in seconds
     */
    void set_timeout(f64 seconds);

    /**
     * @brief Start the timeout detection thread (does nothing if it is already running)
     *
     * @warning Must be called from the main thread, which is the one receiving the abort signal
     * and whose backtrace is reported on timeout.
     */
    void start_timeout_thread();

    /**
     * @brief Reset the timer of the timeout detection thread
     *
     * Sets an atomic flag consumed by the timeout thread at its next check. Thread safe.
     */
    void wakeup_timeout_thread();

} // namespace shamcomm
