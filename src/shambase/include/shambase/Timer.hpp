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
 * @file time.hpp
 * @author Timothée David--Cléris (tim.shamrock@proton.me)
 * @brief
 *
 */

#include "shambase/aliases_float.hpp"
#include "shambase/format_time.hpp"

#ifndef __MACH__
    #ifndef USE_PLF_TIMER
        #define USE_PLF_TIMER
    #endif
#endif

#if defined(USE_PLF_TIMER)
    #include <plf_nanotimer.h>
#else
    #include <chrono>
#endif

namespace shambase {
    /**
     * @brief Class Timer measures the time elapsed since the timer was started.
     */
    class Timer {
        public:
#if defined(USE_PLF_TIMER)
        plf::nanotimer timer; ///< Internal timer
#else
        std::chrono::steady_clock::time_point t_start; ///< Internal timer
#endif
        f64 nanosec; ///< Time in nanoseconds

        /// Constructor, init nanosec to 0
        Timer() : nanosec(0.0) {}

        /**
         * @brief Starts the timer.
         */
        inline void start() {
#if defined(USE_PLF_TIMER)
            timer.start();
#else
            t_start = std::chrono::steady_clock::now();
#endif
        }

        /**
         * @brief Stops the timer and stores the elapsed time in nanoseconds.
         *
         * If the timer has already been stopped, calling this again updates `nanosec` to
         * the new delta since `start()`.
         */
        inline void stop() {
#if defined(USE_PLF_TIMER)
            nanosec = f64(timer.get_elapsed_ns());
#else
            auto t_end = std::chrono::steady_clock::now();
            nanosec    = f64(
                std::chrono::duration_cast<std::chrono::nanoseconds>(t_end - t_start).count());
#endif
        }

        /**
         * @brief Converts the stored nanosecond time to a string representation.
         * @return std::string A string representation of the elapsed time.
         */
        inline std::string get_time_str() const { return nanosec_to_time_str(nanosec); }

        /**
         * @brief Converts the stored nanosecond time to a floating point representation in seconds.
         * @return f64 The elapsed time in seconds.
         */
        [[nodiscard]] inline f64 elapsed_sec() const { return nanosec * 1e-9; }
    };
} // namespace shambase
