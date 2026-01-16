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
#include "shambase/string.hpp"
#include <plf_nanotimer.h>
#include <functional>
#include <iostream>

namespace shambase {

    /**
     * @brief Convert nanoseconds to a human-readable string representation.
     *
     * @param nanosec The duration in nanoseconds.
     * @return std::string The duration in a human-readable format.
     */
    inline std::string nanosec_to_time_str(double nanosec) {
        double sec_int = nanosec;

        std::string unit = "ns";

        if (sec_int > 2000) {
            unit = "us";
            sec_int /= 1000;
        }

        if (sec_int > 2000) {
            unit = "ms";
            sec_int /= 1000;
        }

        if (sec_int > 2000) {
            unit = "s";
            sec_int /= 1000;
        }

        if (sec_int > 2000) {
            unit = "ks";
            sec_int /= 1000;
        }

        if (sec_int > 2000) {
            unit = "Ms";
            sec_int /= 1000;
        }

        if (sec_int > 2000) {
            unit = "Gs";
            sec_int /= 1000;
        }

        return shambase::format_printf("%4.2f", sec_int) + " " + unit;
    }

#ifdef __MACH__
    class Timer {
        public:
        std::chrono::steady_clock::time_point t_start, t_end;
        f64 nanosec;

        Timer() {};

        inline void start() { t_start = std::chrono::steady_clock::now(); }

        inline void end() {
            t_end   = std::chrono::steady_clock::now();
            nanosec = f64(
                std::chrono::duration_cast<std::chrono::nanoseconds>(t_end - t_start).count());
        }

        inline std::string get_time_str() { return nanosec_to_time_str(nanosec); }

        inline f64 elasped_sec() { return f64(nanosec) * 1e-9; }
    };
#else

    /**
     * @brief Class Timer measures the time elapsed since the timer was started.
     */
    class Timer {
        public:
        plf::nanotimer timer; ///< Internal timer

        f64 nanosec; ///< Time in nanosecond

        Timer() {};
        /**
         * @brief Starts the timer.
         */
        inline void start() { timer.start(); }

        /**
         * @brief Stops the timer and stores the elapsed time in nanoseconds.
         */
        inline void end() { nanosec = timer.get_elapsed_ns(); }

        /**
         * @brief Converts the stored nanosecond time to a string representation.
         * @return std::string A string representation of the elapsed time.
         */
        inline std::string get_time_str() const { return nanosec_to_time_str(nanosec); }

        /**
         * @brief Converts the stored nanosecond time to a floating point representation in seconds.
         * @return f64 The elapsed time in seconds.
         */
        [[nodiscard]] inline f64 elasped_sec() const { return f64(nanosec) * 1e-9; }
    };
#endif

    /**
     * @brief Class FunctionTimer measures the time it takes to execute a function.
     *
     * The class FunctionTimer is used to measure the time it takes to execute a function.
     * It does this by creating a Timer object and starting it before executing the function.
     * After the function has been executed, the timer is stopped and the elapsed time is added to
     * the total time. The total time is then divided by the number of times the function has been
     * executed to give the average time it takes to execute the function.
     */
    class FunctionTimer {
        /// The total time it takes to execute the function.
        f64 acc = 0;
        /// The number of times the function has been executed.
        u32 run_count = 0;

        public:
        /**
         * @brief Measures the time it takes to execute a function.
         *
         * @param f The function to be executed.
         */
        template<class Func>
        void time_func(Func &&f) {
            Timer t;
            t.start();
            f();
            t.end();
            acc += t.elasped_sec();
            run_count += 1;
        }

        /**
         * @brief Returns the average time it takes to execute the function.
         *
         * @return f64 The average time it takes to execute the function.
         */
        inline f64 func_time_sec() { return acc / run_count; }
    };

    /**
     * @brief Measures the average time it takes to execute a function.
     *
     * This function measures the average time it takes to execute a function by
     * executing it the specified number of times and dividing the total time by
     * the number of times the function was executed.
     *
     * @tparam Func The type of the function to be executed.
     * @param f The function to be executed.
     * @param relaunch The number of times the function is executed. Default is 1.
     * @return f64 The average time it takes to execute the function.
     */
    template<class Func>
    inline f64 timeit(Func &&f, u32 relaunch = 1) {

        FunctionTimer t;

        for (u32 i = 0; i < relaunch; i++) {
            t.time_func([&]() {
                f();
            });
        }

        return t.func_time_sec();
    }

    /**
     * @brief Measures the average time it takes to execute a function until a maximum duration is
     * reached.
     *
     * This function measures the average time it takes to execute a function until a maximum
     * duration is reached. It does this by executing the function in a loop until the maximum
     * duration is reached and then dividing the total time by the number of times the function was
     * executed.
     *
     * @tparam Func The type of the function to be executed.
     * @param f The function to be executed.
     * @param max_duration The maximum time the function is allowed to take. Default is 1 second.
     * @return f64 The average time it takes to execute the function.
     */
    template<class Func>
    inline f64 timeitfor(Func &&f, f64 max_duration = 1) {

        FunctionTimer t;
        Timer tdur;
        tdur.start();
        do {
            t.time_func([&]() {
                f();
            });
            tdur.end();
        } while (tdur.elasped_sec() < max_duration);

        return t.func_time_sec();
    }

    /**
     * @struct BenchmarkResult
     * @brief Structure to store the results of a benchmark.
     *
     * This structure stores the counts and times of a benchmark. The counts are the input values
     * used in the benchmark, and the times are the corresponding execution times.
     */
    struct BenchmarkResult {
        /**
         * @brief The counts of the benchmark.
         */
        std::vector<f64> counts;

        /**
         * @brief The times of the benchmark.
         */
        std::vector<f64> times;
    };

    /**
     * @brief Benchmark a function with input values following a power law.
     *
     * This function takes a function and a range of input values, and benchmarks it by executing
     * the function with each of the input values. The input values are generated by starting at the
     * `start` value and repeatedly multiplying by `pow_exp` until the `end` value is reached.
     *
     * @param func The function to be benchmarked.
     * @param start The starting value of the input.
     * @param end The ending value of the input.
     * @param pow_exp The power of the exponent to increase the input value.
     * @return BenchmarkResult A structure containing the counts and times of the benchmark.
     */
    inline BenchmarkResult benchmark_pow_len(
        std::function<f64(u32)> func, u32 start, u32 end, f64 pow_exp) {
        BenchmarkResult res;
        for (f64 i = start; i < end; i *= pow_exp) {
            res.counts.push_back(i);
            res.times.push_back(func(u32(i)));
        }

        return res;
    }

} // namespace shambase
