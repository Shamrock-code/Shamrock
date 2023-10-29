// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright(C) 2021-2023 Timothée David--Cléris <timothee.david--cleris@ens-lyon.fr>
// Licensed under CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

#pragma once

/**
 * @file time.hpp
 * @author Timothée David--Cléris (timothee.david--cleris@ens-lyon.fr)
 * @brief 
 * 
 */
 
#include "shambase/string.hpp"
#include "shambackends/typeAliasVec.hpp"
#include "shamsys/legacy/log.hpp"

#include <plf_nanotimer.h>

namespace shambase {


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


    /*
    class Timer {
    public:
        std::chrono::steady_clock::time_point t_start, t_end;
        f64 nanosec;

        Timer(){};

        inline void start() { t_start = std::chrono::steady_clock::now(); }

        inline void end() {
            t_end = std::chrono::steady_clock::now();
            nanosec   = f64(std::chrono::duration_cast<std::chrono::nanoseconds>(t_end - t_start).count());
        }

        inline std::string get_time_str() {
            return nanosec_to_time_str(nanosec);
        }

        inline f64 elasped_sec(){
            return f64(nanosec) * 1e-9;
        }

    };
    */

    class Timer {
    public:
        plf::nanotimer timer;
        f64 nanosec;

        Timer(){};

        inline void start() { timer.start(); }

        inline void end() {
            nanosec   = timer.get_elapsed_ns();
        }

        inline std::string get_time_str()  const  {
            return nanosec_to_time_str(nanosec);
        }

        [[nodiscard]] inline f64 elasped_sec() const {
            return f64(nanosec) * 1e-9;
        }

    };

    class FunctionTimer{
        f64 acc = 0;
        u32 run_count = 0;

        public:

        template<class Func>
        void time_func(Func && f){
            Timer t;
            t.start();
            f();
            t.end();
            acc += t.elasped_sec();
            run_count += 1;
        }

        inline f64 func_time_sec(){
            return acc / run_count;
        }
    };



    template<class Func>
    inline f64 timeit(Func && f, u32 relaunch = 1){

        FunctionTimer t;

        for (u32 i = 0; i < relaunch; i++) {
            t.time_func([&](){f();});
        }

        return t.func_time_sec();
    }

    struct BenchmarkResult{
        std::vector<f64> counts;
        std::vector<f64> times;
    };

    inline BenchmarkResult benchmark_pow_len(std::function<f64(u32)> func, u32 start, u32 end, f64 pow_exp){
        BenchmarkResult res;
        for(f64 i = start; i < end; i*=pow_exp){
            logger::debug_ln("benchmark_pow_len", "N =",i);
            res.counts.push_back(i);
            res.times.push_back(func(u32(i)));
        }

        return res;
    }

}