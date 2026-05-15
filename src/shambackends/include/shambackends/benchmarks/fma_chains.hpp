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
 * @file fma_chains.hpp
 * @author Timothée David--Cléris (tim.shamrock@proton.me)
 * @brief Port of Argonne National Laboratory's FMA chains benchmark flops.cpp
 */

#include "shambase/assert.hpp"
#include "shambase/time.hpp"
#include "shambackends/DeviceBuffer.hpp"
#include "shambackends/DeviceScheduler.hpp"
#include "shambackends/math.hpp"

namespace sham::benchmarks {

    /**
     * @brief Kernel for the fma_chains benchmark.
     *
     * Saturates the FPU to hide memory latency.
     * Since we know that there are 16 * 2 flops per iteration,
     * this kernel can be used to compute the achieved flops.
     *
     * @tparam T value type of the input and output vectors
     * @param i index of the element to process
     * @param nrotation number of FMA-chain rotations to apply
     * @param y0 initial value of the second input vector
     * @param in input vector
     * @param out output vector
     */
    template<class T>
    inline void fma_chains(u32 i, int nrotation, T y0, T *__restrict in, T *__restrict out) {
#define MAD_4(x, y)                                                                                \
    x = y * x + y;                                                                                 \
    y = x * y + x;                                                                                 \
    x = y * x + y;                                                                                 \
    y = x * y + x;
#define MAD_16(x, y)                                                                               \
    MAD_4(x, y);                                                                                   \
    MAD_4(x, y);                                                                                   \
    MAD_4(x, y);                                                                                   \
    MAD_4(x, y);

        T x = in[i];
        T y = y0;
        for (int j = 0; j < nrotation; j++) {
            MAD_16(x, y);
        }
        out[i] = y;

#undef MAD_4
#undef MAD_16
    }

    /// Structure containing the results of an fma_chains benchmark
    struct fma_chains_result {
        std::string func_name; ///< Name of the function
        f64 seconds;           ///< Computation time in seconds
        f64 flops;             ///< Flops per second
        u32 nrotations;        ///< Number of rotation performed
    };

    /**
     * @brief Run the fma_chains benchmark.
     *
     * Based on Argonne's Aurora node performance overview:
     * https://docs.alcf.anl.gov/aurora/node-performance-overview/node-performance-overview/
     *
     * @tparam T value type used in the benchmark
     * @param sched scheduler for the target device
     * @param N number of elements (independent FMA chains) to process
     * @param time_threshold minimum wall-clock time to run the benchmark in seconds
     * @return benchmark results as an fma_chains_result
     */
    template<class T>
    inline fma_chains_result fma_chains_bench(
        DeviceScheduler_ptr sched, int N, f64 time_threshold) {

        sham::DeviceQueue &q = sched->get_queue();

        sham::DeviceBuffer<T> x = {size_t(N), sched};
        sham::DeviceBuffer<T> y = {size_t(N), sched};

        const T x0 = T{1.1};
        const T y0 = -x0;

        x.fill(x0);
        y.fill(y0);

        sham::EventList depends_list;

        auto x_ptr = x.get_write_access(depends_list);
        auto y_ptr = y.get_write_access(depends_list);

        depends_list.wait();

        u32 nrotation = 8;
        double sec    = 0;

        auto run_bench = [&q, &N, &x_ptr, &y_ptr, y0](u32 nrotation) -> f64 {
            sham::EventList empty_list{};

            shambase::Timer t;
            t.start();
            auto e = q.submit(empty_list, [=](sycl::handler &cgh) {
                cgh.parallel_for(sycl::range<1>{size_t(N)}, [=](sycl::item<1> item) {
                    fma_chains(item.get_linear_id(), nrotation, y0, x_ptr, y_ptr);
                });
            });
            e.wait();
            t.stop();

            return t.elapsed_sec();
        };

        // warmup kernel
        run_bench(4);

        double ref = run_bench(0);

        for (;;) {

            sec = run_bench(nrotation);

            if (sec >= time_threshold || nrotation >= 256 * 256 * 4) {
                break;
            }

            nrotation *= 2;
        }

        x.complete_event_state(sycl::event{});
        y.complete_event_state(sycl::event{});

        sec -= ref;

        u64 flop_per_thread = u64(nrotation) * 2_u64 * 16_u64;
        double flop_count   = double(N) * flop_per_thread;
        double flops        = flop_count / (sec);

        return {SourceLocation{}.loc.function_name(), sec, flops, nrotation};
    }

} // namespace sham::benchmarks
