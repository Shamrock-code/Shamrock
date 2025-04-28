// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright (c) 2021-2024 Timothée David--Cléris <tim.shamrock@proton.me>
// SPDX-License-Identifier: CeCILL Free Software License Agreement v2.1
// Shamrock is licensed under the CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

#pragma once

/**
 * @file saxpy.hpp
 * @author Timothée David--Cléris (timothee.david--cleris@ens-lyon.fr)
 * @brief
 */

#include "shambase/assert.hpp"
#include "shambase/time.hpp"
#include "shambackends/DeviceBuffer.hpp"
#include "shambackends/DeviceScheduler.hpp"
#include "shambackends/math.hpp"

namespace sham::benchmarks {

    /** @brief saxpy function for benchmarking.
     *
     * @param[in] i        Index to start the computation.
     * @param[in] n        Number of elements to process.
     * @param[in] a        Coefficient in the saxpy operation.
     * @param[in] x        Input array.
     * @param[inout] y     Output array.
     */
    template<class T>
    inline void saxpy(u32 i, int n, T a, T *__restrict x, T *__restrict y) {
        if (i < n)
            y[i] = a * x[i] + y[i];
    }

    /// Structure containing the results of a saxpy benchmark.
    struct saxpy_result {
        /// Name of the function.
        std::string func_name;
        /// Computation time in milliseconds.
        f64 milliseconds;
        /// Bandwidth in gibibytes per second.
        f64 bandwidth;
    };

    /**
     * @brief saxpy function for benchmarking.
     *
     * @param[in] sched         Device scheduler.
     * @param[in] N             Number of elements to process.
     * @param[in] init_x        Initial value for the input array.
     * @param[in] init_y        Initial value for the output array.
     * @param[in] a             Coefficient in the saxpy operation.
     * @param[in] load_size     Number of bytes processed per element.
     * @param[in] check_correctness Check if the result is correct.
     *
     *From https://developer.nvidia.com/blog/how-implement-performance-metrics-cuda-cc/
     *
     * @return saxpy_result containing the computation time in milliseconds,
     *         the bandwidth in gibibytes per second, and the name of the
     *         function.
     */
    template<class T>
    inline saxpy_result saxpy_bench(
        DeviceScheduler_ptr sched,
        int N,
        T init_x,
        T init_y,
        T a,
        int load_size,
        bool check_correctness) {

        sham::DeviceQueue &q = sched->get_queue();

        sham::DeviceBuffer<T> x = {size_t(N), sched};
        sham::DeviceBuffer<T> y = {size_t(N), sched};

        x.fill(init_x);
        y.fill(init_y);

        sham::EventList depends_list;

        auto x_ptr = x.get_write_access(depends_list);
        auto y_ptr = y.get_write_access(depends_list);

        depends_list.wait();

        sham::EventList empty_list{};

        shambase::Timer t;
        t.start();
        auto e = q.submit(empty_list, [&](sycl::handler &cgh) {
            cgh.parallel_for(sycl::range<1>{size_t(N)}, [=](sycl::item<1> item) {
                // printf("%d\n", item.get_linear_id());
                saxpy(item.get_linear_id(), N, a, x_ptr, y_ptr);
            });
        });
        e.wait();
        t.end();

        x.complete_event_state(sycl::event{});
        y.complete_event_state(sycl::event{});

        double milliseconds = t.elasped_sec() * 1e3;

        auto y_res = y.copy_to_stdvec();

        T expected = a * init_x + init_y;

        if (check_correctness) {
            T maxError = {};
            for (int i = 0; i < N; i++) {
                T delt = y_res[i] - expected;

                if constexpr (std::is_same_v<T, sycl::marray<float, 3>>) {
                    maxError[0] = sham::max(maxError[0], sham::abs(delt[0]));
                    maxError[1] = sham::max(maxError[1], sham::abs(delt[1]));
                    maxError[2] = sham::max(maxError[2], sham::abs(delt[2]));
                } else if constexpr (std::is_same_v<T, sycl::marray<float, 4>>) {
                    maxError[0] = sham::max(maxError[0], sham::abs(delt[0]));
                    maxError[1] = sham::max(maxError[1], sham::abs(delt[1]));
                    maxError[2] = sham::max(maxError[2], sham::abs(delt[2]));
                    maxError[3] = sham::max(maxError[3], sham::abs(delt[3]));
                } else {
                    maxError = sham::max(maxError, sham::abs(delt));
                }
            }

            SHAM_ASSERT(sham::equals(maxError, T{}));
        }

        return {
            SourceLocation{}.loc.function_name(),
            milliseconds,
            double(N) * load_size * 3 / milliseconds / 1e6};
    }

} // namespace sham::benchmarks
