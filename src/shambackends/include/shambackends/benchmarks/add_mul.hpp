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
 * @file add_mul.hpp
 * @author Timothée David--Cléris (timothee.david--cleris@ens-lyon.fr)
 * @brief
 */

#include "shambase/assert.hpp"
#include "shambase/time.hpp"
#include "shambackends/DeviceBuffer.hpp"
#include "shambackends/DeviceScheduler.hpp"
#include "shambackends/math.hpp"

namespace sham::benchmarks {

    template<class T>
    inline void add_mul(
        u32 i, int nrotation, T cs, T sn, T *__restrict in1, T *__restrict in2, T *__restrict out) {

        T x = in1[i];
        T y = in2[i];
        for (int i = 0; i < nrotation; i++) {
            T xx = cs * x - sn * y;
            T yy = cs * y + sn * x;
            x    = xx;
            y    = yy;
        }
        out[i] = dot(x, y);
    }

    struct add_mul_result {
        std::string func_name;
        f64 milliseconds;
        f64 flops;
    };

    // From https://www.bealto.com/gpu-benchmarks_flops.html
    template<class T>
    inline add_mul_result add_mul_bench(
        DeviceScheduler_ptr sched,
        int N,
        T init_x,
        T init_y,
        T cs,
        T sn,
        int nrotation,
        int float_count) {

        sham::DeviceQueue &q = sched->get_queue();

        sham::DeviceBuffer<T> x   = {size_t(N), sched};
        sham::DeviceBuffer<T> y   = {size_t(N), sched};
        sham::DeviceBuffer<T> out = {size_t(N), sched};

        x.fill(init_x);
        y.fill(init_y);

        sham::EventList depends_list;

        auto x_ptr   = x.get_write_access(depends_list);
        auto y_ptr   = y.get_write_access(depends_list);
        auto out_ptr = out.get_write_access(depends_list);

        depends_list.wait();

        sham::EventList empty_list{};

        shambase::Timer t;
        t.start();
        auto e = q.submit(empty_list, [&](sycl::handler &cgh) {
            cgh.parallel_for(sycl::range<1>{size_t(N)}, [=](sycl::item<1> item) {
                add_mul(item.get_linear_id(), nrotation, cs, sn, x_ptr, y_ptr, out_ptr);
            });
        });
        e.wait();
        t.end();

        x.complete_event_state(sycl::event{});
        y.complete_event_state(sycl::event{});
        out.complete_event_state(sycl::event{});

        double milliseconds = t.elasped_sec() * 1e3;

        int flop_per_thread = nrotation * 6;
        double flop_count   = double(N) * flop_per_thread;
        double flops        = flop_count / (milliseconds / 1e3);

        return {SourceLocation{}.loc.function_name(), milliseconds, flops};
    }

} // namespace sham::benchmarks
