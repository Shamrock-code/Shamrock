// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright (c) 2021-2025 Timothée David--Cléris <tim.shamrock@proton.me>
// SPDX-License-Identifier: CeCILL Free Software License Agreement v2.1
// Shamrock is licensed under the CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

#pragma once

/**
 * @file sycl2020reduction.hpp
 * @author Timothée David--Cléris (tim.shamrock@proton.me)
 * @brief
 *
 */

#include "shamalgs/memory.hpp"
#include "shambackends/sycl.hpp"

namespace shamalgs::reduction::details {

    template<class T>
    struct SYCL2020 {
        static T sum(sycl::queue &q, sycl::buffer<T> &buf1, u32 start_id, u32 end_id);

        // static T min(sycl::queue &q, sycl::buffer<T> &buf1, u32 start_id, u32 end_id);

        // static T max(sycl::queue &q, sycl::buffer<T> &buf1, u32 start_id, u32 end_id);
    };

    template<class T, class Op>
    inline T reduce_sycl_2020(
        sycl::queue &q, sycl::buffer<T> &buf1, u32 start_id, u32 end_id, Op op) {

        u32 len = end_id - start_id;

        sycl::buffer<T> buf_int(len);
        shamalgs::memory::write_with_offset_into(q, buf_int, buf1, start_id, len);

        sycl::buffer<T> recov{1};

        q.submit([&](sycl::handler &cgh) {
            sycl::accessor global_mem{buf_int, cgh, sycl::read_only};

#ifdef SYCL_COMP_INTEL_LLVM
            auto reduc = sycl::reduction(recov, cgh, op);
#else
            sycl::accessor acc_rec{recov, cgh, sycl::write_only, sycl::no_init};
            auto reduc = sycl::reduction(acc_rec, op);
#endif

            cgh.parallel_for(sycl::range<1>{len}, reduc, [=](sycl::id<1> idx, auto &sum) {
                sum.combine(global_mem[idx]);
            });
        });

        T rec;
        {
            sycl::host_accessor acc{recov, sycl::read_only};
            rec = acc[0];
        }

        return rec;
    }

    template<class T>
    inline T SYCL2020<T>::sum(sycl::queue &q, sycl::buffer<T> &buf1, u32 start_id, u32 end_id) {
#ifdef SYCL_COMP_INTEL_LLVM
        return reduce_sycl_2020(q, buf1, start_id, end_id, sycl::plus<>{});
#endif

#ifdef SYCL_COMP_ACPP
        return reduce_sycl_2020(q, buf1, start_id, end_id, sycl::plus<T>{});
#endif
    }

} // namespace shamalgs::reduction::details
