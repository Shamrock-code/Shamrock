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
 * @file bufToVtkBuf.hpp
 * @author Timothée David--Cléris (tim.shamrock@proton.me)
 * @brief
 *
 */

#include "shambase/endian.hpp"
#include "shambackends/sycl.hpp"
#include "shambackends/typeAliasVec.hpp"
#include "shambackends/vec.hpp"

namespace shamrock::details {

    template<class RT, class T>
    RT to_vtk_type(T a) {
        return shambase::get_endian_swap<RT>(a);
    }

    template<class RT, class T>
    inline sycl::buffer<RT> to_vtk_buf_type(sycl::queue &q, sycl::buffer<T> &buf_in, u64 len) {
        sycl::buffer<RT> ret(len);

        q.submit([=, &buf_in, &ret](sycl::handler &cgh) {
            sycl::accessor acc_in{buf_in, cgh, sycl::read_only};
            sycl::accessor acc_out{ret, cgh, sycl::write_only, sycl::no_init};
            cgh.parallel_for(sycl::range<1>{len}, [=](sycl::item<1> id) {
                acc_out[id] = to_vtk_type<RT>(acc_in[id]);
            });
        });

        return ret;
    }

    template<class RT, class T, int n>
    inline sycl::buffer<RT> to_vtk_buf_type(
        sycl::queue &q, sycl::buffer<sycl::vec<T, n>> &buf_in, u64 len) {
        sycl::buffer<RT> ret(len * n);

        q.submit([=, &buf_in, &ret](sycl::handler &cgh) {
            sycl::accessor acc_in{buf_in, cgh, sycl::read_only};
            sycl::accessor acc_out{ret, cgh, sycl::write_only, sycl::no_init};

            cgh.parallel_for(sycl::range<1>{len}, [=](sycl::item<1> id) {
                u32 idx = id.get_linear_id() * n;

                if constexpr (n == 2) {
                    acc_out[idx]     = to_vtk_type<RT>(acc_in[id].x());
                    acc_out[idx + 1] = to_vtk_type<RT>(acc_in[id].y());
                }

                if constexpr (n == 3) {
                    acc_out[idx]     = to_vtk_type<RT>(acc_in[id].x());
                    acc_out[idx + 1] = to_vtk_type<RT>(acc_in[id].y());
                    acc_out[idx + 2] = to_vtk_type<RT>(acc_in[id].z());
                }

                if constexpr (n == 4) {
                    acc_out[idx]     = to_vtk_type<RT>(acc_in[id].x());
                    acc_out[idx + 1] = to_vtk_type<RT>(acc_in[id].y());
                    acc_out[idx + 2] = to_vtk_type<RT>(acc_in[id].z());
                    acc_out[idx + 3] = to_vtk_type<RT>(acc_in[id].w());
                }

                if constexpr (n == 8) {
                    acc_out[idx]     = to_vtk_type<RT>(acc_in[id].s0());
                    acc_out[idx + 1] = to_vtk_type<RT>(acc_in[id].s1());
                    acc_out[idx + 2] = to_vtk_type<RT>(acc_in[id].s2());
                    acc_out[idx + 3] = to_vtk_type<RT>(acc_in[id].s3());
                    acc_out[idx + 4] = to_vtk_type<RT>(acc_in[id].s4());
                    acc_out[idx + 5] = to_vtk_type<RT>(acc_in[id].s5());
                    acc_out[idx + 6] = to_vtk_type<RT>(acc_in[id].s6());
                    acc_out[idx + 7] = to_vtk_type<RT>(acc_in[id].s7());
                }

                if constexpr (n == 16) {
                    acc_out[idx]      = to_vtk_type<RT>(acc_in[id].s0());
                    acc_out[idx + 1]  = to_vtk_type<RT>(acc_in[id].s1());
                    acc_out[idx + 2]  = to_vtk_type<RT>(acc_in[id].s2());
                    acc_out[idx + 3]  = to_vtk_type<RT>(acc_in[id].s3());
                    acc_out[idx + 4]  = to_vtk_type<RT>(acc_in[id].s4());
                    acc_out[idx + 5]  = to_vtk_type<RT>(acc_in[id].s5());
                    acc_out[idx + 6]  = to_vtk_type<RT>(acc_in[id].s6());
                    acc_out[idx + 7]  = to_vtk_type<RT>(acc_in[id].s7());
                    acc_out[idx + 8]  = to_vtk_type<RT>(acc_in[id].s8());
                    acc_out[idx + 9]  = to_vtk_type<RT>(acc_in[id].s9());
                    acc_out[idx + 10] = to_vtk_type<RT>(acc_in[id].sA());
                    acc_out[idx + 11] = to_vtk_type<RT>(acc_in[id].sB());
                    acc_out[idx + 12] = to_vtk_type<RT>(acc_in[id].sC());
                    acc_out[idx + 13] = to_vtk_type<RT>(acc_in[id].sD());
                    acc_out[idx + 14] = to_vtk_type<RT>(acc_in[id].sE());
                    acc_out[idx + 15] = to_vtk_type<RT>(acc_in[id].sF());
                }
            });
        });

        return ret;
    }

} // namespace shamrock::details
