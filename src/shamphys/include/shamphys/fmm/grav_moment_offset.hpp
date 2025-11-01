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
 * @file grav_moment_offset.hpp
 * @author Timothée David--Cléris (tim.shamrock@proton.me)
 * @brief
 */

#include "shambase/aliases_int.hpp"
#include "shambase/type_traits.hpp"
#include "shammath/symtensor_collections.hpp"
#include "shamphys/fmm/grav_moments.hpp"

namespace shamphys {

    // Offset the gravitational moment derivatives
    template<class T, u32 low_order, u32 high_order>
    inline shammath::SymTensorCollection<T, low_order, high_order> offset_dM_mat_delta(
        const shammath::SymTensorCollection<T, low_order, high_order> &dM,
        const sycl::vec<T, 3> &offset) {

        using namespace shammath;

        static constexpr T inv_factorial_0 = 1. / 1;
        static constexpr T inv_factorial_1 = 1. / 1;
        static constexpr T inv_factorial_2 = 1. / 2;
        static constexpr T inv_factorial_3 = 1. / 6;
        static constexpr T inv_factorial_4 = 1. / 24;
        static constexpr T inv_factorial_5 = 1. / 120;

        if constexpr (low_order == 0 && high_order == 5) {

            SymTensorCollection<T, 0, 5> h = SymTensorCollection<T, 0, 5>::from_vec(offset);

            auto &h_0 = h.t0;
            auto &h_1 = h.t1;
            auto &h_2 = h.t2;
            auto &h_3 = h.t3;
            auto &h_4 = h.t4;
            auto &h_5 = h.t5;

            auto &dM_0 = dM.t0;
            auto &dM_1 = dM.t1;
            auto &dM_2 = dM.t2;
            auto &dM_3 = dM.t3;
            auto &dM_4 = dM.t4;
            auto &dM_5 = dM.t5;

            shammath::SymTensorCollection<T, 0, 5> dM_ret;

            auto &dM_ret_0 = dM_ret.t0;
            auto &dM_ret_1 = dM_ret.t1;
            auto &dM_ret_2 = dM_ret.t2;
            auto &dM_ret_3 = dM_ret.t3;
            auto &dM_ret_4 = dM_ret.t4;
            auto &dM_ret_5 = dM_ret.t5;

            // dM_k = sum_{l=k}^p \frac{1}{(l-k)!} h^(l-k).dM_l

            dM_ret_0 = inv_factorial_0 * h_0 * dM_0 + inv_factorial_1 * h_1 * dM_1
                       + inv_factorial_2 * h_2 * dM_2 + inv_factorial_3 * h_3 * dM_3
                       + inv_factorial_4 * h_4 * dM_4 + inv_factorial_5 * h_5 * dM_5;

            dM_ret_1 = inv_factorial_0 * h_0 * dM_1 + inv_factorial_1 * h_1 * dM_2
                       + inv_factorial_2 * h_2 * dM_3 + inv_factorial_3 * h_3 * dM_4
                       + inv_factorial_4 * h_4 * dM_5;

            dM_ret_2 = inv_factorial_0 * h_0 * dM_2 + inv_factorial_1 * h_1 * dM_3
                       + inv_factorial_2 * h_2 * dM_4 + inv_factorial_3 * h_3 * dM_5;

            dM_ret_3 = inv_factorial_0 * h_0 * dM_3 + inv_factorial_1 * h_1 * dM_4
                       + inv_factorial_2 * h_2 * dM_5;

            dM_ret_4 = inv_factorial_0 * h_0 * dM_4 + inv_factorial_1 * h_1 * dM_5;
            dM_ret_5 = inv_factorial_0 * h_0 * dM_5;

            return dM_ret;
        } else if constexpr (low_order == 1 && high_order == 5) {

            SymTensorCollection<T, 0, 5> h = SymTensorCollection<T, 0, 5>::from_vec(offset);

            auto &h_0 = h.t0;
            auto &h_1 = h.t1;
            auto &h_2 = h.t2;
            auto &h_3 = h.t3;
            auto &h_4 = h.t4;
            auto &h_5 = h.t5;

            auto &dM_1 = dM.t1;
            auto &dM_2 = dM.t2;
            auto &dM_3 = dM.t3;
            auto &dM_4 = dM.t4;
            auto &dM_5 = dM.t5;

            shammath::SymTensorCollection<T, 1, 5> dM_ret;

            auto &dM_ret_1 = dM_ret.t1;
            auto &dM_ret_2 = dM_ret.t2;
            auto &dM_ret_3 = dM_ret.t3;
            auto &dM_ret_4 = dM_ret.t4;
            auto &dM_ret_5 = dM_ret.t5;

            // dM_k = sum_{l=k}^p \frac{1}{(l-k)!} h^(l-k).dM_l

            dM_ret_1 = inv_factorial_0 * h_0 * dM_1 + inv_factorial_1 * h_1 * dM_2
                       + inv_factorial_2 * h_2 * dM_3 + inv_factorial_3 * h_3 * dM_4
                       + inv_factorial_4 * h_4 * dM_5;

            dM_ret_2 = inv_factorial_0 * h_0 * dM_2 + inv_factorial_1 * h_1 * dM_3
                       + inv_factorial_2 * h_2 * dM_4 + inv_factorial_3 * h_3 * dM_5;

            dM_ret_3 = inv_factorial_0 * h_0 * dM_3 + inv_factorial_1 * h_1 * dM_4
                       + inv_factorial_2 * h_2 * dM_5;

            dM_ret_4 = inv_factorial_0 * h_0 * dM_4 + inv_factorial_1 * h_1 * dM_5;
            dM_ret_5 = inv_factorial_0 * h_0 * dM_5;

            return dM_ret;
        } else if constexpr (low_order == 1 && high_order == 4) {

            SymTensorCollection<T, 0, 4> h = SymTensorCollection<T, 0, 4>::from_vec(offset);

            auto &h_0 = h.t0;
            auto &h_1 = h.t1;
            auto &h_2 = h.t2;
            auto &h_3 = h.t3;
            auto &h_4 = h.t4;

            auto &dM_1 = dM.t1;
            auto &dM_2 = dM.t2;
            auto &dM_3 = dM.t3;
            auto &dM_4 = dM.t4;

            shammath::SymTensorCollection<T, 1, 4> dM_ret;

            auto &dM_ret_1 = dM_ret.t1;
            auto &dM_ret_2 = dM_ret.t2;
            auto &dM_ret_3 = dM_ret.t3;
            auto &dM_ret_4 = dM_ret.t4;

            // dM_k = sum_{l=k}^p \frac{1}{(l-k)!} h^(l-k).dM_l

            dM_ret_1 = inv_factorial_0 * h_0 * dM_1 + inv_factorial_1 * h_1 * dM_2
                       + inv_factorial_2 * h_2 * dM_3 + inv_factorial_3 * h_3 * dM_4;

            dM_ret_2 = inv_factorial_0 * h_0 * dM_2 + inv_factorial_1 * h_1 * dM_3
                       + inv_factorial_2 * h_2 * dM_4;

            dM_ret_3 = inv_factorial_0 * h_0 * dM_3 + inv_factorial_1 * h_1 * dM_4;

            dM_ret_4 = inv_factorial_0 * h_0 * dM_4;

            return dM_ret;
        } else if constexpr (low_order == 1 && high_order == 3) {

            SymTensorCollection<T, 0, 3> h = SymTensorCollection<T, 0, 3>::from_vec(offset);

            auto &h_0 = h.t0;
            auto &h_1 = h.t1;
            auto &h_2 = h.t2;
            auto &h_3 = h.t3;

            auto &dM_1 = dM.t1;
            auto &dM_2 = dM.t2;
            auto &dM_3 = dM.t3;

            shammath::SymTensorCollection<T, 1, 3> dM_ret;

            auto &dM_ret_1 = dM_ret.t1;
            auto &dM_ret_2 = dM_ret.t2;
            auto &dM_ret_3 = dM_ret.t3;

            // dM_k = sum_{l=k}^p \frac{1}{(l-k)!} h^(l-k).dM_l

            dM_ret_1 = inv_factorial_0 * h_0 * dM_1 + inv_factorial_1 * h_1 * dM_2
                       + inv_factorial_2 * h_2 * dM_3;

            dM_ret_2 = inv_factorial_0 * h_0 * dM_2 + inv_factorial_1 * h_1 * dM_3;

            dM_ret_3 = inv_factorial_0 * h_0 * dM_3;

            return dM_ret;
        } else if constexpr (low_order == 1 && high_order == 2) {

            SymTensorCollection<T, 0, 2> h = SymTensorCollection<T, 0, 2>::from_vec(offset);

            auto &h_0 = h.t0;
            auto &h_1 = h.t1;
            auto &h_2 = h.t2;

            auto &dM_1 = dM.t1;
            auto &dM_2 = dM.t2;

            shammath::SymTensorCollection<T, 1, 2> dM_ret;

            auto &dM_ret_1 = dM_ret.t1;
            auto &dM_ret_2 = dM_ret.t2;

            // dM_k = sum_{l=k}^p \frac{1}{(l-k)!} h^(l-k).dM_l

            dM_ret_1 = inv_factorial_0 * h_0 * dM_1 + inv_factorial_1 * h_1 * dM_2;

            dM_ret_2 = inv_factorial_0 * h_0 * dM_2;

            return dM_ret;
        } else {
            static_assert(shambase::always_false_v<T>, "This combination of orders is not valid");
        }
    }

    // Offset the gravitational moment derivatives (with two vecs)
    template<class T, u32 low_order, u32 high_order>
    inline shammath::SymTensorCollection<T, low_order, high_order> offset_dM_mat(
        const shammath::SymTensorCollection<T, low_order, high_order> &dM,
        const sycl::vec<T, 3> &from,
        const sycl::vec<T, 3> &to) {
        return offset_dM_mat_delta(dM, to - from);
    }

} // namespace shamphys
