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
 * @file contract_grav_moment.hpp
 * @author Timothée David--Cléris (tim.shamrock@proton.me)
 * @brief
 */

#include "shambase/aliases_int.hpp"
#include "shammath/symtensor_collections.hpp"
#include "shamphys/fmm/grav_moments.hpp"

namespace shamphys {

    template<class T, u32 order>
    inline sycl::vec<T, 3> contract_grav_moment_to_force(
        shammath::SymTensorCollection<T, 0, order - 1> &a_k,
        shammath::SymTensorCollection<T, 1, order> &dM_k) {

        using vec = sycl::vec<T, 3>;

        auto tensor_to_sycl = [](shammath::SymTensor3d_1<T> a) {
            return sycl::vec<T, 3>{a.v_0, a.v_1, a.v_2};
        };

        auto force_val = tensor_to_sycl(dM_k.t1 * a_k.t0);
        if constexpr (order >= 2) {
            force_val += tensor_to_sycl(dM_k.t2 * a_k.t1) / 1;
        }
        if constexpr (order >= 3) {
            force_val += tensor_to_sycl(dM_k.t3 * a_k.t2) / 2;
        }
        if constexpr (order >= 4) {
            force_val += tensor_to_sycl(dM_k.t4 * a_k.t3) / 6;
        }
        if constexpr (order >= 5) {
            force_val += tensor_to_sycl(dM_k.t5 * a_k.t4) / 24;
        }

        return force_val;
    }
} // namespace shamphys
