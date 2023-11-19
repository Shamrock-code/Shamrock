// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright(C) 2021-2023 Timothée David--Cléris <timothee.david--cleris@ens-lyon.fr>
// Licensed under CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

#pragma once

/**
 * @file math.hpp
 * @author Timothée David--Cléris (timothee.david--cleris@ens-lyon.fr)
 * @brief
 *
 */

#include "shambase/sycl_utils/sycl_utilities.hpp"
#include "shambase/sycl_utils/vec_equals.hpp"

namespace sham {

    template<class T>
    inline T min(T a, T b) {
        return shambase::sycl_utils::g_sycl_min(a, b);
    }

    template<class T>
    inline T max(T a, T b) {
        return shambase::sycl_utils::g_sycl_max(a, b);
    }

    template<class T>
    inline T abs(T a) {
        return shambase::sycl_utils::g_sycl_abs(a);
    }

    template<class T>
    inline T positive_part(T a){
        return (g_sycl_abs(a) + a)/2;
    }

    template<class T>
    inline T negative_part(T a){
        return (g_sycl_abs(a) - a)/2;
    }

    template<class T>
    inline bool equals(T a, T b){
        return shambase::vec_equals(a,b);
    }


    inline auto pack32(u32 a, u32 b) -> u64 {
        return (u64(a) << 32U) + b;
    };

    inline auto unpack32 (u64 v) -> sycl::vec<u32, 2> {
        return {u32(v >> 32U), u32(v)};
    };

} // namespace sham