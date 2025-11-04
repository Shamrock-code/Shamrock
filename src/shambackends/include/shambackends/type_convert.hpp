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
 * @file type_convert.hpp
 * @author Timothée David--Cléris (tim.shamrock@proton.me)
 * @brief Contains functions for converting between SYCL vector types and
 *        C++ standard library array types.
 */

#include "shambase/aliases_float.hpp"
#include "shambase/type_traits.hpp"
#include "shambackends/typeAliasFp16.hpp"
#include "shambackends/typeAliasVec.hpp"

#if __has_include(<nlohmann/json.hpp>)
    #include <nlohmann/json.hpp>
#endif

namespace sham {

    /**
     * @brief Converts a SYCL vector into a C++ standard library array.
     *
     * @param v SYCL vector to convert
     * @return C++ standard library array containing the same elements
     */
    template<class T, int n>
    std::array<T, n> sycl_vec_to_array(sycl::vec<T, n> v) {
        if constexpr (n == 1) {
            return {v[0]};
        } else if constexpr (n == 2) {
            return {v[0], v[1]};
        } else if constexpr (n == 3) {
            return {v[0], v[1], v[2]};
        } else if constexpr (n == 4) {
            return {v[0], v[1], v[2], v[3]};
        } else if constexpr (n == 8) {
            return {v[0], v[1], v[2], v[3], v[4], v[5], v[6], v[7]};
        } else if constexpr (n == 16) {
            return {
                v[0],
                v[1],
                v[2],
                v[3],
                v[4],
                v[5],
                v[6],
                v[7],
                v[8],
                v[9],
                v[10],
                v[11],
                v[12],
                v[13],
                v[14],
                v[15]};
        } else {
            static_assert(shambase::always_false_v<T>, "This case is not handled");
        }
    }

    /**
     * @brief Converts a C++ standard library array into a SYCL vector.
     *
     * @param v C++ standard library array to convert
     * @return SYCL vector containing the same elements
     */
    template<class T, size_t n>
    sycl::vec<T, n> array_to_sycl_vec(std::array<T, n> v) {
        if constexpr (n == 1) {
            return {v[0]};
        } else if constexpr (n == 2) {
            return {v[0], v[1]};
        } else if constexpr (n == 3) {
            return {v[0], v[1], v[2]};
        } else if constexpr (n == 4) {
            return {v[0], v[1], v[2], v[3]};
        } else if constexpr (n == 8) {
            return {v[0], v[1], v[2], v[3], v[4], v[5], v[6], v[7]};
        } else if constexpr (n == 16) {
            return {
                v[0],
                v[1],
                v[2],
                v[3],
                v[4],
                v[5],
                v[6],
                v[7],
                v[8],
                v[9],
                v[10],
                v[11],
                v[12],
                v[13],
                v[14],
                v[15]};
        } else {
            static_assert(shambase::always_false_v<T>, "This case is not handled");
        }
    }

} // namespace sham

#if __has_include(<nlohmann/json.hpp>)
NLOHMANN_JSON_NAMESPACE_BEGIN
template<typename T, int n>
struct adl_serializer<sycl::vec<T, n>> {
    static void to_json(json &j, const sycl::vec<T, n> &p) { j = sham::sycl_vec_to_array(p); }

    static void from_json(const json &j, sycl::vec<T, n> &p) {
        p = sham::array_to_sycl_vec(j.get<std::array<T, n>>());
    }
};
NLOHMANN_JSON_NAMESPACE_END
#endif
