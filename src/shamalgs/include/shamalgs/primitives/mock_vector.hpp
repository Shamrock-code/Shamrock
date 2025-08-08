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
 * @file mock_vector.hpp
 * @author Timothée David--Cléris (tim.shamrock@proton.me)
 * @brief
 *
 */

#include "shamalgs/details/random/random.hpp"
#include <random>

namespace shamalgs {

    template<class T>
    std::vector<T> mock_vector(u64 seed, u32 len, T min_bound, T max_bound) {
        std::vector<T> vec;

        std::mt19937 eng(seed);

        for (u32 i = 0; i < len; i++) {
            vec.push_back(mock_value(eng, min_bound, max_bound));
        }

        return std::move(vec);
    }

    template<class T>
    inline std::vector<T> mock_vector(u64 seed, u32 len) {
        using Prop = shambase::VectorProperties<T>;
        return mock_vector(seed, len, Prop::get_min(), Prop::get_max());
    }

} // namespace shamalgs
