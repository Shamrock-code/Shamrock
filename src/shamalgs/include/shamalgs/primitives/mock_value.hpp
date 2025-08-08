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
 * @file mock_value.hpp
 * @author Timothée David--Cléris (tim.shamrock@proton.me)
 * @brief
 *
 */

#include "shambackends/vec.hpp"
#include <random>

namespace shamalgs {

    template<class T>
    T mock_value(std::mt19937 &eng, T min_bound, T max_bound);

    template<class T>
    inline T mock_value(std::mt19937 &eng) {
        using Prop = shambase::VectorProperties<T>;
        return mock_value<T>(eng, Prop::get_min(), Prop::get_max());
    }

} // namespace shamalgs
