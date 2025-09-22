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
 * @file lower_bound.hpp
 * @author Timothée David--Cléris (tim.shamrock@proton.me)
 * @brief GPU compatible implementation of std::lower_bound
 */

#include "shambase/aliases_int.hpp"
#include "shambase/assert.hpp"

namespace shamalgs::primitives {

    /// GPU compatible implementation of std::lower_bound
    template<class Tkey>
    constexpr u32 binary_search_lower_bound(
        const Tkey *__restrict__ key, u32 first, u32 last, const Tkey &value) {

        // modified from https://mhdm.dev/posts/sb_lower_bound/

        auto length = last - first;
        while (length > 0) {
            auto rem = length % 2;
            length /= 2;
            if (key[first + length] < value) {
                first += length + rem;
            }
        }

        return first;
    }

} // namespace shamalgs::primitives
