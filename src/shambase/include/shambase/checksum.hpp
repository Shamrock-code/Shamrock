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
 * @file checksum.hpp
 * @author Timothée David--Cléris (tim.shamrock@proton.me)
 * @brief
 *
 */

#include "shambase/aliases_int.hpp"
#include <cstddef>

namespace shambase {

    /**
     * @brief Compute the FNV-1a hash of a given data
     *
     * @param data
     * @param size
     * @return u64
     */
    inline u64 fnv1a_hash(const char *data, size_t size) {
        constexpr u64 fnv_offset_basis = 14695981039346656037ULL;
        constexpr u64 fnv_prime        = 1099511628211ULL;

        u64 hash = fnv_offset_basis;
        for (size_t i = 0; i < size; ++i) {
            hash ^= static_cast<u64>(static_cast<unsigned char>(data[i]));
            hash *= fnv_prime;
        }
        return hash;
    }

} // namespace shambase
