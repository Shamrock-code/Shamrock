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
 * @file ptr.hpp
 * @author Timothée David--Cléris (tim.shamrock@proton.me)
 * @brief
 *
 */
#include <cstddef>
#include <cstdint>

namespace shambase {

    /// @brief Check if a pointer is aligned with the given alignment.
    ///
    /// @param[in] ptr the pointer to check
    /// @param[in] alignment the alignment to check for
    ///
    /// @return `true` if the pointer is aligned, `false` otherwise
    inline bool is_aligned(const void *ptr, size_t alignment) noexcept {
        auto iptr = reinterpret_cast<std::uintptr_t>(ptr);
        return !(iptr % alignment);
    }

    /// @brief Check if a pointer is aligned with the given type.
    template<class T>
    inline bool is_aligned(const void *ptr) noexcept {
        return is_aligned(ptr, alignof(T));
    }

} // namespace shambase
