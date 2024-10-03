// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright(C) 2021-2023 Timothée David--Cléris <timothee.david--cleris@ens-lyon.fr>
// Licensed under CeCILL 2.1 License, see LICENSE for more information
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
