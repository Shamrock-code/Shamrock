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
 * @file aliases.hpp
 * @author Timothée David--Cléris (tim.shamrock@proton.me)
 * @brief Type aliases for fmt types used throughout shamformat
 */

#include <fmt/base.h>

namespace sham {

    /**
     * @brief Formatter alias for `fmt::formatter`
     *
     * This alias is used to prevent explicit use of the `fmt` library in the
     * codebase. This way, we can change the formatting library without having
     * to modify all the code that uses it.
     *
     * @tparam T Type to format
     */
    template<class T>
    using formatter = fmt::formatter<T>;

    /// Alias for `fmt::format_error`, the exception type thrown by format errors
    using format_error = fmt::format_error;
} // namespace sham
