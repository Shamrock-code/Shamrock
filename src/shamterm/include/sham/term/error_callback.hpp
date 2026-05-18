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
 * @file error_callback.hpp
 * @author Timothée David--Cléris (tim.shamrock@proton.me)
 * @brief
 *
 */

#include <source_location>
#include <functional>
#include <stdexcept>

namespace sham::term {

    /// @brief Callback signature for parsing error reporting (returns what, receives source
    /// location)
    using term_parse_callback_t
        = std::function<std::invalid_argument(const char *what, std::source_location where)>;

} // namespace sham::term
