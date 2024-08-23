// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright(C) 2021-2023 Timothée David--Cléris <timothee.david--cleris@ens-lyon.fr>
// Licensed under CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

#pragma once

/**
 * @file format.hpp
 * @author Timothée David--Cléris (tim.shamrock@proton.me)
 * @brief
 */

#include <fmt/core.h>
#include <fmt/format.h>
#include <fmt/printf.h>
#include <fmt/ranges.h>

namespace shambase {

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

} // namespace shambase
