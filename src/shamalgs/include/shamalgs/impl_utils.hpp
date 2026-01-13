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
 * @file impl_utils.hpp
 * @author Timothée David--Cléris (tim.shamrock@proton.me)
 * @brief
 *
 */

#include <fmt/core.h>
#include <fmt/format.h>
#include <fmt/printf.h>
#include <fmt/ranges.h>
#include <string>

namespace shamalgs {

    struct impl_param {
        std::string impl_name;
        std::string params;
    };

} // namespace shamalgs

template<>
struct fmt::formatter<shamalgs::impl_param> {

    template<typename ParseContext>
    constexpr auto parse(ParseContext &ctx) {
        return ctx.begin();
    }

    template<typename FormatContext>
    auto format(shamalgs::impl_param c, FormatContext &ctx) const {
        return fmt::format_to(ctx.out(), "(impl_name = {}, params = {})", c.impl_name, c.params);
    }
};
