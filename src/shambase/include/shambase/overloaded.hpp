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
 * @file overloaded.hpp
 * @author Timothée David--Cléris (tim.shamrock@proton.me)
 * @brief
 *
 */

namespace shambase {

    /**
     * @brief Build an overload set out of several callables, for use with std::visit.
     *
     * Usage :
     * @code{.cpp}
     * std::visit(shambase::overloaded{
     *     [](const AltA &a) { ... },
     *     [](const AltB &b) { ... },
     * }, variant);
     * @endcode
     *
     * @tparam Ts the callable types (deduced)
     */
    template<class... Ts>
    struct overloaded : Ts... {
        using Ts::operator()...;
    };

    /// Deduction guide so `overloaded{lambda1, lambda2, ...}` deduces Ts... from the lambdas
    template<class... Ts>
    overloaded(Ts...) -> overloaded<Ts...>;

} // namespace shambase
