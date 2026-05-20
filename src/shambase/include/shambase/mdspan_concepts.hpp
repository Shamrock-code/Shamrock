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
 * @file mdspan_concepts.hpp
 * @author Timothée David--Cléris (tim.shamrock@proton.me)
 * @brief
 *
 */

#include <experimental/mdspan>

namespace shambase {

    namespace details {

        template<class T>
        struct _helper_is_mdspan : std::false_type {};

        template<class T, class Extents, class Layout, class Accessor>
        struct _helper_is_mdspan<std::mdspan<T, Extents, Layout, Accessor>> : std::true_type {};

    } // namespace details

    template<class T>
    inline constexpr bool is_mdspan_v = details::_helper_is_mdspan<std::remove_cvref_t<T>>::value;

    template<class T>
    using mdspan_value_t = typename std::remove_cvref_t<T>::value_type;

    template<class T>
    concept is_mdspan = is_mdspan_v<T>;

    template<class T, std::size_t Rank>
    concept is_mdspan_rank = is_mdspan<T> && (std::remove_cvref_t<T>::rank() == Rank);

    template<class A, class B>
    concept same_mdspan_value_t = std::same_as<mdspan_value_t<A>, mdspan_value_t<B>>;

} // namespace shambase
