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
 * @file json_variant.hpp
 * @author Timothée David--Cléris (tim.shamrock@proton.me)
 * @brief
 *
 */

#include "shambase/exception.hpp"
#include "shambase/string.hpp"
#include <nlohmann/json.hpp>
#include <variant>

namespace shamrock {
    // Primary template (undefined for non-variant types)
    template<typename T>
    struct variant_to_tuple;

    // Specialization for std::variant
    template<typename... Ts>
    struct variant_to_tuple<std::variant<Ts...>> {
        using type = std::tuple<Ts...>;
    };

    // Helper alias
    template<typename T>
    using variant_to_tuple_t = typename variant_to_tuple<T>::type;

    template<class T>
    struct type_tag {
        using type = T;
    };

    template<class Functor, typename... Ts>
    inline bool on_variant_match(
        const std::string &type_id, Functor &&callback, const std::variant<Ts...> &var) {
        bool matched = false;

        (
            [&] {
                if (!matched && type_id == Ts::variant_type_name) {
                    callback(type_tag<Ts>{});
                    matched = true;
                }
            }(),
            ...);

        return matched;
    }

    template<class Functor, typename... Ts>
    inline void on_variant_cases(Functor &&callback, const std::variant<Ts...> &var) {

        (
            [&] {
                callback(type_tag<Ts>{});
            }(),
            ...);
    }

    template<typename... Ts>
    inline void json_deserialize_variant(
        const nlohmann::json &j, const std::string &type_id, std::variant<Ts...> &var) {

        bool matched = on_variant_match(
            type_id,
            [&](auto tag) {
                using Talt = typename decltype(tag)::type;
                var        = std::variant<Ts...>{j.get<Talt>()};
            },
            var);

        if (!matched) {
            std::vector<std::string> available_types;
            on_variant_cases(
                [&](auto tag) {
                    using Talt = typename decltype(tag)::type;
                    available_types.push_back(std::string(Talt::variant_type_name));
                },
                var);

            throw shambase::make_except_with_loc<std::runtime_error>(shambase::format(
                "unknown type: {}\navailable types: {}\njson: {}",
                type_id,
                available_types,
                j.dump(4)));
        }
    }

} // namespace shamrock
