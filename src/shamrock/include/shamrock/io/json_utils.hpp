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
 * @file json_utils.hpp
 * @author Timothée David--Cléris (tim.shamrock@proton.me)
 * @brief
 *
 */

#include "shamrock/io/json_print_diff.hpp"

namespace shamrock {

    /// Shown the changes between two JSON objects to log config changes
    std::string log_json_changes(
        const nlohmann::json &j_current,
        const nlohmann::json &j,
        bool has_used_defaults,
        bool has_updated_config);

} // namespace shamrock

/**
 * @brief Macro to serialize/deserialize an enum to/from a JSON object modified from the
 * NLOHMANN_JSON_SERIALIZE_ENUM macro, throw an exception if the enum value is invalid and
 * do not use the first case by default.
 *
 * @param ENUM_TYPE The enum type to serialize/deserialize.
 * @param ... The pairs of enum values and their JSON representations.
 */
#define SHAMROCK_JSON_SERIALIZE_ENUM(ENUM_TYPE, ...)                                               \
    template<typename BasicJsonType>                                                               \
    inline void to_json(BasicJsonType &j, const ENUM_TYPE &e) {                                    \
        /* NOLINTNEXTLINE(modernize-type-traits) we use C++11 */                                   \
        static_assert(std::is_enum<ENUM_TYPE>::value, #ENUM_TYPE " must be an enum!");             \
        /* NOLINTNEXTLINE(modernize-avoid-c-arrays) we don't want to depend on <array> */          \
        static const std::pair<ENUM_TYPE, BasicJsonType> m[] = __VA_ARGS__;                        \
                                                                                                   \
        auto it = std::find_if(                                                                    \
            std::begin(m),                                                                         \
            std::end(m),                                                                           \
            [e](const std::pair<ENUM_TYPE, BasicJsonType> &ej_pair) -> bool {                      \
                return ej_pair.first == e;                                                         \
            });                                                                                    \
        if (it != std::end(m)) {                                                                   \
            j = it->second;                                                                        \
        } else {                                                                                   \
            throw shambase::make_except_with_loc<std::runtime_error>(                              \
                "Invalid " #ENUM_TYPE " value: " + std::to_string(e));                             \
        }                                                                                          \
    }                                                                                              \
    template<typename BasicJsonType>                                                               \
    inline void from_json(const BasicJsonType &j, ENUM_TYPE &e) {                                  \
        /* NOLINTNEXTLINE(modernize-type-traits) we use C++11 */                                   \
        static_assert(std::is_enum<ENUM_TYPE>::value, #ENUM_TYPE " must be an enum!");             \
        /* NOLINTNEXTLINE(modernize-avoid-c-arrays) we don't want to depend on <array> */          \
        static const std::pair<ENUM_TYPE, BasicJsonType> m[] = __VA_ARGS__;                        \
                                                                                                   \
        auto it = std::find_if(                                                                    \
            std::begin(m),                                                                         \
            std::end(m),                                                                           \
            [&j](const std::pair<ENUM_TYPE, BasicJsonType> &ej_pair) -> bool {                     \
                return ej_pair.second == j;                                                        \
            });                                                                                    \
        if (it != std::end(m)) {                                                                   \
            e = it->first;                                                                         \
        } else {                                                                                   \
            throw shambase::make_except_with_loc<std::runtime_error>(                              \
                "Invalid " #ENUM_TYPE " value: " + std::to_string(e));                             \
        }                                                                                          \
    }
