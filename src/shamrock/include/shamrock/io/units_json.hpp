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
 * @file units_json.hpp
 * @author Timothée David--Cléris (tim.shamrock@proton.me)
 * @author Yona Lapeyre (yona.lapeyre@ens-lyon.fr)
 * @brief
 *
 */

#include "shamrock/scheduler/SerialPatchTree.hpp"
#include <shamunits/Constants.hpp>
#include <shamunits/UnitSystem.hpp>

/**
 * @brief Converts an optional value to a JSON object.
 *
 * @param j The JSON object to be populated.
 * @param p The optional value to be converted.
 *
 *
 * @throws std::bad_optional_access if p is not engaged
 */
template<class T>
inline void to_json_optional(nlohmann::json &j, const std::optional<T> &p) {
    if (p) {
        j = *p;
    } else {
        j = {};
    }
}

/**
 * @brief Deserializes an optional value from a JSON object.
 *
 * @param j The JSON object to deserialize from.
 * @param p The optional value to populate.
 *
 *
 * @throws std::bad_optional_access if j is not a valid JSON object
 */
template<class T>
inline void from_json_optional(const nlohmann::json &j, std::optional<T> &p) {
    if (j.is_null()) {
        p = std::nullopt;
    } else {
        p = j.get<T>();
    }
}

namespace shamunits {

    /**
     * @brief Converts a UnitSystem object to a JSON object.
     *
     * @param j The JSON object to be populated.
     * @param p The UnitSystem object to be converted.
     */
    template<class Tscal>
    inline void to_json(nlohmann::json &j, const ::shamunits::UnitSystem<Tscal> &p) {
        j = nlohmann::json{
            {"unit_time", p.s_inv},
            {"unit_length", p.m_inv},
            {"unit_mass", p.kg_inv},
            {"unit_current", p.A_inv},
            {"unit_temperature", p.K_inv},
            {"unit_qte", p.mol_inv},
            {"unit_lumint", p.cd_inv}};
    }

    /**
     * @brief Deserializes a UnitSystem object from a JSON object.
     *
     * @param j The JSON object to deserialize from.
     * @param p The UnitSystem object to populate.
     */
    template<class Tscal>
    inline void from_json(const nlohmann::json &j, ::shamunits::UnitSystem<Tscal> &p) {
        p = ::shamunits::UnitSystem<Tscal>(
            j.at("unit_time").get<Tscal>(),
            j.at("unit_length").get<Tscal>(),
            j.at("unit_mass").get<Tscal>(),
            j.at("unit_current").get<Tscal>(),
            j.at("unit_temperature").get<Tscal>(),
            j.at("unit_qte").get<Tscal>(),
            j.at("unit_lumint").get<Tscal>());
    }

} // namespace shamunits
