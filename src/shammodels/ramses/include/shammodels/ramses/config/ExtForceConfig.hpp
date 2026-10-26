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
 * @file ExtForceConfig.hpp
 * @author Léodasce Sewanou (leodasce.sewanou@ens-lyon.fr)
 * @author Timothée David--Cléris (tim.shamrock@proton.me)
 * @brief External forces config of the Ramses solver + json serialization/deserialization
 */

#include "shambase/exception.hpp"
#include "shambase/overloaded.hpp"
#include "shambackends/type_convert.hpp"
#include "shambackends/vec.hpp"
#include <nlohmann/json.hpp>
#include <stdexcept>
#include <string>
#include <variant>
#include <vector>

namespace shammodels::basegodunov {

    /**
     * @brief List of the external forces applied to the gas
     *
     * The accelerations of all the forces are summed (in the order of the list) before being
     * applied to the conservative state.
     */
    template<class Tvec>
    struct ExtForceConfig {
        using Tscal = shambase::VecComponent<Tvec>;

        /// Constant gravitational acceleration g
        struct ConstantGravity {
            Tvec g;
        };

        using Variant = std::variant<ConstantGravity>;

        std::vector<Variant> ext_forces;

        inline void add_constant_gravity(Tvec g) { ext_forces.push_back(ConstantGravity{g}); }

        inline bool has_ext_forces() const { return !ext_forces.empty(); }
    };

    template<class Tvec>
    inline void to_json(nlohmann::json &j, const ExtForceConfig<Tvec> &p) {
        using EF = ExtForceConfig<Tvec>;

        nlohmann::json force_list = nlohmann::json::array();
        for (const auto &var_force : p.ext_forces) {
            std::visit(
                shambase::overloaded{
                    [&](const typename EF::ConstantGravity &cfg) {
                        force_list.push_back(
                            nlohmann::json{{"force_type", "constant_gravity"}, {"g", cfg.g}});
                    },
                },
                var_force);
        }

        j = nlohmann::json{{"force_list", force_list}};
    }

    template<class Tvec>
    inline void from_json(const nlohmann::json &j, ExtForceConfig<Tvec> &p) {
        p.ext_forces.clear();

        for (const auto &jforce : j.at("force_list")) {
            const std::string force_type = jforce.at("force_type").get<std::string>();
            if (force_type == "constant_gravity") {
                p.add_constant_gravity(jforce.at("g").get<Tvec>());
            } else {
                shambase::throw_with_loc<std::runtime_error>(
                    "Invalid external force type: " + force_type);
            }
        }
    }

} // namespace shammodels::basegodunov
