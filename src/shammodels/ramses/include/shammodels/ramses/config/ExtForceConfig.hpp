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
#include "shammodels/ramses/config/enum_ExtForceCoupling.hpp"
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
     * applied to the conservative state, with the time quadrature selected by `coupling`.
     */
    template<class Tvec>
    struct ExtForceConfig {
        using Tscal = shambase::VecComponent<Tvec>;

        /// Constant gravitational acceleration g
        struct ConstantGravity {
            Tvec g;
        };

        // Point mass gravity
        struct PointMassGravity {
            Tscal mass         = 1;
            Tvec pos           = Tvec(0.5, 0.5, 0.5);
            Tscal epsilon_soft = 0.05;
        };

        using Variant = std::variant<ConstantGravity, PointMassGravity>;

        std::vector<Variant> ext_forces;
        bool is_coordinates_needed = false;

        /// Time quadrature of the external forces source term
        ExtForceCoupling coupling = ExtForceCoupling::Midpoint;

        inline void set_coupling(ExtForceCoupling mode) { coupling = mode; }

        inline void add_constant_gravity(Tvec g) { ext_forces.push_back(ConstantGravity{g}); }
        inline void add_point_mass_gravity(Tscal mass, Tvec pos, Tscal epsilon_soft) {
            ext_forces.push_back(PointMassGravity{mass, pos, epsilon_soft});
            is_coordinates_needed = true;
        }

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
                    [&](const typename EF::PointMassGravity &cfg) {
                        force_list.push_back(
                            nlohmann::json{
                                {"force_type", "point_mass_gravity"},
                                {"mass", cfg.mass},
                                {"pos", cfg.pos},
                                {"epsilon_soft", cfg.epsilon_soft}});
                    },
                },
                var_force);
        }

        j = nlohmann::json{{"force_list", force_list}, {"coupling", p.coupling}};
    }

    template<class Tvec>
    inline void from_json(const nlohmann::json &j, ExtForceConfig<Tvec> &p) {

        p.ext_forces.clear();

        // configs dumped before the coupling was configurable used the midpoint rule
        p.coupling = ExtForceCoupling::Midpoint;
        if (j.contains("coupling")) {
            j.at("coupling").get_to(p.coupling);
        }

        for (const auto &jforce : j.at("force_list")) {
            const std::string force_type = jforce.at("force_type").get<std::string>();
            using Tscal                  = shambase::VecComponent<Tvec>;
            if (force_type == "constant_gravity") {
                p.add_constant_gravity(jforce.at("g").get<Tvec>());
            } else if (force_type == "point_mass_gravity") {
                p.add_point_mass_gravity(
                    jforce.at("mass").get<Tscal>(),
                    jforce.at("pos").get<Tvec>(),
                    jforce.at("epsilon_soft").get<Tscal>());
            } else {
                shambase::throw_with_loc<std::runtime_error>(
                    "Invalid external force type: " + force_type);
            }
        }
    }

} // namespace shammodels::basegodunov
