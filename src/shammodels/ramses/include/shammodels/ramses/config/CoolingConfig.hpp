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
 * @file CoolingConfig.hpp
 * @author Timothée David--Cléris (tim.shamrock@proton.me)
 * @brief Cooling config of the Ramses solver + json serialization/deserialization
 */

#include "shambase/exception.hpp"
#include "shambase/overloaded.hpp"
#include "shambase/string.hpp"
#include "shambackends/vec.hpp"
#include "shammath/riemann_common.hpp"
#include <nlohmann/json.hpp>
#include <stdexcept>
#include <string>
#include <variant>

namespace shammodels::basegodunov {

    /**
     * @brief Cooling of the gas, applied on top of the (adiabatic) equation of state
     */
    template<class Tvec>
    struct CoolingConfig {
        using Tscal = shambase::VecComponent<Tvec>;

        /// No cooling
        struct None {};

        /**
         * @brief Instantaneous cooling onto a barotrope (RAMSES-like)
         *
         * At the end of each timestep, before the CFL condition, the internal energy of every
         * cell is overwritten by \f$ P_{\rm baro}(\rho) / (\gamma - 1) \f$, where \f$ \gamma \f$
         * is the adiabatic index of the equation of state and \f$ P_{\rm baro} \f$ the pressure
         * of get_barotrope() (isothermal with sound speed cs0 below rho0, adiabatic with index
         * gamma0 above), see shammath::FluidStateBarotropic.
         */
        struct InstantBarotropic {
            Tscal rho0;
            Tscal cs0;
            Tscal gamma0 = 5. / 3.;

            inline shammath::FluidStateBarotropic<Tvec> get_barotrope() const {
                return shammath::FluidStateBarotropic<Tvec>(rho0, cs0, gamma0);
            }
        };

        using Variant = std::variant<None, InstantBarotropic>;

        Variant config = None{};

        inline void set_none() { config = None{}; }
        inline void set_instant_barotropic(Tscal rho0, Tscal cs0, Tscal gamma0) {
            config = InstantBarotropic{rho0, cs0, gamma0};
        }

        inline bool is_cooling_on() const { return !std::holds_alternative<None>(config); }

        /// Throws std::invalid_argument if the parameters of the cooling are invalid
        inline void check_config() const {
            if (const InstantBarotropic *cfg = std::get_if<InstantBarotropic>(&config)) {
                if (!(cfg->rho0 > 0.0) || !(cfg->cs0 > 0.0)) {
                    shambase::throw_with_loc<std::invalid_argument>(sham::format(
                        "InstantBarotropic cooling: rho0 and cs0 must be > 0, currently rho0 = "
                        "{}, cs0 = {}",
                        cfg->rho0,
                        cfg->cs0));
                }
            }
        }
    };

    template<class Tvec>
    inline void to_json(nlohmann::json &j, const CoolingConfig<Tvec> &p) {
        using Cooling = CoolingConfig<Tvec>;

        std::visit(
            shambase::overloaded{
                [&](const typename Cooling::None &) {
                    j = {{"type", "none"}};
                },
                [&](const typename Cooling::InstantBarotropic &cfg) {
                    j
                        = {{"type", "instant_barotropic"},
                           {"rho0", cfg.rho0},
                           {"cs0", cfg.cs0},
                           {"gamma0", cfg.gamma0}};
                },
            },
            p.config);
    }

    template<class Tvec>
    inline void from_json(const nlohmann::json &j, CoolingConfig<Tvec> &p) {
        using Tscal = shambase::VecComponent<Tvec>;

        const std::string type = j.at("type").get<std::string>();
        if (type == "none") {
            p.set_none();
        } else if (type == "instant_barotropic") {
            p.set_instant_barotropic(
                j.at("rho0").get<Tscal>(), j.at("cs0").get<Tscal>(), j.at("gamma0").get<Tscal>());
        } else {
            shambase::throw_with_loc<std::runtime_error>("Invalid cooling type: " + type);
        }
    }

} // namespace shammodels::basegodunov
