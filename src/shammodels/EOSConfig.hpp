// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright (c) 2021-2024 Timothée David--Cléris <tim.shamrock@proton.me>
// SPDX-License-Identifier: CeCILL Free Software License Agreement v2.1
// Shamrock is licensed under the CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

#pragma once

/**
 * @file EOSConfig.hpp
 * @author Timothée David--Cléris (timothee.david--cleris@ens-lyon.fr)
 * @author Yona Lapeyre (yona.lapeyre@ens-lyon.fr)
 * @brief
 *
 */

#include "shambase/exception.hpp"
#include "shambase/type_traits.hpp"
#include "nlohmann/json_fwd.hpp"
#include "shambackends/vec.hpp"
#include "shamphys/eos_config.hpp"
#include "shamsys/legacy/log.hpp"
#include <nlohmann/json.hpp>
#include <type_traits>
#include <stdexcept>
#include <string>
#include <variant>

namespace shammodels {

    /**
     * @brief Configuration struct for the equation of state used in the hydrodynamic models
     *
     * @tparam Tvec The vector type used to store the physical quantities (e.g., velocity,
     * pressure, etc.)
     */
    template<class Tvec>
    struct EOSConfig {
        /// Scalar type associated to the vector template type
        using Tscal = shambase::VecComponent<Tvec>;

        /// Dimension of the vector quantities
        static constexpr u32 dim = shambase::VectorProperties<Tvec>::dimension;

        // EOS types definition usable in the code

        /// Adiabatic equation of state configuration
        using Adiabatic = shamphys::EOS_Config_Adiabatic<Tscal>;

        /// Locally isothermal equation of state configuration
        struct LocallyIsothermal {};

        /// Locally isothermal equation of state configuration from Lodato Price 2007
        using LocallyIsothermalLP07 = shamphys::EOS_Config_LocallyIsothermal_LP07<Tscal>;

        /// Locally isothermal equation of state configuration from Lodato Price 2007
        using LocallyIsothermalFA2014
            = shamphys::EOS_Config_LocallyIsothermalDisc_Farris2014<Tscal>;

        /// Variant type to store the EOS configuration
        using Variant = std::
            variant<Adiabatic, LocallyIsothermal, LocallyIsothermalLP07, LocallyIsothermalFA2014>;

        /// Current EOS configuration
        Variant config = Adiabatic{};

        /**
         * @brief Set the EOS configuration to an adiabatic equation of state
         *
         * @param gamma The adiabatic index
         */
        inline void set_adiabatic(Tscal gamma) { config = Adiabatic{gamma}; }

        /**
         * @brief Set the EOS configuration to a locally isothermal equation of state
         */
        inline void set_locally_isothermal() { config = LocallyIsothermal{}; }

        /**
         * @brief Set the EOS configuration to a locally isothermal equation of state
         * (Lodato Price 2007)
         *
         * The equation of state is given by:
         * \f$ p = c_{s,0}^2 (r / r_0)^{-q} \rho \f$
         *
         * @param cs0 Soundspeed at the reference radius
         * @param q Power exponent of the soundspeed profile
         * @param r0 Reference radius
         */
        inline void set_locally_isothermalLP07(Tscal cs0, Tscal q, Tscal r0) {
            config = LocallyIsothermalLP07{cs0, q, r0};
        }

        inline void set_locally_isothermalFA2014(Tscal h_over_r) {
            config = LocallyIsothermalFA2014{h_over_r};
        }

        /**
         * @brief Print current status of the EOSConfig
         */
        inline void print_status();
    };

} // namespace shammodels

template<class Tvec>
void shammodels::EOSConfig<Tvec>::print_status() {

    std::string s;
    if constexpr (std::is_same_v<f32_3, Tvec>) {
        s = "f32_3";
    }

    if constexpr (std::is_same_v<f64_3, Tvec>) {
        s = "f64_3";
    }

    logger::raw_ln("EOS config", s, ":");
    if (Adiabatic *eos_config = std::get_if<Adiabatic>(&config)) {
        logger::raw_ln("adiabatic : ");
        logger::raw_ln("gamma", eos_config->gamma);
    } else if (LocallyIsothermal *eos_config = std::get_if<LocallyIsothermal>(&config)) {
        logger::raw_ln("locally isothermal : ");
    } else if (LocallyIsothermalLP07 *eos_config = std::get_if<LocallyIsothermalLP07>(&config)) {
        logger::raw_ln("locally isothermal (Lodato Price 2007) : ");
    } else if (
        LocallyIsothermalFA2014 *eos_config = std::get_if<LocallyIsothermalFA2014>(&config)) {
        logger::raw_ln("locally isothermal (Farris 2014) : ");
    } else {
        shambase::throw_unimplemented();
    }
}

namespace shammodels {

    /**
     * @brief Serialize EOSConfig to json
     *
     * This function serializes an EOSConfig to a json object with the following format:
     * \code{.json}
     * {
     *    "Tvec": "f32_3"|"f64_3",
     *    "eos_type": "adiabatic"|"locally_isothermal"|"locally_isothermal_lp07",
     *    "gamma"?: number,
     *    "cs0"?: number,
     *    "q"?: number,
     *    "r0"?: number
     * }
     * \endcode
     *
     * @param j json object
     * @param p EOSConfig to serialize
     */
    template<class Tvec>
    inline void to_json(nlohmann::json &j, const EOSConfig<Tvec> &p) {
        // Serialize EOSConfig to a json object

        using json = nlohmann::json;

        std::string type_id = "";

        if constexpr (std::is_same_v<f32_3, Tvec>) {
            type_id = "f32_3"; // type of the vector quantities (e.g. position)
        } else if constexpr (std::is_same_v<f64_3, Tvec>) {
            type_id = "f64_3"; // type of the vector quantities (e.g. position)
        } else {
            static_assert(shambase::always_false_v<Tvec>, "This Tvec type is not handled");
        }

        using Adiabatic     = typename EOSConfig<Tvec>::Adiabatic;
        using LocIsoT       = typename EOSConfig<Tvec>::LocallyIsothermal;
        using LocIsoTLP07   = typename EOSConfig<Tvec>::LocallyIsothermalLP07;
        using LocIsoTFA2014 = typename EOSConfig<Tvec>::LocallyIsothermalFA2014;

        if (const Adiabatic *eos_config = std::get_if<Adiabatic>(&p.config)) {
            j = json{{"Tvec", type_id}, {"eos_type", "adiabatic"}, {"gamma", eos_config->gamma}};
        } else if (const LocIsoT *eos_config = std::get_if<LocIsoT>(&p.config)) {
            j = json{{"Tvec", type_id}, {"eos_type", "locally_isothermal"}};
        } else if (const LocIsoTLP07 *eos_config = std::get_if<LocIsoTLP07>(&p.config)) {
            j = json{
                {"Tvec", type_id},
                {"eos_type", "locally_isothermal_lp07"},
                {"cs0", eos_config->cs0},
                {"q", eos_config->q},
                {"r0", eos_config->r0}};
        } else if (const LocIsoTFA2014 *eos_config = std::get_if<LocIsoTFA2014>(&p.config)) {
            j = json{
                {"Tvec", type_id},
                {"eos_type", "locally_isothermal_fa2014"},
                {"h_over_r", eos_config->h_over_r}};
        } else {
            shambase::throw_unimplemented(); // should never be reached
        }
    }

    /**
     * @brief Deserializes an EOSConfig<Tvec> from a JSON object
     *
     * @tparam Tvec The vector type of the EOSConfig<Tvec>
     * @param j The JSON object to deserialize
     * @param p The EOSConfig<Tvec> to deserialize to
     *
     * @see shammodels::to_json(nlohmann::json &j, const EOSConfig<Tvec> &p) for details of the
     * format
     *
     * Throws an std::runtime_error if the JSON object is not in the expected format
     */
    template<class Tvec>
    inline void from_json(const nlohmann::json &j, EOSConfig<Tvec> &p) {

        using Tscal = shambase::VecComponent<Tvec>;

        std::string type_id;
        j.at("Tvec").get_to(type_id);

        if constexpr (std::is_same_v<f32_3, Tvec>) {
            if (type_id != "f32_3") {
                shambase::throw_with_loc<std::invalid_argument>(
                    "You are trying to create a EOSConfig with the wrong vector type");
            }
        } else if constexpr (std::is_same_v<f64_3, Tvec>) {
            if (type_id != "f64_3") {
                shambase::throw_with_loc<std::invalid_argument>(
                    "You are trying to create a EOSConfig with the wrong vector type");
            }
        } else {
            static_assert(shambase::always_false_v<Tvec>, "This Tvec type is not handled");
        }

        if (!j.contains("eos_type")) {
            shambase::throw_with_loc<std::runtime_error>("no field eos_type is found in this json");
        }

        std::string eos_type;
        j.at("eos_type").get_to(eos_type);

        using Adiabatic     = typename EOSConfig<Tvec>::Adiabatic;
        using LocIsoT       = typename EOSConfig<Tvec>::LocallyIsothermal;
        using LocIsoTLP07   = typename EOSConfig<Tvec>::LocallyIsothermalLP07;
        using LocIsoTFA2014 = typename EOSConfig<Tvec>::LocallyIsothermalFA2014;

        if (eos_type == "adiabatic") {
            p.config = Adiabatic{j.at("gamma").get<Tscal>()};
        } else if (eos_type == "locally_isothermal") {
            p.config = LocIsoT{};
        } else if (eos_type == "locally_isothermal_lp07") {
            p.config = LocIsoTLP07{
                j.at("cs0").get<Tscal>(), j.at("q").get<Tscal>(), j.at("r0").get<Tscal>()};
        } else if (eos_type == "locally_isothermal_fa2014") {
            p.config = LocIsoTFA2014{j.at("h_over_r").get<Tscal>()};
        } else {
            shambase::throw_unimplemented("wtf !");
        }
    }

} // namespace shammodels
