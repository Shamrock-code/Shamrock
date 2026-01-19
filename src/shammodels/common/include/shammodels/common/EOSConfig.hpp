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
 * @file EOSConfig.hpp
 * @author David Fang (david.fang@ikmail.com)
 * @author Timothée David--Cléris (tim.shamrock@proton.me)
 * @author Yona Lapeyre (yona.lapeyre@ens-lyon.fr) --no git blame--
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

        using Polytropic = shamphys::EOS_Config_Polytropic<Tscal>;

        /// Isothermal equation of state configuration
        using Isothermal = shamphys::EOS_Config_Isothermal<Tscal>;

        /// Locally isothermal equation of state configuration
        struct LocallyIsothermal {};

        /// Locally isothermal equation of state configuration from Lodato Price 2007
        using LocallyIsothermalLP07 = shamphys::EOS_Config_LocallyIsothermal_LP07<Tscal>;

        /// Locally isothermal equation of state configuration from Lodato Price 2007
        using LocallyIsothermalFA2014
            = shamphys::EOS_Config_LocallyIsothermalDisc_Farris2014<Tscal>;

        /// Tillotson equation of state configuration
        using Tillotson = shamphys::EOS_Config_Tillotson<Tscal>;

        /// Variant type to store the EOS configuration
        using Variant = std::variant<
            Isothermal,
            Adiabatic,
            Polytropic,
            LocallyIsothermal,
            LocallyIsothermalLP07,
            LocallyIsothermalFA2014,
            Tillotson>;

        /// Current EOS configuration
        Variant config = Adiabatic{};

        /**
         * @brief Set the EOS configuration to an isothermal equation of state
         *
         * @param cs The sound speed
         */
        inline void set_isothermal(Tscal cs) { config = Isothermal{cs}; }

        /**
         * @brief Set the EOS configuration to an adiabatic equation of state
         *
         * @param gamma The adiabatic index
         */
        inline void set_adiabatic(Tscal gamma) { config = Adiabatic{gamma}; }

        /**
         * @brief Set the EOS configuration to an polytropic equation of state
         *
         * @param gamma The adiabatic index
         */
        inline void set_polytropic(Tscal K, Tscal gamma) { config = Polytropic{K, gamma}; }

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

        inline void set_tillotson(
            Tscal rho0,
            Tscal E0,
            Tscal A,
            Tscal B,
            Tscal a,
            Tscal b,
            Tscal alpha,
            Tscal beta,
            Tscal u_iv,
            Tscal u_cv) {
            config = Tillotson{rho0, E0, A, B, a, b, alpha, beta, u_iv, u_cv};
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
    if (Isothermal *eos_config = std::get_if<Isothermal>(&config)) {
        logger::raw_ln("isothermal : ");
        logger::raw_ln("cs", eos_config->cs);
    } else if (Adiabatic *eos_config = std::get_if<Adiabatic>(&config)) {
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
    void to_json(nlohmann::json &j, const EOSConfig<Tvec> &p);

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
    void from_json(const nlohmann::json &j, EOSConfig<Tvec> &p);

} // namespace shammodels
