// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright (c) 2021-2025 Timothée David--Cléris <tim.shamrock@proton.me>
// SPDX-License-Identifier: CeCILL Free Software License Agreement v2.1
// Shamrock is licensed under the CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

/**
 * @file EOSConfig.cpp
 * @author David Fang (david.fang@ikmail.com)
 * @author Timothée David--Cléris (tim.shamrock@proton.me)
 * @author Yona Lapeyre (yona.lapeyre@ens-lyon.fr) --no git blame--
 * @brief
 *
 */

#include "shammodels/common/EOSConfig.hpp"
#include "shambackends/typeAliasVec.hpp"

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
    void to_json(nlohmann::json &j, const EOSConfig<Tvec> &p) {
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

        using Isothermal    = typename EOSConfig<Tvec>::Isothermal;
        using Adiabatic     = typename EOSConfig<Tvec>::Adiabatic;
        using Polytropic    = typename EOSConfig<Tvec>::Polytropic;
        using LocIsoT       = typename EOSConfig<Tvec>::LocallyIsothermal;
        using LocIsoTLP07   = typename EOSConfig<Tvec>::LocallyIsothermalLP07;
        using LocIsoTFA2014 = typename EOSConfig<Tvec>::LocallyIsothermalFA2014;
        using Fermi         = typename EOSConfig<Tvec>::Fermi;

        if (const Isothermal *eos_config = std::get_if<Isothermal>(&p.config)) {
            j = json{{"Tvec", type_id}, {"eos_type", "isothermal"}, {"cs", eos_config->cs}};
        } else if (const Adiabatic *eos_config = std::get_if<Adiabatic>(&p.config)) {
            j = json{{"Tvec", type_id}, {"eos_type", "adiabatic"}, {"gamma", eos_config->gamma}};
        } else if (const Polytropic *eos_config = std::get_if<Polytropic>(&p.config)) {
            j = json{
                {"Tvec", type_id},
                {"eos_type", "polytropic"},
                {"K", eos_config->K},
                {"gamma", eos_config->gamma}};
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
        } else if (const Fermi *eos_config = std::get_if<Fermi>(&p.config)) {
            j = json{{"Tvec", type_id}, {"eos_type", "fermi"}, {"mu_e", eos_config->mu_e}};
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
    void from_json(const nlohmann::json &j, EOSConfig<Tvec> &p) {

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

        using Isothermal    = typename EOSConfig<Tvec>::Isothermal;
        using Adiabatic     = typename EOSConfig<Tvec>::Adiabatic;
        using Polytropic    = typename EOSConfig<Tvec>::Polytropic;
        using LocIsoT       = typename EOSConfig<Tvec>::LocallyIsothermal;
        using LocIsoTLP07   = typename EOSConfig<Tvec>::LocallyIsothermalLP07;
        using LocIsoTFA2014 = typename EOSConfig<Tvec>::LocallyIsothermalFA2014;
        using Fermi         = typename EOSConfig<Tvec>::Fermi;

        if (eos_type == "isothermal") {
            p.config = Isothermal{j.at("cs").get<Tscal>()};
        } else if (eos_type == "adiabatic") {
            p.config = Adiabatic{j.at("gamma").get<Tscal>()};
        } else if (eos_type == "polytropic") {
            p.config = Polytropic{j.at("K").get<Tscal>(), j.at("gamma").get<Tscal>()};
        } else if (eos_type == "locally_isothermal") {
            p.config = LocIsoT{};
        } else if (eos_type == "locally_isothermal_lp07") {
            p.config = LocIsoTLP07{
                j.at("cs0").get<Tscal>(), j.at("q").get<Tscal>(), j.at("r0").get<Tscal>()};
        } else if (eos_type == "locally_isothermal_fa2014") {
            p.config = LocIsoTFA2014{j.at("h_over_r").get<Tscal>()};
        } else if (eos_type == "fermi") {
            p.config = Fermi{j.at("mu_e").get<Tscal>()};
        } else {
            shambase::throw_unimplemented("Unknown or unsupported eos_type found in json");
        }
    }

    template void to_json<f64_3>(nlohmann::json &j, const EOSConfig<f64_3> &p);
    template void from_json<f64_3>(const nlohmann::json &j, EOSConfig<f64_3> &p);

} // namespace shammodels
