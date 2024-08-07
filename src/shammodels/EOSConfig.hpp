// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright(C) 2021-2023 Timothée David--Cléris <timothee.david--cleris@ens-lyon.fr>
// Licensed under CeCILL 2.1 License, see LICENSE for more information
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

    template<class Tvec>
    struct EOSConfig {

        using Tscal              = shambase::VecComponent<Tvec>;
        static constexpr u32 dim = shambase::VectorProperties<Tvec>::dimension;

        // EOS types definition usable in the code

        using Adiabatic = shamphys::EOS_Config_Adiabatic<Tscal>;

        struct LocallyIsothermal {};


        using LocallyIsothermalLP07 = shamphys::EOS_Config_LocallyIsothermal_LP07<Tscal>;

        // internal wiring of the eos to the code

        using Variant = std::variant<Adiabatic, LocallyIsothermal, LocallyIsothermalLP07>;

        Variant config = Adiabatic{};

        /// Set the config to an adiabatic equation of state
        inline void set_adiabatic(Tscal gamma) { config = Adiabatic{gamma}; }

        /// Set the config to a locally isothermal equation of state
        inline void set_locally_isothermal() { config = LocallyIsothermal{}; }

        /// Set the config to a locally isothermal (Lodato Price 2007) equation of state
        inline void set_locally_isothermalLP07(Tscal cs0, Tscal q, Tscal r0) {
            config = LocallyIsothermalLP07{cs0, q, r0};
        }

        /// Print current status of the EOSConfig
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
    } else {
        shambase::throw_unimplemented();
    }
}

namespace shammodels {

    /**
     * @brief Serialize EOSConfig to json
     *
     * This function is using the following JSON format:
     * \todo
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

        using Adiabatic   = typename EOSConfig<Tvec>::Adiabatic;
        using LocIsoT     = typename EOSConfig<Tvec>::LocallyIsothermal;
        using LocIsoTLP07 = typename EOSConfig<Tvec>::LocallyIsothermalLP07;

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
     * This function is using the following JSON format:
     * \todo
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

        using Adiabatic   = typename EOSConfig<Tvec>::Adiabatic;
        using LocIsoT     = typename EOSConfig<Tvec>::LocallyIsothermal;
        using LocIsoTLP07 = typename EOSConfig<Tvec>::LocallyIsothermalLP07;

        if (eos_type == "adiabatic") {
            p.config = Adiabatic{j.at("gamma").get<Tscal>()};
        } else if (eos_type == "locally_isothermal") {
            p.config = LocIsoT{};
        } else if (eos_type == "locally_isothermal_lp07") {
            p.config = LocIsoTLP07{
                j.at("cs0").get<Tscal>(), j.at("q").get<Tscal>(), j.at("r0").get<Tscal>()};
        } else {
            shambase::throw_unimplemented("wtf !");
        }
    }

} // namespace shammodels
