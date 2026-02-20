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
 * @file AVConfig.hpp
 * @author Timothée David--Cléris (tim.shamrock@proton.me)
 * @author Yona Lapeyre (yona.lapeyre@ens-lyon.fr)
 * @brief
 *
 */

#include "shambackends/type_traits.hpp"
#include "shambackends/vec.hpp"
#include "shamsys/legacy/log.hpp"
#include <nlohmann/json.hpp>
#include <variant>

namespace shammodels::sph {

    /**
     * @brief No artificial viscosity: \f$ q^a_ab = 0\f$
     */
    template<class Tscal>
    struct AVConfig_None {
        static constexpr std::string_view variant_type_name = "none";
    };

    /**
     * @brief Constant artificial viscosity: \f$ \alpha = cte\f$
     */
    template<class Tscal>
    struct AVConfig_Constant {
        static constexpr std::string_view variant_type_name = "constant";
        Tscal alpha_u                                       = 1.0;
        Tscal alpha_AV                                      = 1.0;
        Tscal beta_AV                                       = 2.0;
    };

    /**
     * @brief Morris & Monaghan 1997
     *
     */
    template<class Tscal>
    struct AVConfig_VaryingMM97 {
        static constexpr std::string_view variant_type_name = "varying_mm97";
        Tscal alpha_min                                     = 0.1;
        Tscal alpha_max                                     = 1.0;
        Tscal sigma_decay                                   = 0.1;
        Tscal alpha_u                                       = 1.0;
        Tscal beta_AV                                       = 2.0;
    };

    /**
     * @brief Cullen & Dehnen 2010
     *
     */
    template<class Tscal>
    struct AVConfig_VaryingCD10 {
        static constexpr std::string_view variant_type_name = "varying_cd10";
        Tscal alpha_min                                     = 0.1;
        Tscal alpha_max                                     = 1.0;
        Tscal sigma_decay                                   = 0.1;
        Tscal alpha_u                                       = 1.0;
        Tscal beta_AV                                       = 2.0;
    };

    /**
     * @brief Constant artificial viscosity for alpha disc viscosity
     */
    template<class Tscal>
    struct AVConfig_ConstantDisc {
        static constexpr std::string_view variant_type_name = "constant_disc";
        Tscal alpha_AV                                      = 1.0;
        Tscal alpha_u                                       = 1.0;
        Tscal beta_AV                                       = 2.0;
    };

    /**
     * @brief Configuration for the Artificial Viscosity (AV)
     *
     * This struct contains the information needed to configure the Artificial Viscosity
     * in the SPH algorithm. It is a variant of two possible types of artificial viscosity:
     * - None: no AV
     * - Constant: AV with a constant value
     * - VaryingMM97: AV with a varying value, using the Monaghan & Gingold 1997 prescription
     * - VaryingCD10: AV with a varying value, using the Cullen & Dehnen 2010 prescription
     * - ConstantDisc: AV with a constant value, but only in the disc plane
     *
     * @tparam Tvec type of the vector of coordinates
     */
    template<class Tvec>
    struct AVConfig;
} // namespace shammodels::sph

template<class Tvec>
struct shammodels::sph::AVConfig {

    /// Type of the components of the vector of coordinates
    using Tscal = shambase::VecComponent<Tvec>;
    /// Number of dimensions of the problem
    static constexpr u32 dim = shambase::VectorProperties<Tvec>::dimension;

    using None         = AVConfig_None<Tscal>;
    using Constant     = AVConfig_Constant<Tscal>;
    using VaryingMM97  = AVConfig_VaryingMM97<Tscal>;
    using VaryingCD10  = AVConfig_VaryingCD10<Tscal>;
    using ConstantDisc = AVConfig_ConstantDisc<Tscal>;

    /// Variant of all types of artificial viscosity possible
    using Variant = std::variant<None, Constant, VaryingMM97, VaryingCD10, ConstantDisc>;

    /// The actual configuration (default to constant viscosity)
    Variant config = Constant{};

    /// Set the configuration
    void set(Variant v) { config = v; }

    /**
     * @brief Sets the configuration to use a varying Cullen & Dehnen 2010 artificial viscosity.
     *
     * @param alpha_min the minimum value of alpha
     * @param alpha_max the maximum value of alpha
     * @param sigma_decay the decay rate of sigma
     * @param alpha_u the value of alpha_u
     * @param beta_AV the value of beta_AV
     */
    void set_varying_cd10(
        Tscal alpha_min, Tscal alpha_max, Tscal sigma_decay, Tscal alpha_u, Tscal beta_AV) {
        set(VaryingCD10{alpha_min, alpha_max, sigma_decay, alpha_u, beta_AV});
    }

    /**
     * @brief Checks if the current configuration has an alpha artificial viscosity field.
     *
     * @return true if the configuration has an alpha artificial viscosity field, false otherwise
     */
    inline bool has_alphaAV_field() {
        bool is_varying_alpha
            = bool(std::get_if<VaryingMM97>(&config)) || bool(std::get_if<VaryingCD10>(&config));
        return is_varying_alpha;
    }

    /**
     * @brief Checks if the current configuration need the divergence of the velocity field.
     *
     * @return true if the current configuration need the divergence of the velocity field, false
     * otherwise
     */
    inline bool has_divv_field() {
        bool is_varying_alpha
            = bool(std::get_if<VaryingMM97>(&config)) || bool(std::get_if<VaryingCD10>(&config));
        return is_varying_alpha;
    }

    /**
     * @brief Checks if the current configuration need the curl of the velocity field.
     *
     * @return true if the current configuration need the curl of the velocity field, false
     * otherwise
     */
    inline bool has_curlv_field() {
        bool is_varying_alpha = bool(std::get_if<VaryingCD10>(&config));
        return is_varying_alpha;
    }

    /**
     * @brief Checks if the current configuration has a dtdivv field.
     *
     * @return true if the configuration has a dtdivv field, false otherwise
     */
    inline bool has_dtdivv_field() {
        bool is_varying_alpha = bool(std::get_if<VaryingCD10>(&config));
        return is_varying_alpha;
    }

    /**
     * @brief Checks if the current configuration requires the soundspeed field.
     *
     * @return true if the configuration requires the soundspeed field, false otherwise.
     */
    inline bool has_field_soundspeed() {

        // this should not be needed idealy, but we need the pressure on the ghosts and
        // we don't want to communicate it as it can be recomputed from the other fields
        // hence we copy the soundspeed at the end of the step to a field in the patchdata
        // cf eos module there is another soundspeed field available as a Compute field
        // unifying the patchdata and the ghosts is really needed ...

        bool is_varying_alpha
            = bool(std::get_if<VaryingMM97>(&config)) || bool(std::get_if<VaryingCD10>(&config));
        return is_varying_alpha;
    }

    /**
     * @brief Prints the status of the artificial viscosity configuration.
     */
    inline void print_status() {
        logger::raw_ln("--- artificial viscosity config");

        if (None *v = std::get_if<None>(&config)) {
            logger::raw_ln("  Config Type : None (No artificial viscosity)");
        } else if (Constant *v = std::get_if<Constant>(&config)) {
            logger::raw_ln("  Config Type : Constant (Constant artificial viscosity)");
            logger::raw_ln("  alpha_u  =", v->alpha_u);
            logger::raw_ln("  alpha_AV =", v->alpha_AV);
            logger::raw_ln("  beta_AV  =", v->beta_AV);
        } else if (VaryingMM97 *v = std::get_if<VaryingMM97>(&config)) {
            logger::raw_ln("  Config Type : VaryingMM97 (Morris & Monaghan 1997)");
            logger::raw_ln("  alpha_min   =", v->alpha_min);
            logger::raw_ln("  alpha_max   =", v->alpha_max);
            logger::raw_ln("  sigma_decay =", v->sigma_decay);
            logger::raw_ln("  alpha_u     =", v->alpha_u);
            logger::raw_ln("  beta_AV     =", v->beta_AV);
        } else if (VaryingCD10 *v = std::get_if<VaryingCD10>(&config)) {
            logger::raw_ln("  Config Type : VaryingCD10 (Cullen & Dehnen 2010)");
            logger::raw_ln("  alpha_min   =", v->alpha_min);
            logger::raw_ln("  alpha_max   =", v->alpha_max);
            logger::raw_ln("  sigma_decay =", v->sigma_decay);
            logger::raw_ln("  alpha_u     =", v->alpha_u);
            logger::raw_ln("  beta_AV     =", v->beta_AV);
        } else if (ConstantDisc *v = std::get_if<ConstantDisc>(&config)) {
            logger::raw_ln("  Config Type : constant disc");
            logger::raw_ln("  alpha_AV   =", v->alpha_AV);
            logger::raw_ln("  alpha_u     =", v->alpha_u);
            logger::raw_ln("  beta_AV     =", v->beta_AV);
        } else {
            shambase::throw_unimplemented();
        }

        logger::raw_ln("--- artificial viscosity config (deduced)");

        logger::raw_ln("-------------");
    }

    inline std::optional<Tscal> get_alpha_u() {
        if (const Constant *v = std::get_if<Constant>(&config)) {
            return v->alpha_u;
        } else if (const VaryingMM97 *v = std::get_if<VaryingMM97>(&config)) {
            return v->alpha_u;
        } else if (const VaryingCD10 *v = std::get_if<VaryingCD10>(&config)) {
            return v->alpha_u;
        } else if (const ConstantDisc *v = std::get_if<ConstantDisc>(&config)) {
            return v->alpha_u;
        } else if (const None *v = std::get_if<None>(&config)) {
            return std::nullopt;
        } else {
            shambase::throw_unimplemented();
            return std::nullopt;
        }
    }
};

namespace shammodels::sph {

    template<class Tscal>
    inline void to_json(nlohmann::json &j, const AVConfig_None<Tscal> &p) {}

    template<class Tscal>
    inline void from_json(const nlohmann::json &j, AVConfig_None<Tscal> &p) {
        p = {};
    }

    template<class Tscal>
    inline void to_json(nlohmann::json &j, const AVConfig_Constant<Tscal> &p) {
        j = {
            {"alpha_u", p.alpha_u},
            {"alpha_AV", p.alpha_AV},
            {"beta_AV", p.beta_AV},
        };
    }

    template<class Tscal>
    inline void from_json(const nlohmann::json &j, AVConfig_Constant<Tscal> &p) {
        p = {
            j.at("alpha_u").get<Tscal>(),
            j.at("alpha_AV").get<Tscal>(),
            j.at("beta_AV").get<Tscal>(),
        };
    }

    template<class Tscal>
    inline void to_json(nlohmann::json &j, const AVConfig_VaryingMM97<Tscal> &p) {
        j = {
            {"alpha_min", p.alpha_min},
            {"alpha_max", p.alpha_max},
            {"sigma_decay", p.sigma_decay},
            {"alpha_u", p.alpha_u},
            {"beta_AV", p.beta_AV},
        };
    }

    template<class Tscal>
    inline void from_json(const nlohmann::json &j, AVConfig_VaryingMM97<Tscal> &p) {
        p = {
            j.at("alpha_min").get<Tscal>(),
            j.at("alpha_max").get<Tscal>(),
            j.at("sigma_decay").get<Tscal>(),
            j.at("alpha_u").get<Tscal>(),
            j.at("beta_AV").get<Tscal>(),
        };
    }

    template<class Tscal>
    inline void to_json(nlohmann::json &j, const AVConfig_VaryingCD10<Tscal> &p) {
        j = {
            {"alpha_min", p.alpha_min},
            {"alpha_max", p.alpha_max},
            {"sigma_decay", p.sigma_decay},
            {"alpha_u", p.alpha_u},
            {"beta_AV", p.beta_AV},
        };
    }

    template<class Tscal>
    inline void from_json(const nlohmann::json &j, AVConfig_VaryingCD10<Tscal> &p) {
        p = {
            j.at("alpha_min").get<Tscal>(),
            j.at("alpha_max").get<Tscal>(),
            j.at("sigma_decay").get<Tscal>(),
            j.at("alpha_u").get<Tscal>(),
            j.at("beta_AV").get<Tscal>(),
        };
    }

    template<class Tscal>
    inline void to_json(nlohmann::json &j, const AVConfig_ConstantDisc<Tscal> &p) {
        j = {
            {"alpha_AV", p.alpha_AV},
            {"alpha_u", p.alpha_u},
            {"beta_AV", p.beta_AV},
        };
    }

    template<class Tscal>
    inline void from_json(const nlohmann::json &j, AVConfig_ConstantDisc<Tscal> &p) {
        p = {
            j.at("alpha_AV").get<Tscal>(),
            j.at("alpha_u").get<Tscal>(),
            j.at("beta_AV").get<Tscal>(),
        };
    }

    /**
     * @brief Convert an AVConfig to a json object.
     *
     * @param j the json object to be filled
     * @param p the AVConfig object
     */
    template<class Tvec>
    inline void to_json(nlohmann::json &j, const AVConfig<Tvec> &p) {
        std::visit(
            [&](const auto &value) {
                j         = value;
                j["type"] = value.variant_type_name;
            },
            p.config);
    }

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

    /**
     * @brief Convert a json object to an AVConfig.
     *
     * @param j the json object to be used
     * @param p the AVConfig object to be filled
     */
    template<class Tvec>
    inline void from_json(const nlohmann::json &j, AVConfig<Tvec> &p) {
        using T = AVConfig<Tvec>;

        using Tscal = shambase::VecComponent<Tvec>;

        if (!j.contains("av_type")) {
            throw shambase::make_except_with_loc<std::runtime_error>(
                "no field av_type is found in this json");
        }

        std::string av_type;
        j.at("av_type").get_to(av_type);

        bool matched = on_variant_match(
            av_type,
            [&](auto tag) {
                using Talt = typename decltype(tag)::type;
                p.set(j.get<Talt>());
            },
            p.config);

        if (!matched) {
            std::vector<std::string> available_types;
            on_variant_cases(
                [&](auto tag) {
                    using Talt = typename decltype(tag)::type;
                    available_types.push_back(std::string(Talt::variant_type_name));
                },
                p.config);

            throw shambase::make_except_with_loc<std::runtime_error>(shambase::format(
                "unknown unknown type: {}\navailable types: {}\njson: {}",
                av_type,
                available_types,
                j.dump(4)));
        }
    }

} // namespace shammodels::sph
