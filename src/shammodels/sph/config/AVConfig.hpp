// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright(C) 2021-2023 Timothée David--Cléris <timothee.david--cleris@ens-lyon.fr>
// Licensed under CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

#pragma once

/**
 * @file AVConfig.hpp
 * @author Timothée David--Cléris (timothee.david--cleris@ens-lyon.fr)
 * @brief
 *
 */

#include "shambackends/type_traits.hpp"
#include "shambackends/vec.hpp"
#include "shamsys/legacy/log.hpp"
#include <nlohmann/json.hpp>
#include <variant>

namespace shammodels::sph {

    template<class Tvec>
    struct AVConfig;
}

template<class Tvec>
struct shammodels::sph::AVConfig {

    using Tscal              = shambase::VecComponent<Tvec>;
    static constexpr u32 dim = shambase::VectorProperties<Tvec>::dimension;
    /**
     * @brief cf Price 2018 , q^a_ab = 0
     */
    struct None {};

    struct Constant {
        Tscal alpha_u  = 1.0;
        Tscal alpha_AV = 1.0;
        Tscal beta_AV  = 2.0;
    };

    /**
     * @brief Morris & Monaghan 1997
     *
     */
    struct VaryingMM97 {
        Tscal alpha_min   = 0.1;
        Tscal alpha_max   = 1.0;
        Tscal sigma_decay = 0.1;
        Tscal alpha_u     = 1.0;
        Tscal beta_AV     = 2.0;
    };

    /**
     * @brief Cullen & Dehnen 2010
     *
     */
    struct VaryingCD10 {
        Tscal alpha_min   = 0.1;
        Tscal alpha_max   = 1.0;
        Tscal sigma_decay = 0.1;
        Tscal alpha_u     = 1.0;
        Tscal beta_AV     = 2.0;
    };

    struct ConstantDisc {
        Tscal alpha_AV = 1.0;
        Tscal alpha_u  = 1.0;
        Tscal beta_AV  = 2.0;
    };

    using Variant = std::variant<None, Constant, VaryingMM97, VaryingCD10, ConstantDisc>;

    Variant config = Constant{};

    void set(Variant v) { config = v; }

    void set_varying_cd10(
        Tscal alpha_min, Tscal alpha_max, Tscal sigma_decay, Tscal alpha_u, Tscal beta_AV) {
        set(VaryingCD10{alpha_min, alpha_max, sigma_decay, alpha_u, beta_AV});
    }

    inline bool has_alphaAV_field() {
        bool is_varying_alpha
            = bool(std::get_if<VaryingMM97>(&config)) || bool(std::get_if<VaryingCD10>(&config));
        return is_varying_alpha;
    }

    inline bool has_divv_field() {
        bool is_varying_alpha
            = bool(std::get_if<VaryingMM97>(&config)) || bool(std::get_if<VaryingCD10>(&config));
        return is_varying_alpha;
    }
    inline bool has_curlv_field() {
        bool is_varying_alpha = bool(std::get_if<VaryingCD10>(&config));
        return is_varying_alpha;
    }
    inline bool has_dtdivv_field() {
        bool is_varying_alpha = bool(std::get_if<VaryingCD10>(&config));
        return is_varying_alpha;
    }

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
};

namespace shammodels::sph {
    template<class Tvec>
    inline void to_json(nlohmann::json &j, const AVConfig<Tvec> &p) {
        using T = AVConfig<Tvec>;

        using None         = typename T::None;
        using Constant     = typename T::Constant;
        using VaryingMM97  = typename T::VaryingMM97;
        using VaryingCD10  = typename T::VaryingCD10;
        using ConstantDisc = typename T::ConstantDisc;

        if (const None *v = std::get_if<None>(&p.config)) {
            j = {
                {"av_type", "none"},
            };
        } else if (const Constant *v = std::get_if<Constant>(&p.config)) {
            j = {
                {"av_type", "constant"},
                {"alpha_u", v->alpha_u},
                {"alpha_AV", v->alpha_AV},
                {"beta_AV", v->beta_AV},
            };
        } else if (const VaryingMM97 *v = std::get_if<VaryingMM97>(&p.config)) {
            j = {
                {"av_type", "varying_mm97"},
                {"alpha_min", v->alpha_min},
                {"alpha_max", v->alpha_max},
                {"sigma_decay", v->sigma_decay},
                {"alpha_u", v->alpha_u},
                {"beta_AV", v->beta_AV},
            };
        } else if (const VaryingCD10 *v = std::get_if<VaryingCD10>(&p.config)) {
            j = {
                {"av_type", "varying_cd10"},
                {"alpha_min", v->alpha_min},
                {"alpha_max", v->alpha_max},
                {"sigma_decay", v->sigma_decay},
                {"alpha_u", v->alpha_u},
                {"beta_AV", v->beta_AV},
            };
        } else if (const ConstantDisc *v = std::get_if<ConstantDisc>(&p.config)) {
            j = {
                {"av_type", "constant_disc"},
                {"alpha_u", v->alpha_u},
                {"alpha_AV", v->alpha_AV},
                {"beta_AV", v->beta_AV},
            };
        } else {
            shambase::throw_unimplemented();
        }
    }

    template<class Tvec>
    inline void from_json(const nlohmann::json &j, AVConfig<Tvec> &p) {
        using T = AVConfig<Tvec>;

        using Tscal = shambase::VecComponent<Tvec>;

        if (!j.contains("av_type")) {
            shambase::throw_with_loc<std::runtime_error>("no field eos_type is found in this json");
        }

        std::string av_type;
        j.at("av_type").get_to(av_type);

        using None         = typename T::None;
        using Constant     = typename T::Constant;
        using VaryingMM97  = typename T::VaryingMM97;
        using VaryingCD10  = typename T::VaryingCD10;
        using ConstantDisc = typename T::ConstantDisc;

        if (av_type == "none") {
            p.set(None{});
        } else if (av_type == "constant") {
            p.set(Constant{
                j.at("alpha_u").get<Tscal>(),
                j.at("alpha_AV").get<Tscal>(),
                j.at("beta_AV").get<Tscal>()});
        } else if (av_type == "varying_mm97") {
            p.set(VaryingMM97{
                j.at("alpha_min").get<Tscal>(),
                j.at("alpha_max").get<Tscal>(),
                j.at("sigma_decay").get<Tscal>(),
                j.at("alpha_u").get<Tscal>(),
                j.at("beta_AV").get<Tscal>()});
        } else if (av_type == "varying_cd10") {
            p.set(VaryingCD10{
                j.at("alpha_min").get<Tscal>(),
                j.at("alpha_max").get<Tscal>(),
                j.at("sigma_decay").get<Tscal>(),
                j.at("alpha_u").get<Tscal>(),
                j.at("beta_AV").get<Tscal>()});
        } else if (av_type == "constant_disc") {
            p.set(ConstantDisc{
                j.at("alpha_AV").get<Tscal>(),
                j.at("alpha_u").get<Tscal>(),
                j.at("beta_AV").get<Tscal>()});
        } else {
            shambase::throw_unimplemented("wtf !");
        }
    }
} // namespace shammodels::sph
