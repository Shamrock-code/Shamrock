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

#include "shambase/sycl_utils/vectorProperties.hpp"
#include "shamsys/legacy/log.hpp"
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

    using Variant  = std::variant<None, Constant, VaryingMM97, VaryingCD10>;
    Variant config = Constant{};

    void set(Variant v) { config = v; }

    void set_varying_cd10(
        Tscal alpha_min, Tscal alpha_max, Tscal sigma_decay, Tscal alpha_u, Tscal beta_AV) {
        set(VaryingCD10{alpha_min, alpha_max, sigma_decay, alpha_u, beta_AV});
    }

    inline bool has_alphaAV_field() {
        bool is_varying_alpha =
            bool(std::get_if<VaryingMM97>(&config)) || bool(std::get_if<VaryingCD10>(&config));
        return is_varying_alpha;
    }

    inline bool has_divv_field() {
        bool is_varying_alpha =
            bool(std::get_if<VaryingMM97>(&config)) || bool(std::get_if<VaryingCD10>(&config));
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

        bool is_varying_alpha =
            bool(std::get_if<VaryingMM97>(&config)) || bool(std::get_if<VaryingCD10>(&config));
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
        }

        logger::raw_ln("--- artificial viscosity config (deduced)");

        logger::raw_ln("-------------");
    }
};