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
 * @file MHDConfig.hpp
 * @author Timothée David--Cléris (tim.shamrock@proton.me)
 * @author Yona Lapeyre (yona.lapeyre@ens-lyon.fr)
 * @brief
 *
 */

#include "shambase/exception.hpp"
#include "shambackends/vec.hpp"
#include "shammodels/common/config/enum_TensileCorrection.hpp"
#include "shamrock/experimental_features.hpp"
#include "shamsys/legacy/log.hpp"
#include <nlohmann/json.hpp>
#include <stdexcept>
#include <string>
#include <variant>

namespace shammodels::sph {

    template<class Tvec>
    struct MHDConfig;
} // namespace shammodels::sph

template<class Tvec>
struct shammodels::sph::MHDConfig {

    using Tscal              = shambase::VecComponent<Tvec>;
    static constexpr u32 dim = shambase::VectorProperties<Tvec>::dimension;

    struct None {};

    struct IdealMhdConstrainedHyperPara {
        Tscal sigma_mhd                = 0.1;
        Tscal alpha_B                  = 1.;
        TensileCorrection tensile_corr = TensileCorrection::Borve;
    };

    struct NonIdealMHD {
        Tscal sigma_mhd                = 0.1;
        Tscal alpha_B                  = 1.;
        Tscal etaO                     = 1.;
        Tscal etaH                     = 1.;
        Tscal etaAD                    = 1.;
        TensileCorrection tensile_corr = TensileCorrection::Borve;
    };

    // how to set a new state of a variant as a dummy:
    // a) do everything right
    // b) forget to add the state to the variant
    //-> question your life choices
    using Variant = std::variant<None, IdealMhdConstrainedHyperPara, NonIdealMHD>;

    Variant configMHD = None{};

    void set(Variant v) { configMHD = v; }

    inline bool do_nimhd() {
        bool is_NIMHD = bool(std::get_if<NonIdealMHD>(&configMHD));
        return is_NIMHD;
    }

    inline bool has_b_field() {
        bool is_B = bool(std::get_if<IdealMhdConstrainedHyperPara>(&configMHD))
                    || bool(std::get_if<NonIdealMHD>(&configMHD));
        return is_B;
    }

    inline bool has_psi_field() {
        bool is_psi = bool(std::get_if<IdealMhdConstrainedHyperPara>(&configMHD))
                      || bool(std::get_if<NonIdealMHD>(&configMHD));
        return is_psi;
    }

    inline bool has_div_b_field() { return has_b_field(); }

    inline bool has_curl_b_field() {
        bool is_curlB = bool(std::get_if<NonIdealMHD>(&configMHD));
        return is_curlB;
    }

    inline bool has_dtdiv_b_field() {
        bool is_dtdivB = bool(std::get_if<NonIdealMHD>(&configMHD));
        return is_dtdivB;
    }

    inline void print_status() {
        logger::raw_ln("--- MHD configMHD");

        if (None *v = std::get_if<None>(&configMHD)) {
            logger::raw_ln("  Config MHD Type : None (No MHD)");
        } else if (
            IdealMhdConstrainedHyperPara *v
            = std::get_if<IdealMhdConstrainedHyperPara>(&configMHD)) {
            logger::raw_ln("  Config MHD  : Ideal MHD, constrained hyperbolic/parabolic treatment");
            logger::raw_ln("  sigma_mhd    =", v->sigma_mhd);
            logger::raw_ln("  alpha_B      =", v->alpha_B);
            logger::raw_ln("  tensile_corr =", tensile_correction_to_string(v->tensile_corr));
        } else if (NonIdealMHD *v = std::get_if<NonIdealMHD>(&configMHD)) {
            logger::raw_ln("  Config MHD Type : Non Ideal MHD");
            logger::raw_ln("  sigma_mhd    =", v->sigma_mhd);
            logger::raw_ln("  alpha_B      =", v->alpha_B);
            logger::raw_ln("  etaO         =", v->etaO);
            logger::raw_ln("  etaH         =", v->etaH);
            logger::raw_ln("  etaAD        =", v->etaAD);
            logger::raw_ln("  tensile_corr =", tensile_correction_to_string(v->tensile_corr));
        } else {
            shambase::throw_unimplemented();
        }

        logger::raw_ln("--- MHD configMHD (deduced)");

        logger::raw_ln("-------------");
    }

    inline void check_config() {

        if (do_nimhd()) {

            if (!shamrock::are_experimental_features_allowed()) {
                shambase::throw_with_loc<std::runtime_error>("Non Ideal MHD is experimental");
            }
        }
    }
};

namespace shammodels::sph {

    /**
     * @brief Serialize a MHDConfig to a JSON object
     *
     * @param[out] j  The JSON object to write to
     * @param[in] p  The MHDConfig to serialize
     */
    template<class Tvec>
    inline void to_json(nlohmann::json &j, const MHDConfig<Tvec> &p) {
        using T = MHDConfig<Tvec>;

        using None        = typename T::None;
        using IMHD        = typename T::IdealMhdConstrainedHyperPara;
        using NonIdealMHD = typename T::NonIdealMHD;

        // Write the configMHD type into the JSON object
        if (const None *v = std::get_if<None>(&p.configMHD)) {
            j = {
                {"mhd_type", "none"},
            };
        } else if (const IMHD *v = std::get_if<IMHD>(&p.configMHD)) {
            j = {
                {"mhd_type", "ideal_mhd_constrained_hyper_para"},
                {"sigma_mhd", v->sigma_mhd},
                {"alpha_B", v->alpha_B},
                {"tensile_corr", tensile_correction_to_string(v->tensile_corr)},
            };
        } else if (const NonIdealMHD *v = std::get_if<NonIdealMHD>(&p.configMHD)) {
            // Write the shear base, direction, and speed into the JSON object
            j = {
                {"mhd_type", "non_ideal_mhd"},
                {"sigma_mhd", v->sigma_mhd},
                {"alpha_B", v->alpha_B},
                {"etaO", v->etaO},
                {"etaH", v->etaH},
                {"etaAD", v->etaAD},
                {"tensile_corr", tensile_correction_to_string(v->tensile_corr)},
            };
        } else {
            shambase::throw_unimplemented();
        }
    }

    /**
     * @brief Deserialize a JSON object into a MHDConfig
     *
     * @param[in] j  The JSON object to read from
     * @param[out] p The MHDConfig to deserialize
     */
    template<class Tvec>
    inline void from_json(const nlohmann::json &j, MHDConfig<Tvec> &p) {
        using T = MHDConfig<Tvec>;

        using Tscal = shambase::VecComponent<Tvec>;

        // Check if the JSON object contains the "mhd_type" field
        if (!j.contains("mhd_type")) {
            shambase::throw_with_loc<std::runtime_error>("no field mhd_type is found in this json");
        }

        // Read the configMHD type from the JSON object
        std::string mhd_type;
        j.at("mhd_type").get_to(mhd_type);

        using None        = typename T::None;
        using IMHD        = typename T::IdealMhdConstrainedHyperPara;
        using NonIdealMHD = typename T::NonIdealMHD;

        // older configs have no tensile_corr field, they used the Borve correction
        auto get_tensile_corr = [&](TensileCorrection def) {
            return (j.contains("tensile_corr"))
                       ? tensile_correction_from_string(j.at("tensile_corr").get<std::string>())
                       : def;
        };

        // Set the BCConfig based on the configMHD type
        if (mhd_type == "none") {
            p.set(None{});
        } else if (mhd_type == "ideal_mhd_constrained_hyper_para") {
            // alpha_u, alpha_AV & beta_AV are now in the artificial viscosity config, they are
            // ignored if present in older configs
            IMHD def{};
            p.set(
                IMHD{
                    j.at("sigma_mhd").get<Tscal>(),
                    j.value("alpha_B", def.alpha_B),
                    get_tensile_corr(def.tensile_corr),
                });
        } else if (mhd_type == "non_ideal_mhd") {
            // alpha_u, alpha_AV & beta_AV are now in the artificial viscosity config, they are
            // ignored if present in older configs
            NonIdealMHD def{};
            p.set(
                NonIdealMHD{
                    j.at("sigma_mhd").get<Tscal>(),
                    j.value("alpha_B", def.alpha_B),
                    j.value("etaO", def.etaO),
                    j.value("etaH", def.etaH),
                    j.value("etaAD", def.etaAD),
                    get_tensile_corr(def.tensile_corr),
                });
        } else {
            shambase::throw_unimplemented("wtf !");
        }
    }

} // namespace shammodels::sph
