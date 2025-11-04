// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright (c) 2021-2025 Timothée David--Cléris <tim.shamrock@proton.me>
// SPDX-License-Identifier: CeCILL Free Software License Agreement v2.1
// Shamrock is licensed under the CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

#pragma once

/**
 * @file MHDConfig.hpp
 * @author Timothée David--Cléris (tim.shamrock@proton.me)
 * @author Yona Lapeyre (yona.lapeyre@ens-lyon.fr) --no git blame--
 * @brief
 *
 */

#include "shambackends/vec.hpp"
#include "shamsys/legacy/log.hpp"
#include <nlohmann/json.hpp>
#include <variant>

namespace shammodels::sph {

    template<class Tvec>
    struct MHDConfig;
}

template<class Tvec>
struct shammodels::sph::MHDConfig {

    using Tscal              = shambase::VecComponent<Tvec>;
    static constexpr u32 dim = shambase::VectorProperties<Tvec>::dimension;

    struct None {};

    struct IdealMHD_constrained_hyper_para {
        Tscal sigma_mhd = 0.1;
        Tscal alpha_u   = 1.;
    };

    struct NonIdealMHD {
        Tscal sigma_mhd = 0.1;
        Tscal alpha_u   = 1.;
    };

    // how to set a new state of a variant as a dummy:
    // a) do everything right
    // b) forget to add the state to the variant
    //-> question your life choices
    using Variant = std::variant<None, IdealMHD_constrained_hyper_para, NonIdealMHD>;

    Variant config = None{};

    void set(Variant v) { config = v; }

    inline bool has_B_field() {
        bool is_B = bool(std::get_if<IdealMHD_constrained_hyper_para>(&config))
                    || bool(std::get_if<NonIdealMHD>(&config));
        return is_B;
    }

    inline bool has_psi_field() {
        bool is_psi = bool(std::get_if<IdealMHD_constrained_hyper_para>(&config))
                      || bool(std::get_if<NonIdealMHD>(&config));
        return is_psi;
    }

    inline bool has_divB_field() {
        bool is_divB = bool(std::get_if<IdealMHD_constrained_hyper_para>(&config));
        return is_divB;
    }

    inline bool has_curlB_field() {
        bool is_curlB = bool(std::get_if<NonIdealMHD>(&config));
        return is_curlB;
    }

    inline bool has_dtdivB_field() {
        bool is_dtdivB = bool(std::get_if<NonIdealMHD>(&config));
        return is_dtdivB;
    }

    inline void print_status() {
        logger::raw_ln("--- MHD config");

        if (None *v = std::get_if<None>(&config)) {
            logger::raw_ln("  Config MHD Type : None (No MHD)");
        } else if (
            IdealMHD_constrained_hyper_para *v
            = std::get_if<IdealMHD_constrained_hyper_para>(&config)) {
            logger::raw_ln("  Config MHD  : Ideal MHD, constrained hyperbolic/parabolic treatment");
            logger::raw_ln("  sigma_mhd  =", v->sigma_mhd);
        } else if (NonIdealMHD *v = std::get_if<NonIdealMHD>(&config)) {
            logger::raw_ln("  Config MHD Type : Non Ideal MHD");
            logger::raw_ln("  sigma_mhd   =", v->sigma_mhd);
        } else {
            shambase::throw_unimplemented();
        }

        logger::raw_ln("--- MHD config (deduced)");

        logger::raw_ln("-------------");
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
        using IMHD        = typename T::IdealMHD_constrained_hyper_para;
        using NonIdealMHD = typename T::NonIdealMHD;

        // Write the config type into the JSON object
        if (const None *v = std::get_if<None>(&p.config)) {
            j = {
                {"mhd_type", "none"},
            };
        } else if (const IMHD *v = std::get_if<IMHD>(&p.config)) {
            j = {
                {"mhd_type", "ideal_mhd_constrained_hyper_para"},
                {"sigma_mhd", v->sigma_mhd},
                {"alpha_u", v->alpha_u},
            };
        } else if (const NonIdealMHD *v = std::get_if<NonIdealMHD>(&p.config)) {
            // Write the shear base, direction, and speed into the JSON object
            j = {
                {"mhd_type", "non_ideal_mhd"},
                {"sigma_mhd", v->sigma_mhd},
                {"alpha_u", v->alpha_u},
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

        // Read the config type from the JSON object
        std::string mhd_type;
        j.at("mhd_type").get_to(mhd_type);

        using None        = typename T::None;
        using IMHD        = typename T::IdealMHD_constrained_hyper_para;
        using NonIdealMHD = typename T::NonIdealMHD;

        // Set the BCConfig based on the config type
        if (mhd_type == "none") {
            p.set(None{});
        } else if (mhd_type == "ideal_mhd_constrained_hyper_para") {
            p.set(
                IMHD{
                    j.at("sigma_mhd").get<Tscal>(),
                    j.at("alpha_u").get<Tscal>(),
                });
        } else if (mhd_type == "non_ideal_mhd") {
            p.set(
                NonIdealMHD{
                    j.at("sigma_mhd").get<Tscal>(),
                    j.at("alpha_u").get<Tscal>(),
                });
        } else {
            shambase::throw_unimplemented("wtf !");
        }
    }

} // namespace shammodels::sph
