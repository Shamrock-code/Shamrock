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
 * @file ForceFormulationConfig.hpp
 * @author Guo Yansong (guo.yansong.ngy@gmail.com)
 * @author Timothée David--Cléris (tim.shamrock@proton.me)
 * @author Yona Lapeyre (yona.lapeyre@ens-lyon.fr)
 * @brief Configuration for the GSPH momentum equation formulation
 *
 * GSPH implementations in the literature discretize the momentum equation in more
 * than one way. This selects between:
 * - ChaWhitworth: the symmetric SPH form (nabla_W/rho^2/Omega) with p* substituted
 *   for pressure, following Cha & Whitworth (2003). This is shamrock's default.
 * - InutsukaV2: the effective volume/face form (V2_ij * grad_W_ij) following
 *   Inutsuka (2002), reconstructed here with linear (1st order) face interpolation.
 */

#include "shambackends/type_traits.hpp"
#include "shambackends/vec.hpp"
#include "shamsys/legacy/log.hpp"
#include <nlohmann/json.hpp>
#include <variant>

namespace shammodels::gsph {

    template<class Tvec>
    struct ForceFormulationConfig;

} // namespace shammodels::gsph

template<class Tvec>
struct shammodels::gsph::ForceFormulationConfig {

    using Tscal              = shambase::VecComponent<Tvec>;
    static constexpr u32 dim = shambase::VectorProperties<Tvec>::dimension;

    /**
     * @brief Cha & Whitworth (2003) symmetric SPH momentum equation (default)
     */
    struct ChaWhitworth {};

    /**
     * @brief Inutsuka (2002) effective volume/face momentum equation
     *
     * Uses linear (1st order) interpolation of the volume element to build the
     * effective face between each particle pair (see math/reconstruction.hpp).
     */
    struct InutsukaV2 {};

    using Variant = std::variant<ChaWhitworth, InutsukaV2>;

    Variant config = ChaWhitworth{};

    void set(Variant v) { config = v; }

    void set_cha_whitworth() { set(ChaWhitworth{}); }

    void set_inutsuka_v2() { set(InutsukaV2{}); }

    inline bool is_cha_whitworth() const { return std::holds_alternative<ChaWhitworth>(config); }

    inline bool is_inutsuka_v2() const { return std::holds_alternative<InutsukaV2>(config); }

    inline void print_status() const {
        logger::raw_ln("--- Force formulation config");

        if (std::get_if<ChaWhitworth>(&config)) {
            logger::raw_ln("  Type : ChaWhitworth (2003) - symmetric SPH form");
        } else if (std::get_if<InutsukaV2>(&config)) {
            logger::raw_ln(
                "  Type : InutsukaV2 (2002) - effective volume/face, linear (1st order)");
        } else {
            shambase::throw_unimplemented();
        }

        logger::raw_ln("-------------");
    }
};

namespace shammodels::gsph {

    template<class Tvec>
    inline void to_json(nlohmann::json &j, const ForceFormulationConfig<Tvec> &p) {
        using T            = ForceFormulationConfig<Tvec>;
        using ChaWhitworth = typename T::ChaWhitworth;
        using InutsukaV2   = typename T::InutsukaV2;

        if (std::get_if<ChaWhitworth>(&p.config)) {
            j = {
                {"force_formulation", "cha_whitworth"},
            };
        } else if (std::get_if<InutsukaV2>(&p.config)) {
            j = {
                {"force_formulation", "inutsuka_v2"},
            };
        } else {
            shambase::throw_unimplemented();
        }
    }

    template<class Tvec>
    inline void from_json(const nlohmann::json &j, ForceFormulationConfig<Tvec> &p) {
        using T            = ForceFormulationConfig<Tvec>;
        using ChaWhitworth = typename T::ChaWhitworth;
        using InutsukaV2   = typename T::InutsukaV2;

        if (!j.contains("force_formulation")) {
            shambase::throw_with_loc<std::runtime_error>(
                "no field force_formulation is found in this json");
        }

        std::string force_formulation;
        j.at("force_formulation").get_to(force_formulation);

        if (force_formulation == "cha_whitworth") {
            p.set(ChaWhitworth{});
        } else if (force_formulation == "inutsuka_v2") {
            p.set(InutsukaV2{});
        } else {
            shambase::throw_unimplemented("Unknown force formulation type: " + force_formulation);
        }
    }

} // namespace shammodels::gsph
