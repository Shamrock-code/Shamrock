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
 * @file ReconstructConfig.hpp
 * @author Guo (guo.yansong.ngy@gmail.com) --no git blame--
 * @author Yona Lapeyre (yona.lapeyre@ens-lyon.fr) --no git blame--
 * @brief Configuration for reconstruction methods in GSPH
 *
 * This file contains the configuration structures for spatial reconstruction
 * methods used in Godunov SPH (GSPH). Reconstruction extrapolates primitive
 * variables to particle interfaces for higher-order accuracy.
 */

#include "shambackends/type_traits.hpp"
#include "shambackends/vec.hpp"
#include "shamsys/legacy/log.hpp"
#include <nlohmann/json.hpp>
#include <variant>

namespace shammodels::gsph {

    /**
     * @brief Configuration for reconstruction methods in GSPH
     *
     * This struct contains the configuration for different reconstruction types:
     * - PiecewiseConstant: 1st order, no reconstruction
     * - MUSCL: 2nd order MUSCL-Hancock with slope limiters
     *
     * @tparam Tvec type of the vector of coordinates
     */
    template<class Tvec>
    struct ReconstructConfig;

} // namespace shammodels::gsph

template<class Tvec>
struct shammodels::gsph::ReconstructConfig {

    using Tscal              = shambase::VecComponent<Tvec>;
    static constexpr u32 dim = shambase::VectorProperties<Tvec>::dimension;

    /**
     * @brief Slope limiter types for MUSCL reconstruction
     */
    enum class Limiter {
        VanLeer,  ///< van Leer limiter (smooth)
        Minmod,   ///< Minmod limiter (most diffusive)
        Superbee, ///< Superbee limiter (least diffusive)
        MC        ///< Monotonized Central limiter
    };

    /**
     * @brief Piecewise constant (1st order)
     *
     * No spatial reconstruction. Simple and robust but diffusive.
     * Equivalent to standard Godunov method.
     */
    struct PiecewiseConstant {};

    /**
     * @brief MUSCL reconstruction (2nd order)
     *
     * Monotone Upstream-centered Schemes for Conservation Laws.
     * Uses slope limiters to maintain TVD property.
     * Reference: van Leer (1979)
     */
    struct MUSCL {
        Limiter limiter = Limiter::VanLeer; ///< Slope limiter type
    };

    using Variant = std::variant<PiecewiseConstant, MUSCL>;

    Variant config = PiecewiseConstant{};

    void set(Variant v) { config = v; }

    void set_piecewise_constant() { set(PiecewiseConstant{}); }

    void set_muscl(Limiter limiter = Limiter::VanLeer) { set(MUSCL{limiter}); }

    inline bool is_piecewise_constant() const {
        return std::holds_alternative<PiecewiseConstant>(config);
    }

    inline bool is_muscl() const { return std::holds_alternative<MUSCL>(config); }

    inline bool requires_gradients() const { return is_muscl(); }

    inline void print_status() const {
        logger::raw_ln("--- Reconstruction config");

        if (std::get_if<PiecewiseConstant>(&config)) {
            logger::raw_ln("  Type : PiecewiseConstant (1st order)");
        } else if (const MUSCL *v = std::get_if<MUSCL>(&config)) {
            logger::raw_ln("  Type    : MUSCL (2nd order)");
            switch (v->limiter) {
            case Limiter::VanLeer : logger::raw_ln("  Limiter : VanLeer"); break;
            case Limiter::Minmod  : logger::raw_ln("  Limiter : Minmod"); break;
            case Limiter::Superbee: logger::raw_ln("  Limiter : Superbee"); break;
            case Limiter::MC      : logger::raw_ln("  Limiter : MC"); break;
            }
        } else {
            shambase::throw_unimplemented();
        }

        logger::raw_ln("-------------");
    }
};

namespace shammodels::gsph {

    template<class Tvec>
    inline void to_json(nlohmann::json &j, const ReconstructConfig<Tvec> &p) {
        using T                 = ReconstructConfig<Tvec>;
        using PiecewiseConstant = typename T::PiecewiseConstant;
        using MUSCL             = typename T::MUSCL;
        using Limiter           = typename T::Limiter;

        if (std::get_if<PiecewiseConstant>(&p.config)) {
            j = {
                {"reconstruct_type", "piecewise_constant"},
            };
        } else if (const MUSCL *v = std::get_if<MUSCL>(&p.config)) {
            std::string limiter_str;
            switch (v->limiter) {
            case Limiter::VanLeer : limiter_str = "vanleer"; break;
            case Limiter::Minmod  : limiter_str = "minmod"; break;
            case Limiter::Superbee: limiter_str = "superbee"; break;
            case Limiter::MC      : limiter_str = "mc"; break;
            }
            j = {
                {"reconstruct_type", "muscl"},
                {"limiter", limiter_str},
            };
        } else {
            shambase::throw_unimplemented();
        }
    }

    template<class Tvec>
    inline void from_json(const nlohmann::json &j, ReconstructConfig<Tvec> &p) {
        using T                 = ReconstructConfig<Tvec>;
        using PiecewiseConstant = typename T::PiecewiseConstant;
        using MUSCL             = typename T::MUSCL;
        using Limiter           = typename T::Limiter;

        if (!j.contains("reconstruct_type")) {
            shambase::throw_with_loc<std::runtime_error>(
                "no field reconstruct_type is found in this json");
        }

        std::string reconstruct_type;
        j.at("reconstruct_type").get_to(reconstruct_type);

        if (reconstruct_type == "piecewise_constant") {
            p.set(PiecewiseConstant{});
        } else if (reconstruct_type == "muscl") {
            std::string limiter_str;
            j.at("limiter").get_to(limiter_str);

            Limiter limiter;
            if (limiter_str == "vanleer") {
                limiter = Limiter::VanLeer;
            } else if (limiter_str == "minmod") {
                limiter = Limiter::Minmod;
            } else if (limiter_str == "superbee") {
                limiter = Limiter::Superbee;
            } else if (limiter_str == "mc") {
                limiter = Limiter::MC;
            } else {
                shambase::throw_unimplemented("Unknown limiter type: " + limiter_str);
            }

            p.set(MUSCL{limiter});
        } else {
            shambase::throw_unimplemented("Unknown reconstruction type: " + reconstruct_type);
        }
    }

} // namespace shammodels::gsph
