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
 * @file RiemannConfig.hpp
 * @author Guo Yansong (guo.yansong.ngy@gmail.com)
 * @author Timothée David--Cléris (tim.shamrock@proton.me) --no git blame--
 * @author Yona Lapeyre (yona.lapeyre@ens-lyon.fr) --no git blame--
 * @brief Configuration for Riemann solvers in GSPH
 *
 * This file contains the configuration structures for different Riemann solver
 * types used in Godunov SPH (GSPH). The Riemann solver computes the interface
 * pressure (p*) and velocity (v*) at particle-particle interfaces.
 *
 * The GSPH method originated from:
 * - Inutsuka, S. (2002) "Reformulation of Smoothed Particle Hydrodynamics
 *   with Riemann Solver"
 */

#include "shambackends/type_traits.hpp"
#include "shambackends/vec.hpp"
#include "shamsys/legacy/log.hpp"
#include <nlohmann/json.hpp>
#include <variant>

namespace shammodels::gsph {

    /**
     * @brief Configuration for Riemann solvers in GSPH
     *
     * This struct contains the configuration for different Riemann solver types:
     * - Iterative: van Leer (1997) Newton-Raphson iterative solver
     * - Exact: Exact Riemann solver (Toro)
     * - HLLC: Harten-Lax-van Leer-Contact approximate solver
     * - Roe: Roe linearized solver
     *
     * @tparam Tvec type of the vector of coordinates
     */
    template<class Tvec>
    struct RiemannConfig;

} // namespace shammodels::gsph

template<class Tvec>
struct shammodels::gsph::RiemannConfig {

    using Tscal              = shambase::VecComponent<Tvec>;
    static constexpr u32 dim = shambase::VectorProperties<Tvec>::dimension;

    /**
     * @brief van Leer (1997) iterative Riemann solver
     *
     * Uses Newton-Raphson iteration to solve for the exact p* and v*.
     * Robust and accurate for most cases.
     * Reference: van Leer, B. (1997) "Towards the ultimate conservative difference scheme"
     */
    struct Iterative {
        Tscal tol    = Tscal{1.0e-6}; ///< Convergence tolerance
        u32 max_iter = 20;            ///< Maximum iterations
    };

    /**
     * @brief Exact Riemann solver
     *
     * Solves the Riemann problem exactly using iterative root finding.
     * Most accurate but computationally expensive.
     * Reference: Toro, E.F. (2009) "Riemann Solvers and Numerical Methods for Fluid Dynamics"
     */
    struct Exact {
        Tscal tol = Tscal{1.0e-8}; ///< Convergence tolerance
    };

    /**
     * @brief HLL approximate Riemann solver
     *
     * Harten-Lax-van Leer (1983) 2-wave approximate solver.
     * Uses only S_L and S_R wave speeds (no contact wave S*).
     * Good balance between accuracy and performance.
     * Reference: Harten, Lax & van Leer (1983)
     */
    struct HLL {};

    /**
     * @brief Roe linearized Riemann solver
     *
     * Uses Roe averaging for a linearized approximate solution.
     * Fast but may have issues with expansion shocks.
     * Reference: Roe, P.L. (1981)
     */
    struct Roe {
        Tscal entropy_fix = Tscal{0.1}; ///< Entropy fix parameter
    };

    using Variant = std::variant<Iterative, Exact, HLL, Roe>;

    Variant config = Iterative{};

    void set(Variant v) { config = v; }

    void set_iterative(Tscal tol = Tscal{1.0e-6}, u32 max_iter = 20) {
        set(Iterative{tol, max_iter});
    }

    void set_exact(Tscal tol = Tscal{1.0e-8}) { set(Exact{tol}); }

    void set_hll() { set(HLL{}); }

    void set_roe(Tscal entropy_fix = Tscal{0.1}) { set(Roe{entropy_fix}); }

    inline bool is_iterative() const { return std::holds_alternative<Iterative>(config); }
    inline bool is_exact() const { return std::holds_alternative<Exact>(config); }
    inline bool is_hll() const { return std::holds_alternative<HLL>(config); }
    inline bool is_roe() const { return std::holds_alternative<Roe>(config); }

    inline void print_status() const {
        logger::raw_ln("--- Riemann solver config");

        if (const Iterative *v = std::get_if<Iterative>(&config)) {
            logger::raw_ln("  Type     : Iterative (van Leer 1997)");
            logger::raw_ln("  tol      =", v->tol);
            logger::raw_ln("  max_iter =", v->max_iter);
        } else if (const Exact *v = std::get_if<Exact>(&config)) {
            logger::raw_ln("  Type : Exact (Toro)");
            logger::raw_ln("  tol  =", v->tol);
        } else if (std::get_if<HLL>(&config)) {
            logger::raw_ln("  Type : HLL");
        } else if (const Roe *v = std::get_if<Roe>(&config)) {
            logger::raw_ln("  Type        : Roe");
            logger::raw_ln("  entropy_fix =", v->entropy_fix);
        } else {
            shambase::throw_unimplemented();
        }

        logger::raw_ln("-------------");
    }
};

namespace shammodels::gsph {

    template<class Tvec>
    inline void to_json(nlohmann::json &j, const RiemannConfig<Tvec> &p) {
        using T         = RiemannConfig<Tvec>;
        using Iterative = typename T::Iterative;
        using Exact     = typename T::Exact;
        using HLL       = typename T::HLL;
        using Roe       = typename T::Roe;

        if (const Iterative *v = std::get_if<Iterative>(&p.config)) {
            j = {
                {"riemann_type", "iterative"},
                {"tol", v->tol},
                {"max_iter", v->max_iter},
            };
        } else if (const Exact *v = std::get_if<Exact>(&p.config)) {
            j = {
                {"riemann_type", "exact"},
                {"tol", v->tol},
            };
        } else if (std::get_if<HLL>(&p.config)) {
            j = {
                {"riemann_type", "hll"},
            };
        } else if (const Roe *v = std::get_if<Roe>(&p.config)) {
            j = {
                {"riemann_type", "roe"},
                {"entropy_fix", v->entropy_fix},
            };
        } else {
            shambase::throw_unimplemented();
        }
    }

    template<class Tvec>
    inline void from_json(const nlohmann::json &j, RiemannConfig<Tvec> &p) {
        using T     = RiemannConfig<Tvec>;
        using Tscal = shambase::VecComponent<Tvec>;

        if (!j.contains("riemann_type")) {
            shambase::throw_with_loc<std::runtime_error>(
                "no field riemann_type is found in this json");
        }

        std::string riemann_type;
        j.at("riemann_type").get_to(riemann_type);

        using Iterative = typename T::Iterative;
        using Exact     = typename T::Exact;
        using HLL       = typename T::HLL;
        using Roe       = typename T::Roe;

        if (riemann_type == "iterative") {
            p.set(Iterative{j.at("tol").get<Tscal>(), j.at("max_iter").get<u32>()});
        } else if (riemann_type == "exact") {
            p.set(Exact{j.at("tol").get<Tscal>()});
        } else if (riemann_type == "hll") {
            p.set(HLL{});
        } else if (riemann_type == "roe") {
            p.set(Roe{j.at("entropy_fix").get<Tscal>()});
        } else {
            shambase::throw_unimplemented("Unknown Riemann solver type: " + riemann_type);
        }
    }

} // namespace shammodels::gsph
