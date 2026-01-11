// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright (c) 2021-2025 Timothee David--Cleris <tim.shamrock@proton.me>
// SPDX-License-Identifier: CeCILL Free Software License Agreement v2.1
// Shamrock is licensed under the CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

#pragma once

/**
 * @file RelativityTraits.hpp
 * @author Guo Yansong (guo.yansong.ngy@gmail.com)
 * @brief Relativity level traits for GSPH physics composition
 *
 * Defines Newtonian and SR physics with conserved variables and primitive recovery.
 */

#include "shambackends/sycl.hpp"
#include "shambackends/vec.hpp"
#include "shammodels/gsph/physics/SpacetimeTraits.hpp"

namespace shammodels::gsph::physics {

    /**
     * @brief Newtonian (classical) physics
     *
     * Conserved = Primitive (trivial inversion), signal speed = |v| + c_s
     *
     * @tparam Tvec Vector type
     * @tparam MatterT Matter model
     * @tparam SpacetimeT Spacetime (MinkowskiSpacetime for Newtonian)
     */
    template<class Tvec, class MatterT, class SpacetimeT = MinkowskiSpacetime<Tvec>>
    struct NewtonianPhysics {
        using Tscal              = shambase::VecComponent<Tvec>;
        static constexpr u32 dim = shambase::VectorProperties<Tvec>::dimension;

        using Matter    = MatterT;
        using Spacetime = SpacetimeT;

        static constexpr bool needs_primitive_recovery = false;
        static constexpr bool needs_lorentz_factor     = false;
        static constexpr bool needs_metric             = false;

        struct ConservedVars {
            Tscal rho;
            Tvec momentum;
            Tscal E;
        };

        template<class EOSType>
        SYCL_EXTERNAL static auto recover_primitives(
            const ConservedVars &U,
            const typename Matter::PrimitiveVars & /* W_old */,
            const EOSType &eos) -> typename Matter::PrimitiveVars {

            typename Matter::PrimitiveVars W;
            W.rho      = U.rho;
            W.velocity = U.momentum / U.rho;

            Tscal v2   = sycl::dot(W.velocity, W.velocity);
            W.uint     = U.E / U.rho - Tscal{0.5} * v2;
            W.pressure = eos.pressure_from_rho_u(W.rho, W.uint);

            return W;
        }

        SYCL_EXTERNAL static Tscal max_signal_speed(
            const typename Matter::PrimitiveVars &W, Tscal cs) {
            return sycl::length(W.velocity) + cs;
        }

        SYCL_EXTERNAL static ConservedVars to_conserved(const typename Matter::PrimitiveVars &W) {
            ConservedVars U;
            U.rho      = W.rho;
            U.momentum = W.rho * W.velocity;
            U.E        = W.rho * (W.uint + Tscal{0.5} * sycl::dot(W.velocity, W.velocity));
            return U;
        }
    };

    /**
     * @brief Special Relativistic physics (STUB)
     *
     * Conserved: D = rho W, S = rho h W^2 v, tau = rho h W^2 - P - D
     * Primitive recovery requires 1D root finding.
     *
     * @tparam Tvec Vector type
     * @tparam MatterT Matter model
     */
    template<class Tvec, class MatterT>
    struct SRPhysics {
        using Tscal              = shambase::VecComponent<Tvec>;
        static constexpr u32 dim = shambase::VectorProperties<Tvec>::dimension;

        using Matter    = MatterT;
        using Spacetime = MinkowskiSpacetime<Tvec>;

        static constexpr bool needs_primitive_recovery = true;
        static constexpr bool needs_lorentz_factor     = true;
        static constexpr bool needs_metric             = false;

        struct ConservedVars {
            Tscal D;
            Tvec S;
            Tscal tau;
        };

        SYCL_EXTERNAL static Tscal lorentz_factor(Tvec v) {
            Tscal v2 = sycl::dot(v, v);
            return Tscal{1} / sycl::sqrt(Tscal{1} - v2);
        }

        template<class EOSType>
        SYCL_EXTERNAL static auto recover_primitives(
            const ConservedVars &U,
            const typename Matter::PrimitiveVars &W_old,
            const EOSType &eos,
            u32 max_iter = 50,
            Tscal tol    = Tscal{1e-10}) -> typename Matter::PrimitiveVars {
            (void) U;
            (void) eos;
            (void) max_iter;
            (void) tol;
            return W_old; // STUB
        }

        SYCL_EXTERNAL static Tscal max_signal_speed(
            const typename Matter::PrimitiveVars &W, Tscal cs) {
            Tscal v = sycl::length(W.velocity);
            return (v + cs) / (Tscal{1} + v * cs);
        }
    };

} // namespace shammodels::gsph::physics
