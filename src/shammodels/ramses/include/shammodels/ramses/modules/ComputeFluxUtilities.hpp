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
 * @file ComputeFluxUtilities.hpp
 * @author Léodasce Sewanou (leodasce.sewanou@ens-lyon.fr) --no git blame--
 * @author Timothée David--Cléris (tim.shamrock@proton.me)
 * @brief
 *
 */

#include "shambackends/sycl.hpp"
#include "shammath/riemann.hpp"
#include "shammath/riemann_dust.hpp"
#include "shammodels/ramses/Solver.hpp"

namespace shammodels::basegodunov::modules {

    using RiemannSolverMode     = shammodels::basegodunov::RiemannSolverMode;
    using DustRiemannSolverMode = shammodels::basegodunov::DustRiemannSolverMode;
    using Direction             = shammodels::basegodunov::modules::Direction;

    /**
     * @brief Unit normal vector of a face pointing in the given direction
     */
    template<class Tvec, Direction dir>
    inline constexpr Tvec dir_normal() {
        if constexpr (dir == Direction::xp) {
            return Tvec{1, 0, 0};
        } else if constexpr (dir == Direction::xm) {
            return Tvec{-1, 0, 0};
        } else if constexpr (dir == Direction::yp) {
            return Tvec{0, 1, 0};
        } else if constexpr (dir == Direction::ym) {
            return Tvec{0, -1, 0};
        } else if constexpr (dir == Direction::zp) {
            return Tvec{0, 0, 1};
        } else if constexpr (dir == Direction::zm) {
            return Tvec{0, 0, -1};
        } else {
            static_assert(shambase::always_false_v<decltype(dir)>, "non-exhaustive visitor!");
        }
        return Tvec{};
    }

    /**
     * @brief Dispatch to the gas Riemann solver selected by `mode`, for a face with unit
     *        normal along `dir`. The fluid state spec is supplied by the caller (see
     *        NodeComputeFlux.cpp) rather than constructed here.
     */
    template<shammath::FluidStateAdiabaticSpec FSpec, RiemannSolverMode mode, Direction dir>
    inline constexpr typename FSpec::Tcons riemann_flux(
        const FSpec &fspec,
        const typename FSpec::Tprim &primL,
        const typename FSpec::Tprim &primR) {
        const typename FSpec::Tvec n = dir_normal<typename FSpec::Tvec, dir>();

        if constexpr (mode == RiemannSolverMode::Rusanov) {
            return shammath::rusanov_flux(fspec, primL, primR, n);
        }
        if constexpr (mode == RiemannSolverMode::HLL) {
            return shammath::hll_flux(fspec, primL, primR, n);
        }
        if constexpr (mode == RiemannSolverMode::HLLC) {
            return shammath::hllc_adiab_toro_flux(fspec, primL, primR, n);
        }
    }

    /**
     * @brief Dispatch to the dust Riemann solver selected by `mode`, for a face with unit
     *        normal along `dir`. The dust fluid state spec is supplied by the caller (see
     *        NodeComputeFlux.cpp) rather than constructed here.
     */
    template<shammath::DustFluidStateSpec FSpec, DustRiemannSolverMode mode, Direction dir>
    inline constexpr typename FSpec::Tcons riemann_dust_flux(
        const FSpec &fspec,
        const typename FSpec::Tprim &primL,
        const typename FSpec::Tprim &primR) {
        const typename FSpec::Tvec n = dir_normal<typename FSpec::Tvec, dir>();

        if constexpr (mode == DustRiemannSolverMode::HB) {
            return shammath::huang_bai_flux(fspec, primL, primR, n);
        }
        if constexpr (mode == DustRiemannSolverMode::DHLL) {
            return shammath::d_hll_flux(fspec, primL, primR, n);
        }
    }

} // namespace shammodels::basegodunov::modules
