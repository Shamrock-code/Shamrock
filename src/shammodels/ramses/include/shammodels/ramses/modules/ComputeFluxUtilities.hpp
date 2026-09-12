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

#include "shambackends/kernel_call.hpp"
#include "shambackends/sycl.hpp"
#include "shammath/riemann.hpp"
#include "shammath/riemann_dust.hpp"
#include "shammodels/ramses/Solver.hpp"
#include <array>

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

    template<class Tvec, RiemannSolverMode mode, Direction dir>
    class FluxCompute {
        public:
        using Tcons = shammath::ConsState<Tvec>;
        using Tprim = shammath::PrimState<Tvec>;
        using Tscal = typename Tcons::Tscal;

        inline static constexpr Tcons flux(Tprim pL, Tprim pR, typename Tcons::Tscal gamma) {
            const Tvec n = dir_normal<Tvec, dir>();

            if constexpr (mode == RiemannSolverMode::Rusanov) {
                return shammath::rusanov_flux(pL, pR, gamma, n);
            }
            if constexpr (mode == RiemannSolverMode::HLL) {
                return shammath::hll_flux(pL, pR, gamma, n);
            }
            if constexpr (mode == RiemannSolverMode::HLLC) {
                return shammath::hllc_adiab_toro_flux(pL, pR, gamma, n);
            }
        }
    };

    template<class Tvec, DustRiemannSolverMode mode, Direction dir>
    class DustFluxCompute {
        public:
        using Tcons = shammath::DustConsState<Tvec>;
        using Tprim = shammath::DustPrimState<Tvec>;
        using Tscal = typename Tcons::Tscal;

        inline static constexpr Tcons dustflux(Tprim pL, Tprim pR) {
            const Tvec n = dir_normal<Tvec, dir>();

            if constexpr (mode == DustRiemannSolverMode::HB) {
                return shammath::huang_bai_flux(pL, pR, n);
            }
            if constexpr (mode == DustRiemannSolverMode::DHLL) {
                return shammath::d_hll_flux(pL, pR, n);
            }
        }
    };

    template<RiemannSolverMode mode, class Tvec, class Tscal, Direction dir>
    void compute_fluxes_dir(
        sham::DeviceQueue &q,
        u32 link_count,
        sham::DeviceBuffer<std::array<Tscal, 2>> &rho_face_dir,
        sham::DeviceBuffer<std::array<Tvec, 2>> &vel_face_dir,
        sham::DeviceBuffer<std::array<Tscal, 2>> &press_face_dir,
        sham::DeviceBuffer<Tscal> &flux_rho_face_dir,
        sham::DeviceBuffer<Tvec> &flux_rhov_face_dir,
        sham::DeviceBuffer<Tscal> &flux_rhoe_face_dir,
        Tscal gamma) {

        using Flux = FluxCompute<Tvec, mode, dir>;

        sham::kernel_call(
            q,
            sham::MultiRef{rho_face_dir, vel_face_dir, press_face_dir},
            sham::MultiRef{flux_rho_face_dir, flux_rhov_face_dir, flux_rhoe_face_dir},
            link_count,
            [gamma](
                u32 id_a,
                const std::array<Tscal, 2> *__restrict rho,
                const std::array<Tvec, 2> *__restrict vel,
                const std::array<Tscal, 2> *__restrict press,
                Tscal *__restrict flux_rho,
                Tvec *__restrict flux_rhov,
                Tscal *__restrict flux_rhoe) {
                auto rho_ij   = rho[id_a];
                auto vel_ij   = vel[id_a];
                auto press_ij = press[id_a];

                using Tprim   = shammath::PrimState<Tvec>;
                auto flux_dir = Flux::flux(
                    Tprim{rho_ij[0], press_ij[0], vel_ij[0]},
                    Tprim{rho_ij[1], press_ij[1], vel_ij[1]},
                    gamma);

                flux_rho[id_a]  = flux_dir.rho;
                flux_rhov[id_a] = flux_dir.rhovel;
                flux_rhoe[id_a] = flux_dir.rhoe;
            });
    }

    template<DustRiemannSolverMode mode, class Tvec, class Tscal, Direction dir>
    void dust_compute_fluxes_dir(
        sham::DeviceQueue &q,
        u32 link_count,
        sham::DeviceBuffer<std::array<Tscal, 2>> &rho_dust_dir,
        sham::DeviceBuffer<std::array<Tvec, 2>> &vel_dust_dir,
        sham::DeviceBuffer<Tscal> &flux_rho_dust_dir,
        sham::DeviceBuffer<Tvec> &flux_rhov_dust_dir,
        u32 nvar) {

        using d_Flux = DustFluxCompute<Tvec, mode, dir>;

        sham::kernel_call(
            q,
            sham::MultiRef{rho_dust_dir, vel_dust_dir},
            sham::MultiRef{flux_rho_dust_dir, flux_rhov_dust_dir},
            link_count * nvar,
            [](u32 id_var_a,
               const std::array<Tscal, 2> *__restrict rho_dust,
               const std::array<Tvec, 2> *__restrict vel_dust,
               Tscal *__restrict flux_rho_dust,
               Tvec *__restrict flux_rhov_dust) {
                auto rho_ij = rho_dust[id_var_a];
                auto vel_ij = vel_dust[id_var_a];

                using Tprim = shammath::DustPrimState<Tvec>;
                auto flux_dust_dir
                    = d_Flux::dustflux(Tprim{rho_ij[0], vel_ij[0]}, Tprim{rho_ij[1], vel_ij[1]});

                flux_rho_dust[id_var_a]  = flux_dust_dir.rho;
                flux_rhov_dust[id_var_a] = flux_dust_dir.rhovel;
            });
    }

} // namespace shammodels::basegodunov::modules
