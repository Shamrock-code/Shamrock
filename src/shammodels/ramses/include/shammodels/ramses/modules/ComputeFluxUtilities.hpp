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
#include <array>
#include <string>

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

        using Flux            = FluxCompute<Tvec, mode, dir>;
        std::string flux_name = " ";
        if (mode == RiemannSolverMode::HLL)
            flux_name = "hll flux";
        else if (mode == RiemannSolverMode::HLLC)
            flux_name = "hllc flux ";
        else if (mode == RiemannSolverMode::Rusanov)
            flux_name = "rusanov flux ";
        auto get_dir_name = [&]() {
            if constexpr (dir == Direction::xp) {
                return "xp";
            } else if constexpr (dir == Direction::xm) {
                return "xm";
            } else if constexpr (dir == Direction::yp) {
                return "yp";
            } else if constexpr (dir == Direction::ym) {
                return "ym";
            } else if constexpr (dir == Direction::zp) {
                return "zp";
            } else if constexpr (dir == Direction::zm) {
                return "zm";
            } else {
                static_assert(shambase::always_false_v<decltype(dir)>, "non-exhaustive visitor!");
            }
            return "";
        };
        std::string cur_direction = get_dir_name();
        std::string kernel_name   = (std::string) "compute " + flux_name + cur_direction;
        const char *_kernel_name  = kernel_name.c_str();

        sham::EventList depends_list;
        auto rho   = rho_face_dir.get_read_access(depends_list);
        auto vel   = vel_face_dir.get_read_access(depends_list);
        auto press = press_face_dir.get_read_access(depends_list);

        auto flux_rho  = flux_rho_face_dir.get_write_access(depends_list);
        auto flux_rhov = flux_rhov_face_dir.get_write_access(depends_list);
        auto flux_rhoe = flux_rhoe_face_dir.get_write_access(depends_list);

        auto e = q.submit(depends_list, [&, gamma](sycl::handler &cgh) {
            shambase::parallel_for(cgh, link_count, _kernel_name, [=](u32 id_a) {
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
        });

        rho_face_dir.complete_event_state(e);
        vel_face_dir.complete_event_state(e);
        press_face_dir.complete_event_state(e);

        flux_rho_face_dir.complete_event_state(e);
        flux_rhov_face_dir.complete_event_state(e);
        flux_rhoe_face_dir.complete_event_state(e);
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
        std::string flux_name
            = (mode == DustRiemannSolverMode::DHLL) ? "dust hll flux " : "dust huang-bai flux ";
        auto get_dir_name = [&]() {
            if constexpr (dir == Direction::xp) {
                return "xp";
            } else if constexpr (dir == Direction::xm) {
                return "xm";
            } else if constexpr (dir == Direction::yp) {
                return "yp";
            } else if constexpr (dir == Direction::ym) {
                return "ym";
            } else if constexpr (dir == Direction::zp) {
                return "zp";
            } else if constexpr (dir == Direction::zm) {
                return "zm";
            } else {
                static_assert(shambase::always_false_v<decltype(dir)>, "non-exhaustive visitor!");
            }
            return "";
        };
        std::string cur_direction = get_dir_name();
        std::string kernel_name   = (std::string) "compute " + flux_name + cur_direction;
        const char *_kernel_name  = kernel_name.c_str();

        sham::EventList depends_list;

        auto rho_dust = rho_dust_dir.get_read_access(depends_list);
        auto vel_dust = vel_dust_dir.get_read_access(depends_list);

        auto flux_rho_dust  = flux_rho_dust_dir.get_write_access(depends_list);
        auto flux_rhov_dust = flux_rhov_dust_dir.get_write_access(depends_list);

        auto e = q.submit(depends_list, [&](sycl::handler &cgh) {
            u32 ndust = nvar;
            shambase::parallel_for(cgh, link_count * nvar, _kernel_name, [=](u32 id_var_a) {
                auto rho_ij = rho_dust[id_var_a];
                auto vel_ij = vel_dust[id_var_a];

                using Tprim = shammath::DustPrimState<Tvec>;
                auto flux_dust_dir
                    = d_Flux::dustflux(Tprim{rho_ij[0], vel_ij[0]}, Tprim{rho_ij[1], vel_ij[1]});

                flux_rho_dust[id_var_a]  = flux_dust_dir.rho;
                flux_rhov_dust[id_var_a] = flux_dust_dir.rhovel;
            });
        });

        rho_dust_dir.complete_event_state(e);
        vel_dust_dir.complete_event_state(e);

        flux_rho_dust_dir.complete_event_state(e);
        flux_rhov_dust_dir.complete_event_state(e);
    }

} // namespace shammodels::basegodunov::modules
