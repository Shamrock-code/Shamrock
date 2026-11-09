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
 * @file AnalysisPlasmaBeta.hpp
 * @author Yona Lapeyre (yona.lapeyre@ens-lyon.fr)
 * @brief Volume averaged plasma beta, thermal and magnetic energies of an MHD SPH simulation
 */

#include "shambase/exception.hpp"
#include "shambase/memory.hpp"
#include "shamalgs/collective/reduction.hpp"
#include "shamalgs/primitives/reduction.hpp"
#include "shambackends/DeviceQueue.hpp"
#include "shambackends/DeviceScheduler.hpp"
#include "shammodels/sph/Model.hpp"
#include "shammodels/sph/math/density.hpp"
#include "shamrock/scheduler/PatchScheduler.hpp"
#include "shamrock/scheduler/ShamrockCtx.hpp"
#include <limits>
#include <stdexcept>

namespace shammodels::sph::modules {

    /**
     * @brief Thermal and magnetic energies and the volume averaged plasma beta
     *
     * With the SPH density rho = m (hfact / h)^3 and dV = m / rho:
     * \f[
     *   E_{\rm th} = \sum_a m u_a, \qquad
     *   E_{\rm mag} = \sum_a m \frac{|B_a/\rho_a|^2 \rho_a}{2 \mu_0}, \qquad
     *   \beta = \frac{\int P\, dV}{\int B^2 / (2\mu_0)\, dV} = \frac{(\gamma - 1) E_{\rm th}}
     *   {E_{\rm mag}}
     * \f]
     * All the sums are reduced over the patches and the MPI ranks, every rank gets the result.
     */
    template<class Tvec, template<class> class SPHKernel>
    class AnalysisPlasmaBeta {
        public:
        using Tscal              = shambase::VecComponent<Tvec>;
        static constexpr u32 dim = shambase::VectorProperties<Tvec>::dimension;
        using Kernel             = SPHKernel<Tscal>;

        using Solver = ::shammodels::sph::Solver<Tvec, SPHKernel>;

        Model<Tvec, SPHKernel> &model;
        Solver &solver;
        ShamrockCtx &ctx;

        AnalysisPlasmaBeta(Model<Tvec, SPHKernel> &model)
            : model(model), ctx(model.ctx), solver(model.solver) {};

        /// sum of m u over all the particles (needs the uint field)
        auto get_thermal_energy() -> Tscal {
            if (!solver.solver_config.has_field_uint()) {
                throw shambase::make_except_with_loc<std::invalid_argument>(
                    "AnalysisPlasmaBeta: the thermal energy needs the uint field");
            }

            PatchScheduler &sched = shambase::get_check_ref(ctx.sched);
            auto dev_sched_ptr    = shamsys::instance::get_compute_scheduler_ptr();

            const u32 iuint   = sched.pdl_old().template get_field_idx<Tscal>("uint");
            const Tscal pmass = solver.solver_config.gpart_mass;

            Tscal eth = 0;

            sched.for_each_patchdata_nonempty(
                [&](const shamrock::patch::Patch p, shamrock::patch::PatchDataLayer &pdat) {
                    u32 len                             = pdat.get_obj_cnt();
                    sham::DeviceBuffer<Tscal> &uint_buf = pdat.get_field_buf_ref<Tscal>(iuint);
                    eth += pmass * shamalgs::primitives::sum(dev_sched_ptr, uint_buf, 0, len);
                });

            return shamalgs::collective::allreduce_sum(eth);
        }

        /// sum of m B^2 / (2 mu_0 rho) over all the particles, with the SPH density
        auto get_magnetic_energy() -> Tscal {
            if (!solver.solver_config.has_field_b_on_rho()) {
                throw shambase::make_except_with_loc<std::invalid_argument>(
                    "AnalysisPlasmaBeta: the magnetic energy needs the B/rho field (MHD)");
            }

            PatchScheduler &sched = shambase::get_check_ref(ctx.sched);
            auto dev_sched_ptr    = shamsys::instance::get_compute_scheduler_ptr();
            sham::DeviceQueue &q  = shambase::get_check_ref(dev_sched_ptr).get_queue();

            const u32 ib_on_rho = sched.pdl_old().template get_field_idx<Tvec>("B/rho");
            const u32 ihpart    = sched.pdl_old().template get_field_idx<Tscal>("hpart");
            const Tscal pmass   = solver.solver_config.gpart_mass;
            const Tscal mu_0    = solver.solver_config.get_constant_mu_0();

            Tscal emag = 0;

            sched.for_each_patchdata_nonempty([&](const shamrock::patch::Patch p,
                                                  shamrock::patch::PatchDataLayer &pdat) {
                u32 len = pdat.get_obj_cnt();

                sham::DeviceBuffer<Tscal> emag_part(len, dev_sched_ptr);
                sham::DeviceBuffer<Tvec> &b_on_rho_buf = pdat.get_field_buf_ref<Tvec>(ib_on_rho);
                sham::DeviceBuffer<Tscal> &hpart_buf   = pdat.get_field_buf_ref<Tscal>(ihpart);

                sham::kernel_call(
                    q,
                    sham::MultiRef{b_on_rho_buf, hpart_buf},
                    sham::MultiRef{emag_part},
                    len,
                    [pmass, mu_0](
                        u32 i,
                        const Tvec *__restrict b_on_rho,
                        const Tscal *__restrict hpart,
                        Tscal *__restrict emag_part) {
                        Tscal rho = shamrock::sph::rho_h(pmass, hpart[i], Kernel::hfactd);
                        // m B^2 / (2 mu_0 rho) = m |B/rho|^2 rho / (2 mu_0)
                        emag_part[i]
                            = pmass * sham::dot(b_on_rho[i], b_on_rho[i]) * rho / (2 * mu_0);
                    });

                emag += shamalgs::primitives::sum(dev_sched_ptr, emag_part, 0, len);
            });

            return shamalgs::collective::allreduce_sum(emag);
        }

        /// (gamma - 1) E_th / E_mag, only for an adiabatic EOS (inf without magnetic field)
        auto get_plasma_beta() -> Tscal {
            using Adiabatic = typename Solver::Config::EOSConfig::Adiabatic;

            const Adiabatic *eos = std::get_if<Adiabatic>(&solver.solver_config.eos_config.config);
            if (eos == nullptr) {
                throw shambase::make_except_with_loc<std::invalid_argument>(
                    "AnalysisPlasmaBeta: the plasma beta is only implemented for an adiabatic "
                    "EOS");
            }

            Tscal eth  = get_thermal_energy();
            Tscal emag = get_magnetic_energy();

            if (!(emag > 0)) {
                return std::numeric_limits<Tscal>::infinity();
            }
            return (eos->gamma - 1) * eth / emag;
        }
    };
} // namespace shammodels::sph::modules
