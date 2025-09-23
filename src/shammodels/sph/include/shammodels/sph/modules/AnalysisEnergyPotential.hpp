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
 * @file AnalysisEnergyPotential.hpp
 * @author Timothée David--Cléris (tim.shamrock@proton.me)
 * @brief AnalysisEnergyPotential class with one method
 * AnalysisEnergyPotential.get_potential_energy()
 */

#include "shambase/memory.hpp"
#include "shamalgs/primitives/reduction.hpp"
#include "shambackends/DeviceQueue.hpp"
#include "shambackends/DeviceScheduler.hpp"
#include "shambackends/math.hpp"
#include "shammodels/sph/Model.hpp"
#include "shamrock/scheduler/PatchScheduler.hpp"
#include "shamrock/scheduler/ShamrockCtx.hpp"
#include <utility>

namespace shammodels::sph::modules {

    template<class Tvec, template<class> class SPHKernel>
    class AnalysisEnergyPotential {
        public:
        using Tscal              = shambase::VecComponent<Tvec>;
        static constexpr u32 dim = shambase::VectorProperties<Tvec>::dimension;

        using Solver = ::shammodels::sph::Solver<Tvec, SPHKernel>;

        Model<Tvec, SPHKernel> &model;
        Solver &solver;
        ShamrockCtx &ctx;

        AnalysisEnergyPotential(Model<Tvec, SPHKernel> &model)
            : model(model), ctx(model.ctx), solver(model.solver) {};

        auto get_potential_energy() -> Tscal {
            PatchScheduler &sched = shambase::get_check_ref(ctx.sched);
            auto dev_sched_ptr    = shamsys::instance::get_compute_scheduler_ptr();
            sham::DeviceQueue &q  = shambase::get_check_ref(dev_sched_ptr).get_queue();

            const u32 ixyz    = sched.pdl().template get_field_idx<Tvec>("xyz");
            const Tscal pmass = solver.solver_config.gpart_mass;

            Tscal epot = 0;

            struct GravSource {
                Tvec pos;
                Tscal mass;
            };

            std::vector<GravSource> grav_sources;

            if (!solver.storage.sinks.is_empty()) {
                for (const auto &sink : solver.storage.sinks.get()) {
                    grav_sources.push_back({sink.pos, sink.mass});
                }
            }

            using SolverConfigExtForce = typename Solver::Config::ExtForceConfig;
            using EF_PointMass         = typename SolverConfigExtForce::PointMass;
            using EF_LenseThirring     = typename SolverConfigExtForce::LenseThirring;
            using EF_ShearingBoxForce  = typename SolverConfigExtForce::ShearingBoxForce;

            for (const auto &var_force : solver.solver_config.ext_force_config.ext_forces) {
                if (const EF_PointMass *ext_force = std::get_if<EF_PointMass>(&var_force.val)) {
                    grav_sources.push_back({Tvec{}, ext_force->central_mass});
                } else if (
                    const EF_LenseThirring *ext_force
                    = std::get_if<EF_LenseThirring>(&var_force.val)) {
                    grav_sources.push_back({Tvec{}, ext_force->central_mass});
                }
            }

            if (!grav_sources.empty()) {

                using Tscal4 = sycl::vec<Tscal, 4>;
                std::vector<Tscal4> sources{};

                for (const auto &grav_source : grav_sources) {
                    sources.push_back(
                        {grav_source.pos.x(),
                         grav_source.pos.y(),
                         grav_source.pos.z(),
                         grav_source.mass});
                }

                sham::DeviceBuffer<Tscal4> sources_buf(sources.size(), dev_sched_ptr);
                sources_buf.copy_from_stdvec(sources);

                sched.for_each_patchdata_nonempty([&](const shamrock::patch::Patch p,
                                                      shamrock::patch::PatchDataLayer &pdat) {
                    u32 len = pdat.get_obj_cnt();

                    sham::DeviceBuffer<Tscal> epot_part(len, dev_sched_ptr);
                    sham::DeviceBuffer<Tvec> &xyz_buf = pdat.get_field_buf_ref<Tvec>(ixyz);

                    sham::kernel_call(
                        q,
                        sham::MultiRef{xyz_buf, sources_buf},
                        sham::MultiRef{epot_part},
                        len,
                        [pmass,
                         G            = solver.solver_config.get_constant_G(),
                         source_count = sources.size()](
                            u32 i,
                            const Tvec *__restrict xyz,
                            const Tscal4 *__restrict sources,
                            Tscal *__restrict epot_part) {
                            Tscal loc_epot = 0;

                            Tscal smass;
                            Tvec sink_pos;

                            for (u32 j = 0; j < source_count; ++j) {
                                Tscal4 source = sources[j];

                                smass    = source.w();
                                sink_pos = {source.x(), source.y(), source.z()};

                                loc_epot += -pmass * G * smass / sycl::length(xyz[i] - sink_pos);
                            }
                            epot_part[i] = loc_epot;
                        });

                    epot += shamalgs::primitives::sum(dev_sched_ptr, epot_part, 0, len);
                });
            }

            Tscal tot_epot = shamalgs::collective::allreduce_sum(epot);

            Tscal G = solver.solver_config.get_constant_G();

            for (size_t i = 0; i < grav_sources.size(); ++i) {
                for (size_t j = i + 1; j < grav_sources.size(); ++j) {
                    const auto &sink1 = grav_sources[i];
                    const auto &sink2 = grav_sources[j];

                    Tvec delta = sink1.pos - sink2.pos;
                    Tscal d    = sycl::length(delta);

                    tot_epot += -G * sink1.mass * sink2.mass * sham::inv_sat_positive(d, 1e-16);
                }
            }

            return tot_epot;
        }
    };

} // namespace shammodels::sph::modules
