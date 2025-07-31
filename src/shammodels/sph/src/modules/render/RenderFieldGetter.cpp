// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright (c) 2021-2025 Timothée David--Cléris <tim.shamrock@proton.me>
// SPDX-License-Identifier: CeCILL Free Software License Agreement v2.1
// Shamrock is licensed under the CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

/**
 * @file RenderFieldGetter.cpp
 * @author Timothée David--Cléris (tim.shamrock@proton.me)
 * @brief
 *
 */

#include "shammodels/sph/modules/render/RenderFieldGetter.hpp"
#include "shammodels/sph/math/density.hpp"
#include "shamrock/scheduler/SchedulerUtility.hpp"

namespace shammodels::sph::modules {
    template<class Tvec, class Tfield, template<class> class SPHKernel>
    auto RenderFieldGetter<Tvec, Tfield, SPHKernel>::runner_function(
        std::string field_name, lamda_runner lambda) -> sham::DeviceBuffer<Tfield> {

        if constexpr (std::is_same_v<Tfield, f64>) {
            if (field_name == "rho" && std::is_same_v<Tscal, Tfield>) {
                using namespace shamrock;
                using namespace shamrock::patch;
                shamrock::SchedulerUtility utility(scheduler());
                shamrock::ComputeField<Tscal> density = utility.make_compute_field<Tscal>("rho", 1);

                scheduler().for_each_patchdata_nonempty([&](const Patch p, PatchData &pdat) {
                    shamlog_debug_ln("sph::vtk", "compute rho field for patch ", p.id_patch);

                    auto &buf_h
                        = pdat.get_field<Tscal>(pdat.pdl.get_field_idx<Tscal>("hpart")).get_buf();
                    auto &buf_rho = density.get_buf(p.id_patch);

                    sham::DeviceQueue &q = shamsys::instance::get_compute_scheduler().get_queue();

                    sham::EventList depends_list;

                    auto acc_h   = buf_h.get_read_access(depends_list);
                    auto acc_rho = buf_rho.get_write_access(depends_list);

                    auto e = q.submit(depends_list, [&](sycl::handler &cgh) {
                        const Tscal part_mass = solver_config.gpart_mass;

                        cgh.parallel_for(
                            sycl::range<1>{pdat.get_obj_cnt()}, [=](sycl::item<1> item) {
                                u32 gid = (u32) item.get_id();
                                using namespace shamrock::sph;
                                Tscal rho_ha = rho_h(part_mass, acc_h[gid], Kernel::hfactd);
                                acc_rho[gid] = rho_ha;
                            });
                    });

                    buf_h.complete_event_state(e);
                    buf_rho.complete_event_state(e);
                });

                auto field_source_getter
                    = [&](const shamrock::patch::Patch cur_p,
                          shamrock::patch::PatchData &pdat) -> const sham::DeviceBuffer<Tfield> & {
                    return density.get_buf(cur_p.id_patch);
                };

                return lambda(field_source_getter);
            }
        }

        auto field_source_getter
            = [&](const shamrock::patch::Patch cur_p,
                  shamrock::patch::PatchData &pdat) -> const sham::DeviceBuffer<Tfield> & {
            return pdat.get_field<Tfield>(pdat.pdl.get_field_idx<Tfield>(field_name)).get_buf();
        };

        return lambda(field_source_getter);
    }
} // namespace shammodels::sph::modules

using namespace shammath;
template class shammodels::sph::modules::RenderFieldGetter<f64_3, f64, M4>;
template class shammodels::sph::modules::RenderFieldGetter<f64_3, f64, M6>;
template class shammodels::sph::modules::RenderFieldGetter<f64_3, f64, M8>;

template class shammodels::sph::modules::RenderFieldGetter<f64_3, f64_3, M4>;
template class shammodels::sph::modules::RenderFieldGetter<f64_3, f64_3, M6>;
template class shammodels::sph::modules::RenderFieldGetter<f64_3, f64_3, M8>;
