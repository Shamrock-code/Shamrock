// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright(C) 2021-2023 Timothée David--Cléris <timothee.david--cleris@ens-lyon.fr>
// Licensed under CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

/**
 * @file RenderFieldGetter.cpp
 * @author Timothée David--Cléris (timothee.david--cleris@ens-lyon.fr)
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
                    logger::debug_ln("sph::vtk", "compute rho field for patch ", p.id_patch);
                    shamsys::instance::get_compute_queue().submit([&](sycl::handler &cgh) {
                        sycl::accessor acc_h{
                            shambase::get_check_ref(
                                pdat.get_field<Tscal>(pdat.pdl.get_field_idx<Tscal>("hpart"))
                                    .get_buf()),
                            cgh,
                            sycl::read_only};

                        sycl::accessor acc_rho{
                            shambase::get_check_ref(density.get_buf(p.id_patch)),
                            cgh,
                            sycl::write_only,
                            sycl::no_init};
                        const Tscal part_mass = solver_config.gpart_mass;

                        cgh.parallel_for(
                            sycl::range<1>{pdat.get_obj_cnt()}, [=](sycl::item<1> item) {
                                u32 gid = (u32) item.get_id();
                                using namespace shamrock::sph;
                                Tscal rho_ha = rho_h(part_mass, acc_h[gid], Kernel::hfactd);
                                acc_rho[gid] = rho_ha;
                            });
                    });
                });

                auto field_source_getter
                    = [&](const shamrock::patch::Patch cur_p, shamrock::patch::PatchData &pdat)
                    -> const std::unique_ptr<sycl::buffer<Tfield>> & {
                    return density.get_buf(cur_p.id_patch);
                };

                return lambda(field_source_getter);
            }
        }

        auto field_source_getter =
            [&](const shamrock::patch::Patch cur_p,
                shamrock::patch::PatchData &pdat) -> const std::unique_ptr<sycl::buffer<Tfield>> & {
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
