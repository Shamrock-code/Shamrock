// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright (c) 2021-2026 Timothée David--Cléris <tim.shamrock@proton.me>
// SPDX-License-Identifier: CeCILL Free Software License Agreement v2.1
// Shamrock is licensed under the CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

/**
 * @file RenderFieldGetter.cpp
 * @author Timothée David--Cléris (tim.shamrock@proton.me)
 * @author Yona Lapeyre (yona.lapeyre@ens-lyon.fr)
 * @brief
 *
 */

#include "shambase/DistributedData.hpp"
#include "shambase/exception.hpp"
#include "shambackends/DeviceBuffer.hpp"
#include "shammodels/sph/math/density.hpp"
#include "shammodels/sph/modules/render/RenderFieldGetter.hpp"
#include "shampylib/PatchDataToPy.hpp"
#include "shamrock/patch/PatchDataLayerLayout.hpp"
#include "shamrock/scheduler/SchedulerUtility.hpp"
#include "shamrock/solvergraph/Field.hpp"
#include <string>

namespace shammodels::sph::modules {

    template<class Tvec, class Tfield, template<class> class SPHKernel>
    shamrock::solvergraph::Field<Tfield> RenderFieldGetter<Tvec, Tfield, SPHKernel>::build_field(
        std::string field_name,
        std::optional<std::function<py::array_t<Tfield>(size_t, pybind11::dict &)>> custom_getter) {

        if (field_name != "custom" && custom_getter.has_value()) {
            throw shambase::make_except_with_loc<std::invalid_argument>(
                "custom_getter is only supported for the custom field");
        }

        shambase::DistributedData<u32> sizes{};

        using namespace shamrock;
        using namespace shamrock::patch;

        scheduler().for_each_patchdata_nonempty([&](const Patch p, PatchDataLayer &pdat) {
            sizes.add_obj(p.id_patch, pdat.get_obj_cnt());
        });

        auto make_field = [&](u32 nvar, std::string name, std::string texsymbol) {
            shamrock::solvergraph::Field<Tfield> ret
                = shamrock::solvergraph::Field<Tfield>(nvar, name, texsymbol);
            ret.ensure_sizes(sizes);
            return ret;
        };

        if constexpr (std::is_same_v<Tfield, f64>) {
            if (field_name == "rho" && std::is_same_v<Tscal, Tfield>) {

                auto density = make_field(1, "rho", "rho");

                scheduler().for_each_patchdata_nonempty([&](const Patch p, PatchDataLayer &pdat) {
                    shamlog_debug_ln("sph::vtk", "compute rho field for patch ", p.id_patch);

                    auto &buf_h
                        = pdat.get_field<Tscal>(pdat.pdl().get_field_idx<Tscal>("hpart")).get_buf();
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

                return density;
            } else if (field_name == "inv_hpart" && std::is_same_v<Tscal, Tfield>) {

                auto inv_hpart = make_field(1, "inv_hpart", "inv_hpart");

                scheduler().for_each_patchdata_nonempty([&](const Patch p, PatchDataLayer &pdat) {
                    shamlog_debug_ln("sph::vtk", "compute inv_hpart field for patch ", p.id_patch);

                    auto &buf_h
                        = pdat.get_field<Tscal>(pdat.pdl().get_field_idx<Tscal>("hpart")).get_buf();
                    auto &buf_inv_hpart = inv_hpart.get_buf(p.id_patch);

                    sham::DeviceQueue &q = shamsys::instance::get_compute_scheduler().get_queue();

                    sham::EventList depends_list;

                    auto acc_h         = buf_h.get_read_access(depends_list);
                    auto acc_inv_hpart = buf_inv_hpart.get_write_access(depends_list);

                    auto e = q.submit(depends_list, [&](sycl::handler &cgh) {
                        cgh.parallel_for(
                            sycl::range<1>{pdat.get_obj_cnt()}, [=](sycl::item<1> item) {
                                u32 gid = (u32) item.get_id();
                                using namespace shamrock::sph;
                                acc_inv_hpart[gid] = 1.0 / acc_h[gid];
                            });
                    });

                    buf_h.complete_event_state(e);
                    buf_inv_hpart.complete_event_state(e);
                });

                return inv_hpart;
            } else if (field_name == "unity" && std::is_same_v<Tscal, Tfield>) {
                using namespace shamrock;
                using namespace shamrock::patch;

                auto unity = make_field(1, "unity", "unity");
                sizes.for_each([&](u64 id_patch, u32 size) {
                    unity.get_buf(id_patch).fill(1);
                });

                return unity;
            } else if (field_name == "custom" && custom_getter.has_value()) {
                std::function<py::array_t<Tfield>(size_t, pybind11::dict &)> &field_source_getter
                    = custom_getter.value();

                auto custom = make_field(1, "custom", "custom");

                shambase::Timer timer;
                timer.start();

                scheduler().for_each_patchdata_nonempty([&](const Patch p, PatchDataLayer &pdat) {
                    shamlog_debug_ln("sph::vtk", "compute custom field for patch ", p.id_patch);

                    auto &buf_custom = custom.get_buf(p.id_patch);

                    sham::DeviceQueue &q = shamsys::instance::get_compute_scheduler().get_queue();

                    py::dict dic_out               = shamrock::pdat_to_dic(pdat);
                    std::vector<Tfield> acc_custom = buf_custom.copy_to_stdvec();

                    py::array_t<Tfield> custom_array
                        = field_source_getter(pdat.get_obj_cnt(), dic_out);

                    if (acc_custom.size() != custom_array.size()) {
                        throw shambase::make_except_with_loc<std::invalid_argument>(
                            "custom_array size does not match the number of particles");
                    }

                    acc_custom = custom_array.template cast<std::vector<Tfield>>();

                    buf_custom.copy_from_stdvec(acc_custom);
                });

                timer.end();

                f64 worse_time_rank = shamalgs::collective::allreduce_max(timer.elasped_sec());

                if (shamcomm::world_rank() == 0) {
                    logger::raw_ln(
                        "sph::RenderFieldGetter",
                        "compute custom field took : ",
                        worse_time_rank,
                        "s");
                }

                return custom;
            }
        }

        auto field_source_getter
            = [&](const shamrock::patch::Patch cur_p,
                  shamrock::patch::PatchDataLayer &pdat) -> const sham::DeviceBuffer<Tfield> & {
            return pdat.get_field<Tfield>(pdat.pdl().get_field_idx<Tfield>(field_name)).get_buf();
        };

        FieldDescriptor<Tfield> desc = scheduler().pdl_old().template get_field<Tfield>(field_name);
        u32 ifield = scheduler().pdl_old().template get_field_idx<Tfield>(field_name);

        if (desc.nvar > 1) {
            shambase::throw_unimplemented("this cannot handle cases with nvar > 1, yet ...");
        }

        auto ret = make_field(1, desc.name, desc.name);

        scheduler().for_each_patchdata_nonempty([&](const Patch p, PatchDataLayer &pdat) {
            sham::DeviceBuffer<Tfield> &buf = ret.get_buf(p.id_patch);
            buf.copy_from(
                pdat.get_field<Tfield>(pdat.pdl().get_field_idx<Tfield>(field_name)).get_buf());
        });

        return ret;
    }

    template<class Tvec, class Tfield, template<class> class SPHKernel>
    auto RenderFieldGetter<Tvec, Tfield, SPHKernel>::runner_function(
        std::string field_name,
        lamda_runner lambda,
        std::optional<std::function<py::array_t<Tfield>(size_t, pybind11::dict &)>> custom_getter)
        -> sham::DeviceBuffer<Tfield> {

        auto field = build_field(field_name, custom_getter);

        auto field_source_getter
            = [&](const shamrock::patch::Patch cur_p,
                  shamrock::patch::PatchDataLayer &pdat) -> const sham::DeviceBuffer<Tfield> & {
            return field.get_buf(cur_p.id_patch);
        };

        return lambda(field_source_getter);
    }
} // namespace shammodels::sph::modules

using namespace shammath;
template class shammodels::sph::modules::RenderFieldGetter<f64_3, f64, M4>;
template class shammodels::sph::modules::RenderFieldGetter<f64_3, f64, M6>;
template class shammodels::sph::modules::RenderFieldGetter<f64_3, f64, M8>;

template class shammodels::sph::modules::RenderFieldGetter<f64_3, f64, C2>;
template class shammodels::sph::modules::RenderFieldGetter<f64_3, f64, C4>;
template class shammodels::sph::modules::RenderFieldGetter<f64_3, f64, C6>;

template class shammodels::sph::modules::RenderFieldGetter<f64_3, f64_3, M4>;
template class shammodels::sph::modules::RenderFieldGetter<f64_3, f64_3, M6>;
template class shammodels::sph::modules::RenderFieldGetter<f64_3, f64_3, M8>;

template class shammodels::sph::modules::RenderFieldGetter<f64_3, f64_3, C2>;
template class shammodels::sph::modules::RenderFieldGetter<f64_3, f64_3, C4>;
template class shammodels::sph::modules::RenderFieldGetter<f64_3, f64_3, C6>;
