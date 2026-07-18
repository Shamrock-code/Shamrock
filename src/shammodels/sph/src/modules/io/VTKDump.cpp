// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright (c) 2021-2026 Timothée David--Cléris <tim.shamrock@proton.me>
// SPDX-License-Identifier: CeCILL Free Software License Agreement v2.1
// Shamrock is licensed under the CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

/**
 * @file VTKDump.cpp
 * @author Guo Yansong (guo.yansong.ngy@gmail.com)
 * @author Timothée David--Cléris (tim.shamrock@proton.me)
 * @author Yona Lapeyre (yona.lapeyre@ens-lyon.fr)
 * @brief
 *
 */

#include "shammodels/sph/modules/io/VTKDump.hpp"
#include "shambackends/kernel_call.hpp"
#include "shammodels/common/io/VTKDumpUtils.hpp"
#include "shammodels/sph/math/density.hpp"
#include "shamrock/io/LegacyVtkWriter.hpp"
#include "shamrock/patch/PatchDataFieldSpan.hpp"
#include "shamrock/scheduler/SchedulerUtility.hpp"

// Use shared VTK dump utilities
using shammodels::common::io::start_dump;
using shammodels::common::io::vtk_dump_add_compute_field;
using shammodels::common::io::vtk_dump_add_field;
using shammodels::common::io::vtk_dump_add_patch_id;
using shammodels::common::io::vtk_dump_add_worldrank;

namespace shammodels::sph::modules {

    template<class Tvec, template<class> class SPHKernel>
    void VTKDump<Tvec, SPHKernel>::do_dump(std::string filename, bool add_patch_world_id) {

        StackEntry stack_loc{};

        using namespace shamrock;
        using namespace shamrock::patch;
        shamrock::SchedulerUtility utility(scheduler());

        PatchDataLayerLayout &pdl   = scheduler().pdl_old();
        const u32 ixyz              = pdl.get_field_idx<Tvec>("xyz");
        const u32 ivxyz             = pdl.get_field_idx<Tvec>("vxyz");
        const u32 iaxyz             = pdl.get_field_idx<Tvec>("axyz");
        const u32 iuint             = pdl.get_field_idx<Tscal>("uint");
        const u32 iduint            = pdl.get_field_idx<Tscal>("duint");
        const u32 ihpart            = pdl.get_field_idx<Tscal>("hpart");
        ComputeField<Tscal> density = utility.make_compute_field<Tscal>("rho", 1);

        scheduler().for_each_patchdata_nonempty([&](const Patch p, PatchDataLayer &pdat) {
            shamlog_debug_ln("sph::vtk", "compute rho field for patch ", p.id_patch);

            auto &buf_hpart = pdat.get_field<Tscal>(ihpart).get_buf();

            auto sptr = shamsys::instance::get_compute_scheduler_ptr();
            auto &q   = sptr->get_queue();

            sham::EventList depends_list;
            const Tscal *acc_h = buf_hpart.get_read_access(depends_list);
            auto acc_rho       = density.get_buf(p.id_patch).get_write_access(depends_list);

            auto e = q.submit(depends_list, [&](sycl::handler &cgh) {
                const Tscal part_mass = solver_config.gpart_mass;

                cgh.parallel_for(sycl::range<1>{pdat.get_obj_cnt()}, [=](sycl::item<1> item) {
                    u32 gid = (u32) item.get_id();
                    using namespace shamrock::sph;
                    Tscal rho_ha = rho_h(part_mass, acc_h[gid], Kernel::hfactd);
                    acc_rho[gid] = rho_ha;
                });
            });

            buf_hpart.complete_event_state(e);
            density.get_buf(p.id_patch).complete_event_state(e);
        });

        shamrock::LegacyVtkWriter writer = start_dump<Tvec>(scheduler(), filename);
        writer.add_point_data_section();

        u32 fnum = 0;
        if (add_patch_world_id) {
            fnum += 2;
        }
        fnum++;
        fnum++;
        fnum++;
        fnum++;
        fnum++;

        if (solver_config.has_field_alphaAV()) {
            fnum++;
        }

        if (solver_config.has_field_divv()) {
            fnum++;
        }

        if (solver_config.has_field_curlv()) {
            fnum++;
        }

        if (solver_config.has_field_soundspeed()) {
            fnum++;
        }

        if (solver_config.has_field_dtdivv()) {
            fnum++;
        }

        if (solver_config.compute_luminosity) {
            fnum++;
        }

        if (solver_config.dust_config.has_epsilon_field()) {
            const u32 ndust = solver_config.dust_config.get_dust_nvar();
            fnum += ndust;
        }

        if (solver_config.dust_config.has_deltav_field()) {
            const u32 ndust = solver_config.dust_config.get_dust_nvar();
            fnum += ndust;
        }

        if (solver_config.dust_config.has_s_j_field()) {
            const u32 ndust = solver_config.dust_config.get_dust_nvar();
            fnum += ndust * 3; // s_j, ds_j_dt and delta_v
        }

        writer.add_field_data_section(fnum);

        if (add_patch_world_id) {
            vtk_dump_add_patch_id(scheduler(), writer);
            vtk_dump_add_worldrank(scheduler(), writer);
        }

        vtk_dump_add_field<Tscal>(scheduler(), writer, ihpart, "h");
        vtk_dump_add_field<Tscal>(scheduler(), writer, iuint, "u");
        vtk_dump_add_field<Tvec>(scheduler(), writer, ivxyz, "v");
        vtk_dump_add_field<Tvec>(scheduler(), writer, iaxyz, "a");

        if (solver_config.has_field_alphaAV()) {
            const u32 ialpha_AV = pdl.get_field_idx<Tscal>("alpha_AV");
            vtk_dump_add_field<Tscal>(scheduler(), writer, ialpha_AV, "alpha_AV");
        }

        if (solver_config.has_field_divv()) {
            const u32 idivv = pdl.get_field_idx<Tscal>("divv");
            vtk_dump_add_field<Tscal>(scheduler(), writer, idivv, "divv");
        }

        if (solver_config.has_field_dtdivv()) {
            const u32 idtdivv = pdl.get_field_idx<Tscal>("dtdivv");
            vtk_dump_add_field<Tscal>(scheduler(), writer, idtdivv, "dtdivv");
        }

        if (solver_config.has_field_curlv()) {
            const u32 icurlv = pdl.get_field_idx<Tvec>("curlv");
            vtk_dump_add_field<Tvec>(scheduler(), writer, icurlv, "curlv");
        }

        if (solver_config.has_field_soundspeed()) {
            const u32 isoundspeed = pdl.get_field_idx<Tscal>("soundspeed");
            vtk_dump_add_field<Tscal>(scheduler(), writer, isoundspeed, "soundspeed");
        }

        if (solver_config.compute_luminosity) {
            const u32 iluminosity = pdl.get_field_idx<Tscal>("luminosity");
            vtk_dump_add_field<Tscal>(scheduler(), writer, iluminosity, "luminosity");
        }

        vtk_dump_add_compute_field(scheduler(), writer, density, "rho");

        if (solver_config.dust_config.has_epsilon_field()) {
            const u32 iepsilon = pdl.get_field_idx<Tscal>("epsilon");
            const u32 ndust    = solver_config.dust_config.get_dust_nvar();

            for (u32 idust = 0; idust < ndust; idust++) {
                ComputeField<Tscal> tmp_epsilon
                    = utility.make_compute_field<Tscal>("tmp_epsilon", 1);

                scheduler().for_each_patchdata_nonempty([&](const Patch p, PatchDataLayer &pdat) {
                    shamlog_debug_ln(
                        "sph::vtk",
                        "compute extract epsilon field with idust =",
                        idust,
                        p.id_patch);

                    auto &buf_epsilon = pdat.get_field<Tscal>(iepsilon);
                    PatchDataFieldSpan<Tscal> span_epsilon{buf_epsilon, 0, pdat.get_obj_cnt()};

                    auto sptr = shamsys::instance::get_compute_scheduler_ptr();
                    auto &q   = sptr->get_queue();

                    sham::kernel_call(
                        q,
                        sham::MultiRef{span_epsilon},
                        sham::MultiRef{tmp_epsilon.get_buf(p.id_patch)},
                        pdat.get_obj_cnt(),
                        [&, idust](u32 i, auto epsilon_field, Tscal *acc_epsilon) {
                            acc_epsilon[i] = epsilon_field(i, idust);
                        });
                });

                vtk_dump_add_compute_field(
                    scheduler(), writer, tmp_epsilon, "epsilon_" + std::to_string(idust));
            }
        }

        if (solver_config.dust_config.has_deltav_field()) {
            const u32 ideltav = pdl.get_field_idx<Tvec>("deltav");
            const u32 ndust   = solver_config.dust_config.get_dust_nvar();

            for (u32 idust = 0; idust < ndust; idust++) {
                ComputeField<Tvec> tmp_deltav = utility.make_compute_field<Tvec>("tmp_deltav", 1);

                scheduler().for_each_patchdata_nonempty([&](const Patch p, PatchDataLayer &pdat) {
                    shamlog_debug_ln(
                        "sph::vtk", "compute extract deltav field with idust =", idust, p.id_patch);

                    auto &buf_deltav = pdat.get_field<Tvec>(ideltav);
                    PatchDataFieldSpan<Tvec> span_deltav{buf_deltav, 0, pdat.get_obj_cnt()};

                    auto sptr = shamsys::instance::get_compute_scheduler_ptr();
                    auto &q   = sptr->get_queue();

                    sham::kernel_call(
                        q,
                        sham::MultiRef{span_deltav},
                        sham::MultiRef{tmp_deltav.get_buf(p.id_patch)},
                        pdat.get_obj_cnt(),
                        [&, idust](u32 i, auto deltav_field, Tvec *acc_deltav) {
                            acc_deltav[i] = deltav_field(i, idust);
                        });
                });

                vtk_dump_add_compute_field(
                    scheduler(), writer, tmp_deltav, "deltav_" + std::to_string(idust));
            }
        }

        if (solver_config.dust_config.has_s_j_field()) {
            const u32 is_j  = pdl.get_field_idx<Tscal>("s_j");
            const u32 ndust = solver_config.dust_config.get_dust_nvar();

            for (u32 idust = 0; idust < ndust; idust++) {
                ComputeField<Tscal> tmp_s_j = utility.make_compute_field<Tscal>("tmp_s_j", 1);

                scheduler().for_each_patchdata_nonempty([&](const Patch p, PatchDataLayer &pdat) {
                    shamlog_debug_ln(
                        "sph::vtk", "compute extract s_j field with idust =", idust, p.id_patch);

                    auto &buf_s_j = pdat.get_field<Tscal>(is_j);
                    PatchDataFieldSpan<Tscal> span_s_j{buf_s_j, 0, pdat.get_obj_cnt()};

                    auto sptr = shamsys::instance::get_compute_scheduler_ptr();
                    auto &q   = sptr->get_queue();

                    sham::kernel_call(
                        q,
                        sham::MultiRef{span_s_j},
                        sham::MultiRef{tmp_s_j.get_buf(p.id_patch)},
                        pdat.get_obj_cnt(),
                        [&, idust](u32 i, auto s_j_field, Tscal *acc_s_j) {
                            acc_s_j[i] = s_j_field(i, idust);
                        });
                });

                vtk_dump_add_compute_field(
                    scheduler(), writer, tmp_s_j, "s_j_" + std::to_string(idust));
            }
        }

        if (solver_config.dust_config.has_s_j_field()) {
            const u32 ids_j_dt = pdl.get_field_idx<Tscal>("ds_j_dt");
            const u32 ndust    = solver_config.dust_config.get_dust_nvar();

            for (u32 idust = 0; idust < ndust; idust++) {
                ComputeField<Tscal> tmp_ds_j_dt
                    = utility.make_compute_field<Tscal>("tmp_ds_j_dt", 1);

                scheduler().for_each_patchdata_nonempty([&](const Patch p, PatchDataLayer &pdat) {
                    shamlog_debug_ln(
                        "sph::vtk",
                        "compute extract ds_j_dt field with idust =",
                        idust,
                        p.id_patch);

                    auto &buf_ds_j_dt = pdat.get_field<Tscal>(ids_j_dt);
                    PatchDataFieldSpan<Tscal> span_ds_j_dt{buf_ds_j_dt, 0, pdat.get_obj_cnt()};

                    auto sptr = shamsys::instance::get_compute_scheduler_ptr();
                    auto &q   = sptr->get_queue();

                    sham::kernel_call(
                        q,
                        sham::MultiRef{span_ds_j_dt},
                        sham::MultiRef{tmp_ds_j_dt.get_buf(p.id_patch)},
                        pdat.get_obj_cnt(),
                        [&, idust](u32 i, auto ds_j_dt_field, Tscal *acc_ds_j_dt) {
                            acc_ds_j_dt[i] = ds_j_dt_field(i, idust);
                        });
                });

                vtk_dump_add_compute_field(
                    scheduler(), writer, tmp_ds_j_dt, "ds_j_dt_" + std::to_string(idust));
            }
        }

        if (solver_config.dust_config.has_s_j_field()) {
            const u32 idelta_v = pdl.get_field_idx<Tvec>("delta_v");
            const u32 ndust    = solver_config.dust_config.get_dust_nvar();

            for (u32 idust = 0; idust < ndust; idust++) {
                ComputeField<Tvec> tmp_delta_v = utility.make_compute_field<Tvec>("tmp_delta_v", 1);

                scheduler().for_each_patchdata_nonempty([&](const Patch p, PatchDataLayer &pdat) {
                    shamlog_debug_ln(
                        "sph::vtk",
                        "compute extract delta_v field with idust =",
                        idust,
                        p.id_patch);

                    auto &buf_delta_v = pdat.get_field<Tvec>(idelta_v);
                    PatchDataFieldSpan<Tvec> span_delta_v{buf_delta_v, 0, pdat.get_obj_cnt()};

                    auto sptr = shamsys::instance::get_compute_scheduler_ptr();
                    auto &q   = sptr->get_queue();

                    sham::kernel_call(
                        q,
                        sham::MultiRef{span_delta_v},
                        sham::MultiRef{tmp_delta_v.get_buf(p.id_patch)},
                        pdat.get_obj_cnt(),
                        [&, idust](u32 i, auto delta_v_field, Tvec *acc_delta_v) {
                            acc_delta_v[i] = delta_v_field(i, idust);
                        });
                });

                vtk_dump_add_compute_field(
                    scheduler(), writer, tmp_delta_v, "delta_v_" + std::to_string(idust));
            }
        }
    }

} // namespace shammodels::sph::modules

using namespace shammath;

template class shammodels::sph::modules::VTKDump<f64_3, M4>;
template class shammodels::sph::modules::VTKDump<f64_3, M6>;
template class shammodels::sph::modules::VTKDump<f64_3, M8>;

template class shammodels::sph::modules::VTKDump<f64_3, C2>;
template class shammodels::sph::modules::VTKDump<f64_3, C4>;
template class shammodels::sph::modules::VTKDump<f64_3, C6>;
