// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright (c) 2021-2024 Timothée David--Cléris <tim.shamrock@proton.me>
// SPDX-License-Identifier: CeCILL Free Software License Agreement v2.1
// Shamrock is licensed under the CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

/**
 * @file VTKDump.cpp
 * @author Timothée David--Cléris (timothee.david--cleris@ens-lyon.fr)
 * @brief
 *
 */

#include "shammodels/sph/modules/io/VTKDump.hpp"
#include "shambackends/kernel_call.hpp"
#include "shammodels/sph/math/density.hpp"
#include "shamrock/io/LegacyVtkWritter.hpp"
#include "shamrock/patch/PatchDataFieldSpan.hpp"
#include "shamrock/scheduler/SchedulerUtility.hpp"

template<class vec>
shamrock::LegacyVtkWritter start_dump(PatchScheduler &sched, std::string dump_name) {
    StackEntry stack_loc{};
    shamrock::LegacyVtkWritter writer(dump_name, true, shamrock::UnstructuredGrid);

    using namespace shamrock::patch;

    u64 num_obj = sched.get_rank_count();

    shamlog_debug_mpi_ln("sph::BasicGas", "rank count =", num_obj);

    std::unique_ptr<sycl::buffer<vec>> pos = sched.rankgather_field<vec>(0);

    writer.write_points(pos, num_obj);

    return writer;
}

void vtk_dump_add_patch_id(PatchScheduler &sched, shamrock::LegacyVtkWritter &writter) {
    StackEntry stack_loc{};

    u64 num_obj = sched.get_rank_count();

    using namespace shamrock::patch;

    if (num_obj > 0) {
        // TODO aggregate field ?
        sycl::buffer<u64> idp(num_obj);

        u64 ptr = 0; // TODO accumulate_field() in scheduler ?
        sched.for_each_patchdata_nonempty([&](Patch cur_p, PatchData &pdat) {
            using namespace shamalgs::memory;
            using namespace shambase;

            write_with_offset_into(
                shamsys::instance::get_compute_queue(),
                idp,
                cur_p.id_patch,
                ptr,
                pdat.get_obj_cnt());

            ptr += pdat.get_obj_cnt();
        });

        writter.write_field("patchid", idp, num_obj);
    } else {
        writter.write_field_no_buf<u64>("patchid");
    }
}

void vtk_dump_add_worldrank(PatchScheduler &sched, shamrock::LegacyVtkWritter &writter) {
    StackEntry stack_loc{};

    using namespace shamrock::patch;
    u64 num_obj = sched.get_rank_count();

    if (num_obj > 0) {

        // TODO aggregate field ?
        sycl::buffer<u32> idp(num_obj);

        u64 ptr = 0; // TODO accumulate_field() in scheduler ?
        sched.for_each_patchdata_nonempty([&](Patch cur_p, PatchData &pdat) {
            using namespace shamalgs::memory;
            using namespace shambase;

            write_with_offset_into<u32>(
                shamsys::instance::get_compute_queue(),
                idp,
                shamcomm::world_rank(),
                ptr,
                pdat.get_obj_cnt());

            ptr += pdat.get_obj_cnt();
        });

        writter.write_field("world_rank", idp, num_obj);

    } else {
        writter.write_field_no_buf<u32>("world_rank");
    }
}

template<class T>
void vtk_dump_add_compute_field(
    PatchScheduler &sched,
    shamrock::LegacyVtkWritter &writter,
    shamrock::ComputeField<T> &field,
    std::string field_dump_name) {
    StackEntry stack_loc{};

    using namespace shamrock::patch;
    u64 num_obj = sched.get_rank_count();

    if (num_obj > 0) {
        std::unique_ptr<sycl::buffer<T>> field_vals = field.rankgather_computefield(sched);

        writter.write_field(field_dump_name, field_vals, num_obj);
    } else {
        writter.write_field_no_buf<T>(field_dump_name);
    }
}

template<class T>
void vtk_dump_add_field(
    PatchScheduler &sched,
    shamrock::LegacyVtkWritter &writter,
    u32 field_idx,
    std::string field_dump_name) {
    StackEntry stack_loc{};

    using namespace shamrock::patch;
    u64 num_obj = sched.get_rank_count();

    if (num_obj > 0) {
        std::unique_ptr<sycl::buffer<T>> field_vals = sched.rankgather_field<T>(field_idx);

        writter.write_field(field_dump_name, field_vals, num_obj);
    } else {
        writter.write_field_no_buf<T>(field_dump_name);
    }
}

namespace shammodels::sph::modules {

    template<class Tvec, template<class> class SPHKernel>
    void VTKDump<Tvec, SPHKernel>::do_dump(std::string filename, bool add_patch_world_id) {

        StackEntry stack_loc{};

        using namespace shamrock;
        using namespace shamrock::patch;
        shamrock::SchedulerUtility utility(scheduler());
        PatchDataLayout &pdl        = scheduler().pdl;
        const u32 ixyz              = pdl.get_field_idx<Tvec>("xyz");
        const u32 ivxyz             = pdl.get_field_idx<Tvec>("vxyz");
        const u32 iaxyz             = pdl.get_field_idx<Tvec>("axyz");
        const u32 iuint             = pdl.get_field_idx<Tscal>("uint");
        const u32 iduint            = pdl.get_field_idx<Tscal>("duint");
        const u32 ihpart            = pdl.get_field_idx<Tscal>("hpart");
        ComputeField<Tscal> density = utility.make_compute_field<Tscal>("rho", 1);

        scheduler().for_each_patchdata_nonempty([&](const Patch p, PatchData &pdat) {
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

        shamrock::LegacyVtkWritter writter = start_dump<Tvec>(scheduler(), filename);
        writter.add_point_data_section();

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

        if (solver_config.dust_config.has_epsilon_field()) {
            const u32 ndust = solver_config.dust_config.get_dust_nvar();
            fnum += ndust;
        }

        if (solver_config.dust_config.has_deltav_field()) {
            const u32 ndust = solver_config.dust_config.get_dust_nvar();
            fnum += ndust;
        }

        writter.add_field_data_section(fnum);

        if (add_patch_world_id) {
            vtk_dump_add_patch_id(scheduler(), writter);
            vtk_dump_add_worldrank(scheduler(), writter);
        }

        vtk_dump_add_field<Tscal>(scheduler(), writter, ihpart, "h");
        vtk_dump_add_field<Tscal>(scheduler(), writter, iuint, "u");
        vtk_dump_add_field<Tvec>(scheduler(), writter, ivxyz, "v");
        vtk_dump_add_field<Tvec>(scheduler(), writter, iaxyz, "a");

        if (solver_config.has_field_alphaAV()) {
            const u32 ialpha_AV = pdl.get_field_idx<Tscal>("alpha_AV");
            vtk_dump_add_field<Tscal>(scheduler(), writter, ialpha_AV, "alpha_AV");
        }

        if (solver_config.has_field_divv()) {
            const u32 idivv = pdl.get_field_idx<Tscal>("divv");
            vtk_dump_add_field<Tscal>(scheduler(), writter, idivv, "divv");
        }

        if (solver_config.has_field_dtdivv()) {
            const u32 idtdivv = pdl.get_field_idx<Tscal>("dtdivv");
            vtk_dump_add_field<Tscal>(scheduler(), writter, idtdivv, "dtdivv");
        }

        if (solver_config.has_field_curlv()) {
            const u32 icurlv = pdl.get_field_idx<Tvec>("curlv");
            vtk_dump_add_field<Tvec>(scheduler(), writter, icurlv, "curlv");
        }

        if (solver_config.has_field_soundspeed()) {
            const u32 isoundspeed = pdl.get_field_idx<Tscal>("soundspeed");
            vtk_dump_add_field<Tscal>(scheduler(), writter, isoundspeed, "soundspeed");
        }

        vtk_dump_add_compute_field(scheduler(), writter, density, "rho");

        if (solver_config.dust_config.has_epsilon_field()) {
            const u32 iepsilon = pdl.get_field_idx<Tscal>("epsilon");
            const u32 ndust    = solver_config.dust_config.get_dust_nvar();

            for (u32 idust = 0; idust < ndust; idust++) {
                ComputeField<Tscal> tmp_epsilon
                    = utility.make_compute_field<Tscal>("tmp_epsilon", 1);

                scheduler().for_each_patchdata_nonempty([&](const Patch p, PatchData &pdat) {
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
                    scheduler(), writter, tmp_epsilon, "epsilon_" + std::to_string(idust));
            }
        }

        if (solver_config.dust_config.has_deltav_field()) {
            const u32 ideltav = pdl.get_field_idx<Tvec>("deltav");
            const u32 ndust   = solver_config.dust_config.get_dust_nvar();

            for (u32 idust = 0; idust < ndust; idust++) {
                ComputeField<Tvec> tmp_deltav = utility.make_compute_field<Tvec>("tmp_deltav", 1);

                scheduler().for_each_patchdata_nonempty([&](const Patch p, PatchData &pdat) {
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
                    scheduler(), writter, tmp_deltav, "deltav_" + std::to_string(idust));
            }
        }
    }

} // namespace shammodels::sph::modules

using namespace shammath;

template class shammodels::sph::modules::VTKDump<f64_3, M4>;
template class shammodels::sph::modules::VTKDump<f64_3, M6>;
template class shammodels::sph::modules::VTKDump<f64_3, M8>;
