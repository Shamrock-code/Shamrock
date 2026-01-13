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
 * @author Yona Lapeyre (yona.lapeyre@ens-lyon.fr) --no git blame--
 * @brief VTK dump implementation for GSPH solver
 */

#include "shammodels/gsph/modules/io/VTKDump.hpp"
#include "shamalgs/memory.hpp"
#include "shamcomm/worldInfo.hpp"
#include "shammodels/gsph/config/FieldNames.hpp"
#include "shammodels/sph/math/density.hpp"
#include "shamrock/io/LegacyVtkWritter.hpp"
#include "shamrock/scheduler/SchedulerUtility.hpp"
#include "shamsys/NodeInstance.hpp"

namespace {

    template<class Tvec>
    shamrock::LegacyVtkWritter start_dump(PatchScheduler &sched, std::string dump_name) {
        StackEntry stack_loc{};
        shamrock::LegacyVtkWritter writer(dump_name, true, shamrock::UnstructuredGrid);

        using namespace shamrock::patch;

        u64 num_obj = sched.get_rank_count();

        shamlog_debug_mpi_ln("gsph::vtk", "rank count =", num_obj);

        const u32 ixyz = sched.pdl().get_field_idx<Tvec>(gsph::names::common::xyz);
        std::unique_ptr<sycl::buffer<Tvec>> pos = sched.rankgather_field<Tvec>(ixyz);

        writer.write_points(pos, num_obj);

        return writer;
    }

    void vtk_dump_add_patch_id(PatchScheduler &sched, shamrock::LegacyVtkWritter &writter) {
        StackEntry stack_loc{};

        u64 num_obj = sched.get_rank_count();

        using namespace shamrock::patch;

        if (num_obj > 0) {
            sycl::buffer<u64> idp(num_obj);

            u64 ptr = 0;
            sched.for_each_patchdata_nonempty([&](Patch cur_p, PatchDataLayer &pdat) {
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
            sycl::buffer<u32> idp(num_obj);

            u64 ptr = 0;
            sched.for_each_patchdata_nonempty([&](Patch cur_p, PatchDataLayer &pdat) {
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

} // anonymous namespace

namespace shammodels::gsph::modules {

    template<class Tvec, template<class> class SPHKernel>
    void VTKDump<Tvec, SPHKernel>::do_dump(std::string filename, bool add_patch_world_id) {

        StackEntry stack_loc{};

        using namespace shamrock;
        using namespace shamrock::patch;
        shamrock::SchedulerUtility utility(scheduler());

        PatchDataLayerLayout &pdl = scheduler().pdl();
        const u32 ixyz            = pdl.get_field_idx<Tvec>(gsph::names::common::xyz);
        const u32 ivxyz           = pdl.get_field_idx<Tvec>(gsph::names::newtonian::vxyz);
        const u32 iaxyz           = pdl.get_field_idx<Tvec>(gsph::names::newtonian::axyz);
        const u32 ihpart          = pdl.get_field_idx<Tscal>(gsph::names::common::hpart);

        // Check for optional internal energy field
        const bool has_uint = solver_config.has_field_uint();
        const u32 iuint     = has_uint ? pdl.get_field_idx<Tscal>(gsph::names::newtonian::uint) : 0;

        // Compute density field from smoothing length
        ComputeField<Tscal> density = utility.make_compute_field<Tscal>("rho", 1);

        scheduler().for_each_patchdata_nonempty([&](const Patch p, PatchDataLayer &pdat) {
            shamlog_debug_ln("gsph::vtk", "compute rho field for patch ", p.id_patch);

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

        // Compute pressure field from EOS
        ComputeField<Tscal> pressure_field = utility.make_compute_field<Tscal>("P", 1);

        scheduler().for_each_patchdata_nonempty([&](const Patch p, PatchDataLayer &pdat) {
            auto &buf_hpart = pdat.get_field<Tscal>(ihpart).get_buf();

            auto sptr = shamsys::instance::get_compute_scheduler_ptr();
            auto &q   = sptr->get_queue();

            sham::EventList depends_list;
            const Tscal *acc_h = buf_hpart.get_read_access(depends_list);
            auto acc_P         = pressure_field.get_buf(p.id_patch).get_write_access(depends_list);

            const Tscal *acc_u = nullptr;
            if (has_uint) {
                acc_u = pdat.get_field<Tscal>(iuint).get_buf().get_read_access(depends_list);
            }

            auto e = q.submit(depends_list, [&](sycl::handler &cgh) {
                const Tscal part_mass = solver_config.gpart_mass;
                const Tscal gamma     = solver_config.get_eos_gamma();
                const bool do_uint    = has_uint;

                cgh.parallel_for(sycl::range<1>{pdat.get_obj_cnt()}, [=](sycl::item<1> item) {
                    u32 gid = (u32) item.get_id();
                    using namespace shamrock::sph;
                    Tscal rho = rho_h(part_mass, acc_h[gid], Kernel::hfactd);

                    if (do_uint && acc_u != nullptr) {
                        // Adiabatic EOS: P = (gamma - 1) * rho * u
                        acc_P[gid] = (gamma - Tscal(1)) * rho * acc_u[gid];
                    } else {
                        // Isothermal: use cs = 1 by default
                        acc_P[gid] = rho; // P = cs^2 * rho with cs = 1
                    }
                });
            });

            buf_hpart.complete_event_state(e);
            pressure_field.get_buf(p.id_patch).complete_event_state(e);
            if (has_uint) {
                pdat.get_field<Tscal>(iuint).get_buf().complete_event_state(e);
            }
        });

        shamrock::LegacyVtkWritter writter = start_dump<Tvec>(scheduler(), filename);
        writter.add_point_data_section();

        // Count fields to write
        u32 fnum = 0;
        if (add_patch_world_id) {
            fnum += 2; // patchid and world_rank
        }
        fnum++; // h
        fnum++; // v
        fnum++; // a
        fnum++; // rho
        fnum++; // P

        if (has_uint) {
            fnum++; // u
        }

        writter.add_field_data_section(fnum);

        if (add_patch_world_id) {
            vtk_dump_add_patch_id(scheduler(), writter);
            vtk_dump_add_worldrank(scheduler(), writter);
        }

        vtk_dump_add_field<Tscal>(scheduler(), writter, ihpart, "h");
        vtk_dump_add_field<Tvec>(scheduler(), writter, ivxyz, "v");
        vtk_dump_add_field<Tvec>(scheduler(), writter, iaxyz, "a");

        if (has_uint) {
            vtk_dump_add_field<Tscal>(scheduler(), writter, iuint, "u");
        }

        vtk_dump_add_compute_field(scheduler(), writter, density, "rho");
        vtk_dump_add_compute_field(scheduler(), writter, pressure_field, "P");
    }

} // namespace shammodels::gsph::modules

// Explicit template instantiations
using namespace shammath;

template class shammodels::gsph::modules::VTKDump<f64_3, M4>;
template class shammodels::gsph::modules::VTKDump<f64_3, M6>;
template class shammodels::gsph::modules::VTKDump<f64_3, M8>;

template class shammodels::gsph::modules::VTKDump<f64_3, C2>;
template class shammodels::gsph::modules::VTKDump<f64_3, C4>;
template class shammodels::gsph::modules::VTKDump<f64_3, C6>;
