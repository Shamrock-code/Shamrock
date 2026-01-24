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
#include "shamrock/io/LegacyVtkWritter.hpp"
#include "shamsys/NodeInstance.hpp"

namespace {

    template<class Tvec>
    shamrock::LegacyVtkWritter start_dump(PatchScheduler &sched, std::string dump_name) {
        StackEntry stack_loc{};
        shamrock::LegacyVtkWritter writer(dump_name, true, shamrock::UnstructuredGrid);

        using namespace shamrock::patch;

        u64 num_obj = sched.get_rank_count();

        shamlog_debug_mpi_ln("gsph::vtk", "rank count =", num_obj);

        const u32 ixyz = sched.pdl().get_field_idx<Tvec>(shammodels::gsph::names::common::xyz);
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

        using namespace gsph::names;

        PatchDataLayerLayout &pdl = scheduler().pdl();
        const u32 ivxyz           = pdl.get_field_idx<Tvec>(gsph::names::newtonian::vxyz);
        const u32 iaxyz           = pdl.get_field_idx<Tvec>(gsph::names::newtonian::axyz);
        const u32 ihpart          = pdl.get_field_idx<Tscal>(gsph::names::common::hpart);

        // Thermodynamic fields from patchdata (computed during timestep, persistent across
        // restarts)
        const u32 idensity    = pdl.get_field_idx<Tscal>(gsph::names::newtonian::density);
        const u32 ipressure   = pdl.get_field_idx<Tscal>(gsph::names::newtonian::pressure);
        const u32 isoundspeed = pdl.get_field_idx<Tscal>(gsph::names::newtonian::soundspeed);

        // Check for optional internal energy field
        const bool has_uint = solver_config.has_field_uint();
        const u32 iuint     = has_uint ? pdl.get_field_idx<Tscal>(gsph::names::newtonian::uint) : 0;

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
        fnum++; // cs

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

        // Read precomputed thermodynamic fields from patchdata
        vtk_dump_add_field<Tscal>(scheduler(), writter, idensity, "rho");
        vtk_dump_add_field<Tscal>(scheduler(), writter, ipressure, "P");
        vtk_dump_add_field<Tscal>(scheduler(), writter, isoundspeed, "cs");
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
