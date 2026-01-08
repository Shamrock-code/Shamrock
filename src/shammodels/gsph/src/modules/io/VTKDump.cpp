// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright (c) 2021-2025 Timothée David--Cléris <tim.shamrock@proton.me>
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
 *
 * Physics-agnostic VTK output. Field selection is delegated to the physics
 * mode via get_output_field_names() - this module has no physics knowledge.
 */

#include "shammodels/gsph/modules/io/VTKDump.hpp"
#include "shamalgs/memory.hpp"
#include "shammodels/common/io/VTKDumpUtils.hpp"
#include "shammodels/gsph/FieldNames.hpp"
#include "shamrock/io/LegacyVtkWritter.hpp"
#include "shamrock/scheduler/SchedulerUtility.hpp"
#include "shamrock/solvergraph/Field.hpp"
#include "shamrock/solvergraph/IFieldRefs.hpp"
#include "shamsys/NodeInstance.hpp"

// Use shared VTK dump utilities (DRY principle)
using shammodels::common::io::start_dump;
using shammodels::common::io::vtk_dump_add_field;
using shammodels::common::io::vtk_dump_add_patch_id;
using shammodels::common::io::vtk_dump_add_worldrank;

namespace shammodels::gsph::modules {

    namespace {
        // Helper to write a solvergraph::Field to VTK
        template<class T>
        void vtk_write_field(
            PatchScheduler &sched,
            shamrock::LegacyVtkWritter &writer,
            shamrock::solvergraph::IFieldRefs<T> &field,
            const std::string &name) {

            using namespace shamrock::patch;
            u64 num_obj = sched.get_rank_count();

            if (num_obj > 0) {
                sycl::buffer<T> buf(num_obj);
                auto &refs = field.get_refs();

                u64 ptr = 0;
                sched.for_each_patchdata_nonempty([&](Patch cur_p, PatchDataLayer &pdat) {
                    if (!refs.has_key(cur_p.id_patch)) {
                        return;
                    }
                    auto &pdf = refs.get(cur_p.id_patch).get();
                    // Use real particle count from pdat, not pdf which may include ghosts
                    u32 cnt   = pdat.get_obj_cnt();
                    if (cnt == 0) {
                        return;
                    }

                    shamalgs::memory::write_with_offset_into(
                        shamsys::instance::get_compute_scheduler().get_queue(),
                        buf,
                        pdf.get_buf(),
                        ptr,
                        cnt);

                    ptr += cnt;
                });

                writer.write_field(name, buf, num_obj);
            } else {
                writer.write_field_no_buf<T>(name);
            }
        }
    } // namespace

    template<class Tvec, template<class> class SPHKernel>
    void VTKDump<Tvec, SPHKernel>::do_dump(
        std::string filename,
        bool add_patch_world_id,
        const std::vector<std::string> &physics_field_names) {

        StackEntry stack_loc{};

        using namespace shamrock;
        using namespace shamrock::patch;
        shamrock::SchedulerUtility utility(scheduler());

        PatchDataLayerLayout &pdl = scheduler().pdl();
        const u32 ivxyz           = pdl.get_field_idx<Tvec>(fields::VXYZ);
        const u32 iaxyz           = pdl.get_field_idx<Tvec>(fields::AXYZ);
        const u32 ihpart          = pdl.get_field_idx<Tscal>(fields::HPART);

        const bool has_uint = solver_config.has_field_uint();
        const u32 iuint     = has_uint ? pdl.get_field_idx<Tscal>(fields::UINT) : 0;

        // ════════════════════════════════════════════════════════════════════════
        // Count fields to write
        // ════════════════════════════════════════════════════════════════════════

        u32 fnum = 0;
        if (add_patch_world_id) {
            fnum += 2; // patchid and world_rank
        }
        fnum++; // h
        fnum++; // v
        fnum++; // a

        if (has_uint) {
            fnum++; // u
        }

        // Count physics fields that exist in storage field maps
        for (const auto &field_name : physics_field_names) {
            if (storage.scalar_fields.count(field_name) > 0) {
                fnum++;
            } else if (storage.vector_fields.count(field_name) > 0) {
                fnum++;
            }
        }

        // ════════════════════════════════════════════════════════════════════════
        // Write VTK file
        // ════════════════════════════════════════════════════════════════════════

        shamrock::LegacyVtkWritter writter = start_dump<Tvec>(scheduler(), filename);
        writter.add_point_data_section();
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

        // Write physics fields from field maps (completely physics-agnostic)
        for (const auto &field_name : physics_field_names) {
            if (auto it = storage.scalar_fields.find(field_name);
                it != storage.scalar_fields.end() && it->second) {
                vtk_write_field<Tscal>(scheduler(), writter, *it->second, field_name);
            } else if (auto vit = storage.vector_fields.find(field_name);
                       vit != storage.vector_fields.end() && vit->second) {
                vtk_write_field<Tvec>(scheduler(), writter, *vit->second, field_name);
            }
        }
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

template class shammodels::gsph::modules::VTKDump<f64_3, TGauss3>;
