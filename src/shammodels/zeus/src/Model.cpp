// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright (c) 2021-2024 Timothée David--Cléris <tim.shamrock@proton.me>
// SPDX-License-Identifier: CeCILL Free Software License Agreement v2.1
// Shamrock is licensed under the CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

/**
 * @file Model.cpp
 * @author Timothée David--Cléris (tim.shamrock@proton.me)
 * @brief
 *
 */

#include "shambase/memory.hpp"
#include "shambackends/sycl_utils.hpp"
#include "shammodels/zeus/Model.hpp"
#include "shamrock/io/LegacyVtkWritter.hpp"
#include "shamrock/scheduler/PatchScheduler.hpp"
#include "shamsys/NodeInstance.hpp"

template<class Tvec, class TgridVec>
void shammodels::zeus::Model<Tvec, TgridVec>::init_scheduler(u32 crit_split, u32 crit_merge) {

    solver.init_required_fields();
    // solver.init_ghost_layout();
    ctx.init_sched(crit_split, crit_merge);

    using namespace shamrock::patch;

    PatchScheduler &sched = shambase::get_check_ref(ctx.sched);

    // sched.add_root_patch();

    // std::cout << "build local" << std::endl;
    // sched.owned_patch_id = sched.patch_list.build_local();
    // sched.patch_list.build_local_idx_map();
    // sched.update_local_dtcnt_value();
    // sched.update_local_load_value();
}

template<class Tvec, class TgridVec>
void shammodels::zeus::Model<Tvec, TgridVec>::make_base_grid(
    TgridVec bmin, TgridVec cell_size, u32_3 cell_count) {
    shamrock::amr::AMRGrid<TgridVec, 3> grid(shambase::get_check_ref(ctx.sched));
    grid.make_base_grid(bmin, cell_size, {cell_count.x(), cell_count.y(), cell_count.z()});

    PatchScheduler &sched = shambase::get_check_ref(ctx.sched);

    sched.owned_patch_id = sched.patch_list.build_local();
    sched.patch_list.build_local_idx_map();
    sched.update_local_load_value([&](shamrock::patch::Patch p) {
        return sched.patch_data.owned_data.get(p.id_patch).get_obj_cnt();
    });
    sched.scheduler_step(true, true);
}

template<class Tvec, class TgridVec>
void shammodels::zeus::Model<Tvec, TgridVec>::dump_vtk(std::string filename) {

    StackEntry stack_loc{};
    shamrock::LegacyVtkWritter writer(filename, true, shamrock::UnstructuredGrid);

    PatchScheduler &sched = shambase::get_check_ref(ctx.sched);

    u32 block_size = Solver::AMRBlock::block_size;

    u64 num_obj = sched.get_rank_count();

    std::unique_ptr<sycl::buffer<TgridVec>> pos1 = sched.rankgather_field<TgridVec>(0);
    std::unique_ptr<sycl::buffer<TgridVec>> pos2 = sched.rankgather_field<TgridVec>(1);

    sycl::buffer<Tvec> pos_min_cell(num_obj * block_size);
    sycl::buffer<Tvec> pos_max_cell(num_obj * block_size);

    shamsys::instance::get_compute_queue().submit([&, block_size](sycl::handler &cgh) {
        sycl::accessor acc_p1{shambase::get_check_ref(pos1), cgh, sycl::read_only};
        sycl::accessor acc_p2{shambase::get_check_ref(pos2), cgh, sycl::read_only};
        sycl::accessor cell_min{pos_min_cell, cgh, sycl::write_only, sycl::no_init};
        sycl::accessor cell_max{pos_max_cell, cgh, sycl::write_only, sycl::no_init};

        using Block = typename Solver::AMRBlock;

        shambase::parralel_for(cgh, num_obj, "rescale cells", [=](u64 id_a) {
            Tvec block_min = acc_p1[id_a].template convert<Tscal>();
            Tvec block_max = acc_p2[id_a].template convert<Tscal>();

            Tvec delta_cell = (block_max - block_min) / Block::side_size;
#pragma unroll
            for (u32 ix = 0; ix < Block::side_size; ix++) {
#pragma unroll
                for (u32 iy = 0; iy < Block::side_size; iy++) {
#pragma unroll
                    for (u32 iz = 0; iz < Block::side_size; iz++) {
                        u32 i                           = Block::get_index({ix, iy, iz});
                        Tvec delta_val                  = delta_cell * Tvec{ix, iy, iz};
                        cell_min[id_a * block_size + i] = block_min + delta_val;
                        cell_max[id_a * block_size + i] = block_min + (delta_cell) + delta_val;
                    }
                }
            }
        });
    });

    writer.write_voxel_cells(pos_min_cell, pos_max_cell, num_obj * block_size);

    writer.add_cell_data_section();
    writer.add_field_data_section(3);

    std::unique_ptr<sycl::buffer<Tscal>> fields_rho = sched.rankgather_field<Tscal>(2);
    writer.write_field("rho", fields_rho, num_obj * block_size);

    std::unique_ptr<sycl::buffer<Tscal>> fields_eint = sched.rankgather_field<Tscal>(3);
    writer.write_field("eint", fields_eint, num_obj * block_size);

    std::unique_ptr<sycl::buffer<Tvec>> fields_vel = sched.rankgather_field<Tvec>(4);
    writer.write_field("vel", fields_vel, num_obj * block_size);
}

template<class Tvec, class TgridVec>
auto shammodels::zeus::Model<Tvec, TgridVec>::evolve_once(Tscal t_current, Tscal dt_input)
    -> Tscal {
    return solver.evolve_once(t_current, dt_input);
}

template class shammodels::zeus::Model<f64_3, i64_3>;
