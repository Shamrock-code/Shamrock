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
#include "shammodels/ramses/Model.hpp"
#include "shammodels/ramses/modules/AMRSetup.hpp"
#include "shamrock/io/LegacyVtkWritter.hpp"
#include "shamrock/scheduler/PatchScheduler.hpp"
#include "shamsys/NodeInstance.hpp"
#include <string>

template<class Tvec, class TgridVec>
void shammodels::basegodunov::Model<Tvec, TgridVec>::init_scheduler(
    u32 crit_split, u32 crit_merge) {

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

    solver.init_solver_graph();
}

template<class Tvec, class TgridVec>
void shammodels::basegodunov::Model<Tvec, TgridVec>::make_base_grid(
    TgridVec bmin, TgridVec cell_size, u32_3 cell_count) {

    if (cell_size.x() < Solver::Config::AMRBlock::Nside) {
        shambase::throw_with_loc<std::invalid_argument>(shambase::format(
            "the x block size must be larger than {}, currently : cell_size = {}",
            Solver::Config::AMRBlock::Nside,
            cell_size));
    }
    if (cell_size.y() < Solver::Config::AMRBlock::Nside) {
        shambase::throw_with_loc<std::invalid_argument>(shambase::format(
            "the y block size must be larger than {}, currently : cell_size = {}",
            Solver::Config::AMRBlock::Nside,
            cell_size));
    }
    if (cell_size.z() < Solver::Config::AMRBlock::Nside) {
        shambase::throw_with_loc<std::invalid_argument>(shambase::format(
            "the z block size must be larger than {}, currently : cell_size = {}",
            Solver::Config::AMRBlock::Nside,
            cell_size));
    }

    modules::AMRSetup<Tvec, TgridVec> setup(ctx, solver.solver_config, solver.storage);
    setup.make_base_grid(bmin, cell_size, {cell_count[0], cell_count[1], cell_count[2]});
    return;

    /* Old cell injection
    shamrock::amr::AMRGrid<TgridVec, 3> grid(shambase::get_check_ref(ctx.sched));
    grid.make_base_grid(bmin, cell_size, {cell_count.x(), cell_count.y(), cell_count.z()});

    PatchScheduler &sched = shambase::get_check_ref(ctx.sched);

    sched.owned_patch_id = sched.patch_list.build_local();
    sched.patch_list.build_local_idx_map();
    sched.update_local_load_value([&](shamrock::patch::Patch p) {
        return sched.patch_data.owned_data.get(p.id_patch).get_obj_cnt();
    });
    sched.scheduler_step(true, true);
    */
}

template<class Tvec, class TgridVec>
void shammodels::basegodunov::Model<Tvec, TgridVec>::dump_vtk(std::string filename) {

    StackEntry stack_loc{};
    shamrock::LegacyVtkWritter writer(filename, true, shamrock::UnstructuredGrid);

    try {

        PatchScheduler &sched = shambase::get_check_ref(ctx.sched);

        u32 block_size = Solver::AMRBlock::block_size;

        u64 num_obj = sched.get_rank_count();

        std::unique_ptr<sycl::buffer<TgridVec>> pos1 = sched.rankgather_field<TgridVec>(0);
        std::unique_ptr<sycl::buffer<TgridVec>> pos2 = sched.rankgather_field<TgridVec>(1);

        sycl::buffer<Tvec> pos_min_cell(num_obj * block_size);
        sycl::buffer<Tvec> pos_max_cell(num_obj * block_size);

        if (num_obj > 0) {

            shamsys::instance::get_compute_queue().submit([&, block_size](sycl::handler &cgh) {
                sycl::accessor acc_p1{shambase::get_check_ref(pos1), cgh, sycl::read_only};
                sycl::accessor acc_p2{shambase::get_check_ref(pos2), cgh, sycl::read_only};
                sycl::accessor cell_min{pos_min_cell, cgh, sycl::write_only, sycl::no_init};
                sycl::accessor cell_max{pos_max_cell, cgh, sycl::write_only, sycl::no_init};

                using Block = typename Solver::AMRBlock;

                shambase::parallel_for(cgh, num_obj, "rescale cells", [=](u64 id_a) {
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
                                cell_max[id_a * block_size + i]
                                    = block_min + (delta_cell) + delta_val;
                            }
                        }
                    }
                });
            });
        }

        writer.write_voxel_cells(pos_min_cell, pos_max_cell, num_obj * block_size);

        writer.add_cell_data_section();

        u32 fieldnum = 3;
        if (solver.solver_config.is_dust_on()) {
            u32 ndust = solver.solver_config.dust_config.ndust;
            fieldnum += 2 * ndust;
        }
        writer.add_field_data_section(fieldnum);

        std::unique_ptr<sycl::buffer<Tscal>> fields_rho = sched.rankgather_field<Tscal>(2);
        writer.write_field("rho", fields_rho, num_obj * block_size);

        std::unique_ptr<sycl::buffer<Tvec>> fields_vel = sched.rankgather_field<Tvec>(3);
        writer.write_field("rhovel", fields_vel, num_obj * block_size);

        std::unique_ptr<sycl::buffer<Tscal>> fields_eint = sched.rankgather_field<Tscal>(4);
        writer.write_field("rhoetot", fields_eint, num_obj * block_size);

        if (solver.solver_config.is_dust_on()) {
            u32 ndust = solver.solver_config.dust_config.ndust;

            shamrock::patch::PatchDataLayout &pdl = solver.scheduler().pdl;
            const u32 irho_dust                   = pdl.get_field_idx<Tscal>("rho_dust");
            const u32 irhovel_dust                = pdl.get_field_idx<Tvec>("rhovel_dust");

            std::unique_ptr<sycl::buffer<Tscal>> fields_rho_dust
                = sched.rankgather_field<Tscal>(irho_dust);
            // writer.write_field("rho_dust", fields_rho_dust, ndust*num_obj*block_size);

            if (fields_rho_dust) {
                u32 nobj   = fields_rho_dust->size();
                u32 nsplit = ndust;

                for (u32 off = 0; off < nsplit; off++) {

                    sycl::buffer<Tscal> partition(nobj / nsplit);

                    shamsys::instance::get_compute_queue()
                        .submit([&, off, nsplit](sycl::handler &cgh) {
                            sycl::accessor out{partition, cgh, sycl::write_only, sycl::no_init};
                            sycl::accessor in{*fields_rho_dust, cgh, sycl::read_only};

                            shambase::parallel_for(
                                cgh, nobj / nsplit, "split field for dump", [=](u64 i) {
                                    out[i] = in[i * nsplit + off];
                                });
                        })
                        .wait();

                    writer.write_field(
                        std::string("rho_dust") + std::to_string(off),
                        partition,
                        num_obj * block_size);
                }
            }

            std::unique_ptr<sycl::buffer<Tvec>> fields_vel_dust
                = sched.rankgather_field<Tvec>(irhovel_dust);
            if (fields_vel_dust) {
                u32 nobj   = fields_vel_dust->size();
                u32 nsplit = ndust;

                for (u32 off = 0; off < nsplit; off++) {

                    sycl::buffer<Tvec> partition(nobj / nsplit);

                    shamsys::instance::get_compute_queue()
                        .submit([&, off, nsplit](sycl::handler &cgh) {
                            sycl::accessor out{partition, cgh, sycl::write_only, sycl::no_init};
                            sycl::accessor in{*fields_vel_dust, cgh, sycl::read_only};

                            shambase::parallel_for(
                                cgh, nobj / nsplit, "split field for dump", [=](u64 i) {
                                    out[i] = in[i * nsplit + off];
                                });
                        })
                        .wait();

                    writer.write_field(
                        std::string("rhovel_dust") + std::to_string(off),
                        partition,
                        num_obj * block_size);
                }
            }
        }

    } catch (std::runtime_error e) {
        logger::err_ln(
            "Godunov",
            "std::runtime_error catched while MPI file open -> unrecoverable\n what():\n",
            e.what());
    } catch (std::exception e) {
        logger::err_ln(
            "Godunov",
            "exception catched while MPI file open -> unrecoverable\n what():\n",
            e.what());
    } catch (...) {
        logger::err_ln("Godunov", "something unknwon catched while MPI file open -> unrecoverable");
    }
}

template class shammodels::basegodunov::Model<f64_3, i64_3>;
