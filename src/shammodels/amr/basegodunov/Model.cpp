// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright(C) 2021-2023 Timothée David--Cléris <timothee.david--cleris@ens-lyon.fr>
// Licensed under CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

/**
 * @file Model.cpp
 * @author Timothée David--Cléris (timothee.david--cleris@ens-lyon.fr)
 * @brief 
 * 
 */
 
#include "Model.hpp"
#include "shambase/memory.hpp"
#include "shambase/sycl_utils.hpp"
#include "shamrock/io/LegacyVtkWritter.hpp"
#include "shamrock/scheduler/scheduler_mpi.hpp"
#include "shamsys/NodeInstance.hpp"

template<class Tvec, class TgridVec>
using Model = shammodels::basegodunov::Model<Tvec, TgridVec>;

template<class Tvec, class TgridVec>
void Model<Tvec, TgridVec>::init_scheduler(u32 crit_split, u32 crit_merge){

    solver.init_required_fields();
    //solver.init_ghost_layout();
    ctx.init_sched(crit_split, crit_merge);

    using namespace shamrock::patch;

    PatchScheduler &sched = shambase::get_check_ref(ctx.sched);

    //sched.add_root_patch();

    //std::cout << "build local" << std::endl;
    //sched.owned_patch_id = sched.patch_list.build_local();
    //sched.patch_list.build_local_idx_map();
    //sched.update_local_dtcnt_value();
    //sched.update_local_load_value();
}

template<class Tvec, class TgridVec>
void Model<Tvec, TgridVec>::make_base_grid(TgridVec bmin, TgridVec cell_size, u32_3 cell_count){
    shamrock::amr::AMRGrid<TgridVec, 3> grid (shambase::get_check_ref(ctx.sched));
    grid.make_base_grid(bmin, cell_size, {cell_count.x(), cell_count.y(), cell_count.z()});

    PatchScheduler &sched = shambase::get_check_ref(ctx.sched);

    sched.owned_patch_id = sched.patch_list.build_local();
    sched.patch_list.build_local_idx_map();
    sched.update_local_dtcnt_value();
    sched.update_local_load_value();
    sched.scheduler_step(true, true);

}

template<class Tvec, class TgridVec>
void Model<Tvec, TgridVec>::dump_vtk(std::string filename){

    StackEntry stack_loc{};
    shamrock::LegacyVtkWritter writer(filename, true, shamrock::UnstructuredGrid);

    PatchScheduler & sched = shambase::get_check_ref(ctx.sched);

    u64 num_obj = sched.get_rank_count();

    std::unique_ptr<sycl::buffer<TgridVec>> pos1 = sched.rankgather_field<TgridVec>(0);
    std::unique_ptr<sycl::buffer<TgridVec>> pos2 = sched.rankgather_field<TgridVec>(1);

    sycl::buffer<Tvec> pos_min_cell(num_obj);
    sycl::buffer<Tvec> pos_max_cell(num_obj);

    shamsys::instance::get_compute_queue().submit([&](sycl::handler & cgh){
        sycl::accessor acc_p1 {shambase::get_check_ref(pos1), cgh, sycl::read_only};
        sycl::accessor acc_p2 {shambase::get_check_ref(pos2), cgh, sycl::read_only};
        sycl::accessor cell_min {pos_min_cell, cgh, sycl::write_only, sycl::no_init};
        sycl::accessor cell_max {pos_max_cell, cgh, sycl::write_only, sycl::no_init};

        shambase::parralel_for(cgh, num_obj,"rescale cells", [=](u64 id_a){
            cell_min[id_a] = acc_p1[id_a].template convert<Tscal>();
            cell_max[id_a] = acc_p2[id_a].template convert<Tscal>();
        });
    });
    
    writer.write_voxel_cells(pos_min_cell,pos_max_cell, num_obj);

    writer.add_cell_data_section();
    writer.add_field_data_section(1);

    std::unique_ptr<sycl::buffer<Tscal>> field_vals = sched.rankgather_field<Tscal>(2);
    writer.write_field("rho", field_vals, num_obj);

}

template<class Tvec, class TgridVec>
auto Model<Tvec, TgridVec>::evolve_once(Tscal t_current,Tscal dt_input)-> Tscal{
    return solver.evolve_once(t_current, dt_input);
}

template class shammodels::basegodunov::Model<f64_3, i64_3>;