// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright(C) 2021-2023 Timothée David--Cléris <timothee.david--cleris@ens-lyon.fr>
// Licensed under CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

#include "shammodels/amr/zeus/Solver.hpp"
#include "shammodels/amr/zeus/modules/AMRTree.hpp"
#include "shammodels/amr/zeus/modules/ComputePressure.hpp"
#include "shammodels/amr/zeus/modules/DiffOperator.hpp"
#include "shammodels/amr/zeus/modules/FaceFlagger.hpp"
#include "shammodels/amr/zeus/modules/GhostZones.hpp"
#include "shammodels/amr/zeus/modules/SourceStep.hpp"
#include "shammodels/amr/zeus/modules/TransportStep.hpp"
#include "shammodels/amr/zeus/modules/ValueLoader.hpp"
#include "shammodels/amr/zeus/modules/WriteBack.hpp"
#include "shamrock/scheduler/SchedulerUtility.hpp"

template<class Tvec, class TgridVec>
using Solver = shammodels::zeus::Solver<Tvec, TgridVec>;

template<class Tvec, class TgridVec>
auto Solver<Tvec, TgridVec>::evolve_once(Tscal t_current, Tscal dt_input) -> Tscal{

    StackEntry stack_loc{};
    shambase::Timer tstep;
    tstep.start();

    SerialPatchTree<TgridVec> _sptree = SerialPatchTree<TgridVec>::build(scheduler());
    _sptree.attach_buf();
    storage.serial_patch_tree.set(std::move(_sptree));

    //ghost zone exchange
    modules::GhostZones gz(context,solver_config,storage);
    gz.build_ghost_cache();

    gz.exchange_ghost();


    //compute bound received
    //round to next pow of 2
    //build radix trees
    modules::AMRTree amrtree(context,solver_config,storage);
    amrtree.build_trees();

    //build neigh table
    amrtree.build_neigh_cache();


    modules::ComputePressure comp_eos(context, solver_config, storage);
    comp_eos.compute_p();
    
    modules::FaceFlagger compute_face_flag(context,solver_config,storage);
    compute_face_flag.flag_faces();
    compute_face_flag.split_face_list();

    
    //modules::DiffOperator diff_op(context,solver_config,storage);
    //diff_op.compute_gradu();





    //save velocity field
    shamrock::patch::PatchDataLayout &ghost_layout = storage.ghost_layout.get();
    u32 irho_interf                                = ghost_layout.get_field_idx<Tscal>("rho");
    u32 ieint_interf                                = ghost_layout.get_field_idx<Tscal>("eint");
    u32 ivel_interf                                = ghost_layout.get_field_idx<Tvec>("vel");

    shamrock::SchedulerUtility utility(scheduler());
    storage.vel_n.set(
        utility.save_field_custom<Tvec>("vel_n", [&](u64 id_patch)-> PatchDataField<Tvec> & {
            using MergedPDat = shamrock::MergedPatchData;
            MergedPDat &mpdat = storage.merged_patchdata_ghost.get().get(id_patch);
            return mpdat.pdat.get_field<Tvec>(ivel_interf);
        })
    );

    //prepare velocity gradients
    modules::ValueLoader<Tvec, TgridVec, Tvec> val_load_vec(context, solver_config, storage);
    storage.vel_n_xp.set( val_load_vec.load_value_with_gz("vel", {1, 0, 0}, "vel_n_xp"));
    storage.vel_n_yp.set( val_load_vec.load_value_with_gz("vel", {0, 1, 0}, "vel_n_yp"));
    storage.vel_n_zp.set( val_load_vec.load_value_with_gz("vel", {0, 0, 1}, "vel_n_zp"));

    modules::ValueLoader<Tvec, TgridVec, Tscal> val_load_scal(context, solver_config, storage);
    storage.rho_n_xm.set( val_load_scal.load_value_with_gz("rho", {-1, 0, 0}, "rho_n_xm"));
    storage.rho_n_ym.set( val_load_scal.load_value_with_gz("rho", {0, -1, 0}, "rho_n_ym"));
    storage.rho_n_zm.set( val_load_scal.load_value_with_gz("rho", {0, 0, -1}, "rho_n_zm"));

    shamrock::ComputeField<Tscal> &pressure_field = storage.pressure.get();
    storage.pres_n_xm.set( val_load_scal.load_value_with_gz(pressure_field, {-1, 0, 0}, "pres_n_xm"));
    storage.pres_n_ym.set( val_load_scal.load_value_with_gz(pressure_field, {0, -1, 0}, "pres_n_ym"));
    storage.pres_n_zm.set( val_load_scal.load_value_with_gz(pressure_field, {0, 0, -1}, "pres_n_zm"));

    modules::SourceStep src_step(context,solver_config,storage);
    src_step.compute_forces();
    src_step.apply_force(dt_input);

    src_step.compute_AV();

    shamrock::ComputeField<Tvec> &q_AV = storage.q_AV.get();
    storage.q_AV_n_xm.set( val_load_vec.load_value_with_gz(q_AV, {-1, 0, 0}, "q_AV_n_xm"));
    storage.q_AV_n_ym.set( val_load_vec.load_value_with_gz(q_AV, {0, -1, 0}, "q_AV_n_ym"));
    storage.q_AV_n_zm.set( val_load_vec.load_value_with_gz(q_AV, {0, 0, -1}, "q_AV_n_zm"));
    
    src_step.apply_AV(dt_input);

    modules::WriteBack wb (context, solver_config,storage);
    wb.write_back_merged_data();


    storage.merged_patchdata_ghost.reset();
    storage.ghost_layout.reset();

    storage.vel_n.reset();
    storage.vel_n_xp.reset();
    storage.vel_n_yp.reset();
    storage.vel_n_zp.reset();

    storage.rho_n_xm.reset();
    storage.rho_n_ym.reset();
    storage.rho_n_zm.reset();

    storage.pres_n_xm.reset();
    storage.pres_n_ym.reset();
    storage.pres_n_zm.reset();

    storage.q_AV.reset();
    storage.q_AV_n_xm.reset();
    storage.q_AV_n_ym.reset();
    storage.q_AV_n_zm.reset();

    
    


    //transport step
    gz.exchange_ghost();

    storage.vel_n_xp.set( val_load_vec.load_value_with_gz("vel", {1, 0, 0}, "vel_n_xp"));
    storage.vel_n_yp.set( val_load_vec.load_value_with_gz("vel", {0, 1, 0}, "vel_n_yp"));
    storage.vel_n_zp.set( val_load_vec.load_value_with_gz("vel", {0, 0, 1}, "vel_n_zp"));

    modules::TransportStep transport (context, solver_config,storage);
    transport.compute_cell_centered_momentas();


    storage.vel_n_xp.reset();
    storage.vel_n_yp.reset();
    storage.vel_n_zp.reset();

    transport.compute_limiter();

    transport.compute_face_centered_moments(dt_input);

    storage.a_x.reset();
    storage.a_y.reset();
    storage.a_z.reset();
    storage.Q_xm.reset();
    storage.Q_ym.reset();
    storage.Q_zm.reset();

    transport.exchange_face_centered_gz();

    transport.compute_flux();

    transport.compute_stencil_flux();

    transport.update_Q(dt_input);

    transport.compute_new_qte();




    wb.write_back_merged_data();



    storage.Q.reset();
    storage.Q_xm.reset();
    storage.Q_ym.reset();
    storage.Q_zm.reset();


    storage.face_lists.reset();
    storage.pressure.reset();
    storage.trees.reset();
    storage.merge_patch_bounds.reset();
    storage.merged_patchdata_ghost.reset();
    storage.ghost_layout.reset();
    storage.ghost_zone_infos.reset();
    storage.serial_patch_tree.reset();


    tstep.end();

    u64 rank_count = scheduler().get_rank_count()*AMRBlock::block_size;
    f64 rate = f64(rank_count) / tstep.elasped_sec();

    std::string log_rank_rate = shambase::format(
        "\n| {:<4} |    {:.4e}    | {:11} |   {:.3e}   |  {:3.0f} % | {:3.0f} % | {:3.0f} % |", 
        shamsys::instance::world_rank,rate,  rank_count,  tstep.elasped_sec(),
        100*(storage.timings_details.interface / tstep.elasped_sec()),
        100*(storage.timings_details.neighbors / tstep.elasped_sec()),
        100*(storage.timings_details.io / tstep.elasped_sec())
        );

    std::string gathered = "";
    shamalgs::collective::gather_str(log_rank_rate, gathered);

    if(shamsys::instance::world_rank == 0){
        std::string print = "processing rate infos : \n";
        print+=("---------------------------------------------------------------------------------\n");
        print+=("| rank |  rate  (N.s^-1)  |      N      | t compute (s) | interf | neigh |   io  |\n");
        print+=("---------------------------------------------------------------------------------");
        print+=(gathered) + "\n";
        print+=("---------------------------------------------------------------------------------");
        logger::info_ln("amr::Zeus",print);
    }

    storage.timings_details.reset();

    return 0;
}


template class shammodels::zeus::Solver<f64_3, i64_3>;