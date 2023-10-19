// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright(C) 2021-2023 Timothée David--Cléris <timothee.david--cleris@ens-lyon.fr>
// Licensed under CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

/**
 * @file Solver.cpp
 * @author Timothée David--Cléris (timothee.david--cleris@ens-lyon.fr)
 * @brief 
 * 
 */
 
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
#include "shamrock/io/AsciiSplitDump.hpp"
#include "shamrock/scheduler/SchedulerUtility.hpp"

template<class Tvec, class TgridVec>
using Solver = shammodels::zeus::Solver<Tvec, TgridVec>;

template<class Tvec, class TgridVec>
auto Solver<Tvec, TgridVec>::evolve_once(Tscal t_current, Tscal dt_input) -> Tscal{

    StackEntry stack_loc{};

    if(shamsys::instance::world_rank == 0){ 
        logger::normal_ln("amr::Zeus", shambase::format("t = {}, dt = {}", t_current, dt_input));
    }

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

    amrtree.correct_bounding_box();

    //build neigh table
    amrtree.build_neigh_cache();


    modules::ComputePressure comp_eos(context, solver_config, storage);
    comp_eos.compute_p();
    
    modules::FaceFlagger compute_face_flag(context,solver_config,storage);
    compute_face_flag.flag_faces();
    compute_face_flag.split_face_list();

    
    //modules::DiffOperator diff_op(context,solver_config,storage);
    //diff_op.compute_gradu();

    
    using namespace shamrock::patch;
    using namespace shamrock;
    using Block = typename Config::AMRBlock;
    AsciiSplitDump debug_dump("ghost_dump_debug"+std::to_string(t_current));

    bool do_debug_dump = false;

    if(do_debug_dump){
    scheduler().for_each_patchdata_nonempty([&](Patch p, PatchData &pdat) {
        debug_dump.create_id(p.id_patch);
    });

    scheduler().for_each_patchdata_nonempty([&](Patch p, PatchData &pdat) {
        using MergedPDat = shamrock::MergedPatchData;
        MergedPDat &mpdat = storage.merged_patchdata_ghost.get().get(p.id_patch);
        debug_dump.get_file(p.id_patch).change_table_name("Nobj_original", "u32");
        debug_dump.get_file(p.id_patch).write_val(mpdat.original_elements);
        debug_dump.get_file(p.id_patch).change_table_name("Nobj_total", "u32");
        debug_dump.get_file(p.id_patch).write_val(mpdat.total_elements);
    });

    scheduler().for_each_patchdata_nonempty([&](Patch p, PatchData &pdat) {
        using MergedPDat = shamrock::MergedPatchData;
        MergedPDat &mpdat = storage.merged_patchdata_ghost.get().get(p.id_patch);


        shamrock::patch::PatchDataLayout &ghost_layout = storage.ghost_layout.get();
        u32 irho_interf                                = ghost_layout.get_field_idx<Tscal>("rho");
        u32 ieint_interf                                = ghost_layout.get_field_idx<Tscal>("eint");
        u32 ivel_interf                                = ghost_layout.get_field_idx<Tvec>("vel");

        sycl::buffer<TgridVec> &cell_min = mpdat.pdat.get_field_buf_ref<TgridVec>(0);
        sycl::buffer<TgridVec> &cell_max = mpdat.pdat.get_field_buf_ref<TgridVec>(1);

        sycl::buffer<Tscal> &rho_merged = mpdat.pdat.get_field_buf_ref<Tscal>(irho_interf);
        sycl::buffer<Tscal> &eint_merged = mpdat.pdat.get_field_buf_ref<Tscal>(ieint_interf);
        sycl::buffer<Tvec> &vel_merged = mpdat.pdat.get_field_buf_ref<Tvec>(ivel_interf);

        debug_dump.get_file(p.id_patch).change_table_name("cell_min", "i64_3");
        debug_dump.get_file(p.id_patch).write_table(cell_min, mpdat.total_elements);
        debug_dump.get_file(p.id_patch).change_table_name("cell_max", "i64_3");
        debug_dump.get_file(p.id_patch).write_table(cell_max, mpdat.total_elements);

        debug_dump.get_file(p.id_patch).change_table_name("rho", "f64");
        debug_dump.get_file(p.id_patch).write_table(rho_merged, mpdat.total_elements*AMRBlock::block_size);
        debug_dump.get_file(p.id_patch).change_table_name("eint", "f64");
        debug_dump.get_file(p.id_patch).write_table(eint_merged, mpdat.total_elements*AMRBlock::block_size);
        debug_dump.get_file(p.id_patch).change_table_name("vel", "f64_3");
        debug_dump.get_file(p.id_patch).write_table(vel_merged, mpdat.total_elements*AMRBlock::block_size);

    });
    }


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

    if(do_debug_dump){
        scheduler().for_each_patchdata_nonempty([&](Patch p, PatchData &pdat) {
            using MergedPDat = shamrock::MergedPatchData;
            MergedPDat &mpdat = storage.merged_patchdata_ghost.get().get(p.id_patch);

            sycl::buffer<Tvec> &forces_buf = storage.forces.get().get_buf_check(p.id_patch);

            debug_dump.get_file(p.id_patch).change_table_name("force_press", "f64_3");
            debug_dump.get_file(p.id_patch).write_table(forces_buf, mpdat.total_elements*AMRBlock::block_size);

        });
    }


    src_step.apply_force(dt_input);



    src_step.compute_AV();

    shamrock::ComputeField<Tvec> &q_AV = storage.q_AV.get();
    storage.q_AV_n_xm.set( val_load_vec.load_value_with_gz(q_AV, {-1, 0, 0}, "q_AV_n_xm"));
    storage.q_AV_n_ym.set( val_load_vec.load_value_with_gz(q_AV, {0, -1, 0}, "q_AV_n_ym"));
    storage.q_AV_n_zm.set( val_load_vec.load_value_with_gz(q_AV, {0, 0, -1}, "q_AV_n_zm"));
    
    src_step.apply_AV(dt_input);
    if(do_debug_dump){
        scheduler().for_each_patchdata_nonempty([&](Patch p, PatchData &pdat) {
            using MergedPDat = shamrock::MergedPatchData;
            MergedPDat &mpdat = storage.merged_patchdata_ghost.get().get(p.id_patch);


            shamrock::patch::PatchDataLayout &ghost_layout = storage.ghost_layout.get();
            u32 irho_interf                                = ghost_layout.get_field_idx<Tscal>("rho");
            u32 ieint_interf                                = ghost_layout.get_field_idx<Tscal>("eint");
            u32 ivel_interf                                = ghost_layout.get_field_idx<Tvec>("vel");

            sycl::buffer<TgridVec> &cell_min = mpdat.pdat.get_field_buf_ref<TgridVec>(0);
            sycl::buffer<TgridVec> &cell_max = mpdat.pdat.get_field_buf_ref<TgridVec>(1);

            sycl::buffer<Tscal> &rho_merged = mpdat.pdat.get_field_buf_ref<Tscal>(irho_interf);
            sycl::buffer<Tscal> &eint_merged = mpdat.pdat.get_field_buf_ref<Tscal>(ieint_interf);
            sycl::buffer<Tvec> &vel_merged = mpdat.pdat.get_field_buf_ref<Tvec>(ivel_interf);

            debug_dump.get_file(p.id_patch).change_table_name("eint_post_source", "f64");
            debug_dump.get_file(p.id_patch).write_table(eint_merged, mpdat.total_elements*AMRBlock::block_size);
            debug_dump.get_file(p.id_patch).change_table_name("vel_post_source", "f64_3");
            debug_dump.get_file(p.id_patch).write_table(vel_merged, mpdat.total_elements*AMRBlock::block_size);

        });
    }


    src_step.compute_div_v();
    src_step.update_eint_eos(dt_input);

    if(do_debug_dump){
        scheduler().for_each_patchdata_nonempty([&](Patch p, PatchData &pdat) {
            using MergedPDat = shamrock::MergedPatchData;
            MergedPDat &mpdat = storage.merged_patchdata_ghost.get().get(p.id_patch);

            sycl::buffer<Tscal> &divv = storage.div_v_n.get().get_buf_check(p.id_patch);

            debug_dump.get_file(p.id_patch).change_table_name("divv_source", "f64");
            debug_dump.get_file(p.id_patch).write_table(divv, mpdat.total_elements*AMRBlock::block_size);

        });
    }

    storage.div_v_n.reset();

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

    if(do_debug_dump){
    scheduler().for_each_patchdata_nonempty([&](Patch p, PatchData &pdat) {
        using MergedPDat = shamrock::MergedPatchData;
        MergedPDat &mpdat = storage.merged_patchdata_ghost.get().get(p.id_patch);


        shamrock::patch::PatchDataLayout &ghost_layout = storage.ghost_layout.get();
        u32 irho_interf                                = ghost_layout.get_field_idx<Tscal>("rho");
        u32 ieint_interf                                = ghost_layout.get_field_idx<Tscal>("eint");
        u32 ivel_interf                                = ghost_layout.get_field_idx<Tvec>("vel");

        sycl::buffer<TgridVec> &cell_min = mpdat.pdat.get_field_buf_ref<TgridVec>(0);
        sycl::buffer<TgridVec> &cell_max = mpdat.pdat.get_field_buf_ref<TgridVec>(1);

        sycl::buffer<Tscal> &rho_merged = mpdat.pdat.get_field_buf_ref<Tscal>(irho_interf);
        sycl::buffer<Tscal> &eint_merged = mpdat.pdat.get_field_buf_ref<Tscal>(ieint_interf);
        sycl::buffer<Tvec> &vel_merged = mpdat.pdat.get_field_buf_ref<Tvec>(ivel_interf);

        debug_dump.get_file(p.id_patch).change_table_name("eint_start_transp", "f64");
        debug_dump.get_file(p.id_patch).write_table(eint_merged, mpdat.total_elements*AMRBlock::block_size);
        debug_dump.get_file(p.id_patch).change_table_name("vel_start_transp", "f64_3");
        debug_dump.get_file(p.id_patch).write_table(vel_merged, mpdat.total_elements*AMRBlock::block_size);

    });
    }


    modules::ValueLoader<Tvec, TgridVec, Tvec> val_load_vec_v2(context, solver_config, storage);
    storage.vel_n_xp.set( val_load_vec_v2.load_value_with_gz("vel", {1, 0, 0}, "vel_n_xp"));
    storage.vel_n_yp.set( val_load_vec_v2.load_value_with_gz("vel", {0, 1, 0}, "vel_n_yp"));
    storage.vel_n_zp.set( val_load_vec_v2.load_value_with_gz("vel", {0, 0, 1}, "vel_n_zp"));

    if(do_debug_dump){
    scheduler().for_each_patchdata_nonempty([&](Patch p, PatchData &pdat) {
        using MergedPDat = shamrock::MergedPatchData;
        MergedPDat &mpdat = storage.merged_patchdata_ghost.get().get(p.id_patch);

        sycl::buffer<Tvec> &vel_n_xp = storage.vel_n_xp.get().get_buf_check(p.id_patch);

        debug_dump.get_file(p.id_patch).change_table_name("vel_n_xp", "f64_3");
        debug_dump.get_file(p.id_patch).write_table(vel_n_xp, mpdat.total_elements*AMRBlock::block_size);

    });
    }

    /*
    using namespace shamrock::patch;
    using namespace shamrock;
    using Block = typename Config::AMRBlock;
    scheduler().for_each_patchdata_nonempty([&](Patch p, PatchData &pdat) {

        using MergedPDat = shamrock::MergedPatchData;
        MergedPDat &mpdat = storage.merged_patchdata_ghost.get().get(p.id_patch);

        sycl::buffer<Tscal> &rho_merged = mpdat.pdat.get_field_buf_ref<Tscal>(irho_interf);
        sycl::buffer<Tscal> &eint_merged = mpdat.pdat.get_field_buf_ref<Tscal>(ieint_interf);
        sycl::buffer<Tvec> &vel_merged = mpdat.pdat.get_field_buf_ref<Tvec>(ivel_interf);

        PatchData &patch_dest = scheduler().patch_data.get_pdat(p.id_patch);
        sycl::buffer<Tscal> &rho_dest = patch_dest.get_field_buf_ref<Tscal>(irho_interf);
        sycl::buffer<Tscal> &eint_dest = patch_dest.get_field_buf_ref<Tscal>(ieint_interf);
        sycl::buffer<Tvec> &vel_dest = patch_dest.get_field_buf_ref<Tvec>(ivel_interf);


        sycl::buffer<Tvec> &forces_buf = storage.vel_n_xp.get().get_buf_check(p.id_patch);
        //sycl::buffer<Tscal> & tmp = storage.pres_n_xm.get().get_buf_check(p.id_patch);
        //sycl::buffer<sycl::vec<Tscal, 8>> & Q_tmp = storage.Q.get().get_buf_check(p.id_patch);
        sycl::buffer<Tscal> &buf_p   = pressure_field.get_buf_check(p.id_patch);

        shamsys::instance::get_compute_queue().submit([&](sycl::handler & cgh){

            sycl::accessor acc_rho_src{buf_p, cgh, sycl::read_only};
            sycl::accessor acc_eint_src{eint_merged, cgh, sycl::read_only};
            sycl::accessor acc_vel_src{vel_merged, cgh, sycl::read_only};
            sycl::accessor acc_vel_src_xp{forces_buf, cgh, sycl::read_only};
            //sycl::accessor Q{Q_tmp, cgh, sycl::read_only};

            sycl::accessor acc_rho_dest{rho_dest, cgh, sycl::write_only};
            sycl::accessor acc_eint_dest{eint_dest, cgh, sycl::write_only};
            sycl::accessor acc_vel_dest{vel_dest, cgh, sycl::write_only};

            shambase::parralel_for(cgh, mpdat.original_elements*Block::block_size, "tmp copy_ack", [=](u32 id){
                //acc_rho_dest[id] = acc_rho_src[id];
                acc_eint_dest[id] = acc_vel_src_xp[id].x();
                acc_vel_dest[id] = acc_vel_src[id];
            });
        });

        if (mpdat.pdat.has_nan()) {
            logger::err_ln("[Zeus]", "nan detected in write back");
            throw shambase::throw_with_loc<std::runtime_error>("detected nan");
        }

    });

    return 0;
    */

    modules::TransportStep transport (context, solver_config,storage);
    transport.compute_cell_centered_momentas();

    if(do_debug_dump){
    using Tscal8 = sycl::vec<Tscal, 8>;
    scheduler().for_each_patchdata_nonempty([&](Patch p, PatchData &pdat) {
        using MergedPDat = shamrock::MergedPatchData;
        MergedPDat &mpdat = storage.merged_patchdata_ghost.get().get(p.id_patch);

        sycl::buffer<Tscal8> &Q_buf = storage.Q.get().get_buf_check(p.id_patch);

        debug_dump.get_file(p.id_patch).change_table_name("Q", "f64_8");
        debug_dump.get_file(p.id_patch).write_table(Q_buf, mpdat.total_elements*AMRBlock::block_size);

    });
    }


    storage.vel_n_xp.reset();
    storage.vel_n_yp.reset();
    storage.vel_n_zp.reset();

    transport.compute_limiter();

    if(do_debug_dump){
    using Tscal8 = sycl::vec<Tscal, 8>;
    scheduler().for_each_patchdata_nonempty([&](Patch p, PatchData &pdat) {
        using MergedPDat = shamrock::MergedPatchData;
        MergedPDat &mpdat = storage.merged_patchdata_ghost.get().get(p.id_patch);

        sycl::buffer<Tscal8> &ax_buf = storage.a_x.get().get_buf_check(p.id_patch);
        sycl::buffer<Tscal8> &ay_buf = storage.a_y.get().get_buf_check(p.id_patch);
        sycl::buffer<Tscal8> &az_buf = storage.a_z.get().get_buf_check(p.id_patch);

        debug_dump.get_file(p.id_patch).change_table_name("ax", "f64_8");
        debug_dump.get_file(p.id_patch).write_table(ax_buf, mpdat.total_elements*AMRBlock::block_size);
        debug_dump.get_file(p.id_patch).change_table_name("ay", "f64_8");
        debug_dump.get_file(p.id_patch).write_table(ay_buf, mpdat.total_elements*AMRBlock::block_size);
        debug_dump.get_file(p.id_patch).change_table_name("az", "f64_8");
        debug_dump.get_file(p.id_patch).write_table(az_buf, mpdat.total_elements*AMRBlock::block_size);

    });
    }

    transport.compute_face_centered_moments(dt_input);

    storage.a_x.reset();
    storage.a_y.reset();
    storage.a_z.reset();
    storage.Q_xm.reset();
    storage.Q_ym.reset();
    storage.Q_zm.reset();

    transport.exchange_face_centered_gz();

    if(do_debug_dump){
    using Tscal8 = sycl::vec<Tscal, 8>;
    scheduler().for_each_patchdata_nonempty([&](Patch p, PatchData &pdat) {
        using MergedPDat = shamrock::MergedPatchData;
        MergedPDat &mpdat = storage.merged_patchdata_ghost.get().get(p.id_patch);

        sycl::buffer<Tscal8> &Qstarx_buf = storage.Qstar_x.get().get_buf_check(p.id_patch);
        sycl::buffer<Tscal8> &Qstary_buf = storage.Qstar_y.get().get_buf_check(p.id_patch);
        sycl::buffer<Tscal8> &Qstarz_buf = storage.Qstar_z.get().get_buf_check(p.id_patch);

        debug_dump.get_file(p.id_patch).change_table_name("Qstar_x", "f64_8");
        debug_dump.get_file(p.id_patch).write_table(Qstarx_buf, mpdat.total_elements*AMRBlock::block_size);
        debug_dump.get_file(p.id_patch).change_table_name("Qstar_y", "f64_8");
        debug_dump.get_file(p.id_patch).write_table(Qstary_buf, mpdat.total_elements*AMRBlock::block_size);
        debug_dump.get_file(p.id_patch).change_table_name("Qstar_z", "f64_8");
        debug_dump.get_file(p.id_patch).write_table(Qstarz_buf, mpdat.total_elements*AMRBlock::block_size);

    });
    }

    transport.compute_flux();

    if(do_debug_dump){
    using Tscal8 = sycl::vec<Tscal, 8>;
    scheduler().for_each_patchdata_nonempty([&](Patch p, PatchData &pdat) {
        using MergedPDat = shamrock::MergedPatchData;
        MergedPDat &mpdat = storage.merged_patchdata_ghost.get().get(p.id_patch);

        sycl::buffer<Tscal8> &Fluxx_buf = storage.Flux_x.get().get_buf_check(p.id_patch);
        sycl::buffer<Tscal8> &Fluxy_buf = storage.Flux_y.get().get_buf_check(p.id_patch);
        sycl::buffer<Tscal8> &Fluxz_buf = storage.Flux_z.get().get_buf_check(p.id_patch);

        debug_dump.get_file(p.id_patch).change_table_name("Flux_x", "f64_8");
        debug_dump.get_file(p.id_patch).write_table(Fluxx_buf, mpdat.total_elements*AMRBlock::block_size);
        debug_dump.get_file(p.id_patch).change_table_name("Flux_y", "f64_8");
        debug_dump.get_file(p.id_patch).write_table(Fluxy_buf, mpdat.total_elements*AMRBlock::block_size);
        debug_dump.get_file(p.id_patch).change_table_name("Flux_z", "f64_8");
        debug_dump.get_file(p.id_patch).write_table(Fluxz_buf, mpdat.total_elements*AMRBlock::block_size);

    });
    }

    transport.compute_stencil_flux();

    transport.update_Q(dt_input);

    transport.compute_new_qte();


    if(do_debug_dump){
    scheduler().for_each_patchdata_nonempty([&](Patch p, PatchData &pdat) {
        using MergedPDat = shamrock::MergedPatchData;
        MergedPDat &mpdat = storage.merged_patchdata_ghost.get().get(p.id_patch);

        shamrock::patch::PatchDataLayout &ghost_layout = storage.ghost_layout.get();
        u32 irho_interf                                = ghost_layout.get_field_idx<Tscal>("rho");
        u32 ieint_interf                                = ghost_layout.get_field_idx<Tscal>("eint");
        u32 ivel_interf                                = ghost_layout.get_field_idx<Tvec>("vel");

        sycl::buffer<Tscal> &rho_merged = mpdat.pdat.get_field_buf_ref<Tscal>(irho_interf);
        sycl::buffer<Tscal> &eint_merged = mpdat.pdat.get_field_buf_ref<Tscal>(ieint_interf);
        sycl::buffer<Tvec> &vel_merged = mpdat.pdat.get_field_buf_ref<Tvec>(ivel_interf);

        debug_dump.get_file(p.id_patch).change_table_name("rho_end_transp", "f64");
        debug_dump.get_file(p.id_patch).write_table(rho_merged, mpdat.total_elements*AMRBlock::block_size);
        debug_dump.get_file(p.id_patch).change_table_name("eint_end_transp", "f64");
        debug_dump.get_file(p.id_patch).write_table(eint_merged, mpdat.total_elements*AMRBlock::block_size);
        debug_dump.get_file(p.id_patch).change_table_name("vel_end_transp", "f64_3");
        debug_dump.get_file(p.id_patch).write_table(vel_merged, mpdat.total_elements*AMRBlock::block_size);

    });
    }

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



    if(do_debug_dump){
    scheduler().for_each_patchdata_nonempty([&](Patch p, PatchData &pdat) {
        debug_dump.get_file(p.id_patch).close();
    });
    }

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