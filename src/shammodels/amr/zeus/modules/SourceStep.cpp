// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright(C) 2021-2023 Timothée David--Cléris <timothee.david--cleris@ens-lyon.fr>
// Licensed under CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

#include "shammodels/amr/zeus/modules/SourceStep.hpp"
#include "shammodels/amr/zeus/modules/FaceFlagger.hpp"
#include "shammodels/amr/zeus/modules/ValueLoader.hpp"
#include "shamrock/scheduler/InterfacesUtility.hpp"
#include "shamrock/scheduler/SchedulerUtility.hpp"
#include "shamsys/legacy/log.hpp"

template<class Tvec, class TgridVec>
using Module = shammodels::zeus::modules::SourceStep<Tvec, TgridVec>;

template<class Tvec, class TgridVec>
void Module<Tvec, TgridVec>::compute_forces() {
    StackEntry stack_loc{};

    using namespace shamrock::patch;
    using namespace shamrock;
    using namespace shammath;
    using MergedPDat = shamrock::MergedPatchData;

    using Flagger = FaceFlagger<Tvec, TgridVec>;

    using Block = typename Config::AMRBlock;

    ComputeField<Tscal> &rho_xm = storage.rho_n_xm.get();
    ComputeField<Tscal> &rho_ym = storage.rho_n_ym.get();
    ComputeField<Tscal> &rho_zm = storage.rho_n_zm.get();

    ComputeField<Tscal> &pressure_field = storage.pressure.get();
    ComputeField<Tscal> &p_xm           = storage.pres_n_xm.get();
    ComputeField<Tscal> &p_ym           = storage.pres_n_ym.get();
    ComputeField<Tscal> &p_zm           = storage.pres_n_zm.get();

    shamrock::SchedulerUtility utility(scheduler());
    storage.forces.set(utility.make_compute_field<Tvec>("forces", Block::block_size, [&](u64 id) {
        return storage.merged_patchdata_ghost.get().get(id).total_elements;
    }));

    shamrock::patch::PatchDataLayout &ghost_layout = storage.ghost_layout.get();
    u32 irho_interf                                = ghost_layout.get_field_idx<Tscal>("rho");

    scheduler().for_each_patchdata_nonempty([&](Patch p, PatchData &pdat) {
        MergedPDat &mpdat = storage.merged_patchdata_ghost.get().get(p.id_patch);

        sycl::buffer<TgridVec> &buf_cell_min = mpdat.pdat.get_field_buf_ref<TgridVec>(0);
        sycl::buffer<TgridVec> &buf_cell_max = mpdat.pdat.get_field_buf_ref<TgridVec>(1);

        Tscal coord_conv_fact = solver_config.grid_coord_to_pos_fact/Block::block_size;

        sycl::buffer<Tscal> &buf_p   = pressure_field.get_buf_check(p.id_patch);
        sycl::buffer<Tscal> &buf_rho = mpdat.pdat.get_field_buf_ref<Tscal>(irho_interf);

        sycl::buffer<Tscal> &buf_rho_xm = rho_xm.get_buf_check(p.id_patch);
        sycl::buffer<Tscal> &buf_rho_ym = rho_ym.get_buf_check(p.id_patch);
        sycl::buffer<Tscal> &buf_rho_zm = rho_zm.get_buf_check(p.id_patch);

        sycl::buffer<Tscal> &buf_p_xm = p_xm.get_buf_check(p.id_patch);
        sycl::buffer<Tscal> &buf_p_ym = p_ym.get_buf_check(p.id_patch);
        sycl::buffer<Tscal> &buf_p_zm = p_zm.get_buf_check(p.id_patch);

        sycl::buffer<Tvec> &forces_buf = storage.forces.get().get_buf_check(p.id_patch);

        shamsys::instance::get_compute_queue().submit([&](sycl::handler &cgh) {
            sycl::accessor cell_min{buf_cell_min, cgh, sycl::read_only};
            sycl::accessor cell_max{buf_cell_max, cgh, sycl::read_only};

            sycl::accessor grad_p{forces_buf, cgh, sycl::write_only, sycl::no_init};
            sycl::accessor rho{buf_rho, cgh, sycl::read_only};
            sycl::accessor rho_xm{buf_rho_xm, cgh, sycl::read_only};
            sycl::accessor rho_ym{buf_rho_ym, cgh, sycl::read_only};
            sycl::accessor rho_zm{buf_rho_zm, cgh, sycl::read_only};
            sycl::accessor p{buf_p, cgh, sycl::read_only};
            sycl::accessor p_xm{buf_p_xm, cgh, sycl::read_only};
            sycl::accessor p_ym{buf_p_ym, cgh, sycl::read_only};
            sycl::accessor p_zm{buf_p_zm, cgh, sycl::read_only};

            shambase::parralel_for(
                cgh, mpdat.total_elements * Block::block_size, "compute grad p", [=](u64 id_a) {
                    u32 block_id = id_a / Block::block_size;
                    Tvec d_cell =
                        (cell_max[block_id] - cell_min[block_id]).template convert<Tscal>() *
                        coord_conv_fact;

                    // clang-format off
                Tscal rho_i_j_k   = rho[id_a];
                Tscal rho_im1_j_k = rho_xm[id_a];
                Tscal rho_i_jm1_k = rho_ym[id_a];
                Tscal rho_i_j_km1 = rho_zm[id_a];

                Tscal p_i_j_k   = p[id_a];
                Tscal p_im1_j_k = p_xm[id_a];
                Tscal p_i_jm1_k = p_ym[id_a];
                Tscal p_i_j_km1 = p_zm[id_a];

                Tvec dp = {
                    p_i_j_k - p_im1_j_k,
                    p_i_j_k - p_i_jm1_k, 
                    p_i_j_k - p_i_j_km1
                };

                //sycl::ext::oneapi::experimental::printf("%f %f %f\n", dp.x(),dp.y(),dp.z());

                Tvec avg_rho =
                    Tvec{
                        rho_i_j_k + rho_im1_j_k, 
                        rho_i_j_k + rho_i_jm1_k, 
                        rho_i_j_k + rho_i_j_km1
                        } * Tscal{0.5};

                Tvec grad_p_source_term = dp / (avg_rho * d_cell);

                grad_p[id_a] = grad_p_source_term;
                    // clang-format on
                });
        });
    });

    scheduler().for_each_patchdata_nonempty([&](Patch p, PatchData &pdat) {
        MergedPDat &mpdat = storage.merged_patchdata_ghost.get().get(p.id_patch);

        sycl::buffer<TgridVec> &buf_cell_min = mpdat.pdat.get_field_buf_ref<TgridVec>(0);
        sycl::buffer<TgridVec> &buf_cell_max = mpdat.pdat.get_field_buf_ref<TgridVec>(1);

        sycl::buffer<Tvec> &forces_buf = storage.forces.get().get_buf_check(p.id_patch);

        sycl::buffer<Tscal> &buf_rho = mpdat.pdat.get_field_buf_ref<Tscal>(irho_interf);

        Tscal coord_conv_fact = solver_config.grid_coord_to_pos_fact;

        shamsys::instance::get_compute_queue().submit([&](sycl::handler &cgh) {
            sycl::accessor cell_min{buf_cell_min, cgh, sycl::read_only};
            sycl::accessor cell_max{buf_cell_max, cgh, sycl::read_only};

            sycl::accessor forces{forces_buf, cgh, sycl::read_write};
            sycl::accessor rho{buf_rho, cgh, sycl::read_only};

            shambase::parralel_for(cgh, pdat.get_obj_cnt(), "add ext force", [=](u64 id_a) {
                Tvec block_min    = cell_min[id_a].template convert<Tscal>();
                Tvec block_max    = cell_max[id_a].template convert<Tscal>();
                Tvec delta_cell   = (block_max - block_min) / Block::side_size;
                Tvec delta_cell_h = delta_cell * Tscal(0.5);

                Block::for_each_cell_in_block(delta_cell, [=](u32 lid, Tvec delta) {
                    auto get_ext_force = [](Tvec r) {
                        Tscal d = sycl::length(r);
                        return r / (d * d * d + 1e-5);
                    };

                    //forces[id_a * Block::block_size + lid] +=
                    //    get_ext_force(block_min + delta + delta_cell_h);
                });
            });
        });
        if (storage.forces.get().get_field(p.id_patch).has_nan()) {
            logger::err_ln("[Zeus]", "nan detected in forces");
            throw shambase::throw_with_loc<std::runtime_error>("detected nan");
        }
        //logger::raw_ln(storage.forces.get().get_field(p.id_patch).compute_max());
    });
}

template<class Tvec, class TgridVec>
void Module<Tvec, TgridVec>::apply_force(Tscal dt) {
    StackEntry stack_loc{};

    using namespace shamrock::patch;
    using namespace shamrock;
    using namespace shammath;
    using MergedPDat = shamrock::MergedPatchData;

    using Flagger = FaceFlagger<Tvec, TgridVec>;

    using Block = typename Config::AMRBlock;

    shamrock::patch::PatchDataLayout &ghost_layout = storage.ghost_layout.get();
    u32 ivel_interf                                = ghost_layout.get_field_idx<Tvec>("vel");

    scheduler().for_each_patchdata_nonempty([&](Patch p, PatchData &pdat) {
        MergedPDat &mpdat              = storage.merged_patchdata_ghost.get().get(p.id_patch);
        sycl::buffer<Tvec> &forces_buf = storage.forces.get().get_buf_check(p.id_patch);
        sycl::buffer<Tvec> &vel_buf    = mpdat.pdat.get_field_buf_ref<Tvec>(ivel_interf);

        shamsys::instance::get_compute_queue().submit([&](sycl::handler &cgh) {
            sycl::accessor forces{forces_buf, cgh, sycl::read_only};
            sycl::accessor vel{vel_buf, cgh, sycl::read_write};

            shambase::parralel_for(
                cgh, mpdat.total_elements * Block::block_size, "add ext force", [=](u64 id_a) {
                    vel[id_a] += dt * forces[id_a];
                });
        });

        //logger::raw_ln(storage.forces.get().get_field(p.id_patch).compute_max());
    });

    storage.forces.reset();
}

template<class Tvec, class TgridVec>
void Module<Tvec, TgridVec>::compute_AV() {
    StackEntry stack_loc{};

    using namespace shamrock::patch;
    using namespace shamrock;
    using namespace shammath;
    using MergedPDat = shamrock::MergedPatchData;

    using Block = typename Config::AMRBlock;

    ComputeField<Tvec> &vel_n    = storage.vel_n.get();
    ComputeField<Tvec> &vel_n_xp = storage.vel_n_xp.get();
    ComputeField<Tvec> &vel_n_yp = storage.vel_n_yp.get();
    ComputeField<Tvec> &vel_n_zp = storage.vel_n_zp.get();

    shamrock::SchedulerUtility utility(scheduler());
    storage.q_AV.set(utility.make_compute_field<Tvec>("q_AV", Block::block_size, [&](u64 id) {
        return storage.merged_patchdata_ghost.get().get(id).total_elements;
    }));

    shamrock::patch::PatchDataLayout &ghost_layout = storage.ghost_layout.get();
    u32 irho_interf                                = ghost_layout.get_field_idx<Tscal>("rho");
    u32 ivel_interf                                = ghost_layout.get_field_idx<Tvec>("vel");

    scheduler().for_each_patchdata_nonempty([&](Patch p, PatchData &pdat) {
        MergedPDat &mpdat = storage.merged_patchdata_ghost.get().get(p.id_patch);

        sycl::buffer<TgridVec> &buf_cell_min = mpdat.pdat.get_field_buf_ref<TgridVec>(0);
        sycl::buffer<TgridVec> &buf_cell_max = mpdat.pdat.get_field_buf_ref<TgridVec>(1);

        Tscal coord_conv_fact = solver_config.grid_coord_to_pos_fact/Block::block_size;

        sycl::buffer<Tscal> &buf_rho = mpdat.pdat.get_field_buf_ref<Tscal>(irho_interf);

        sycl::buffer<Tvec> &buf_vel    = vel_n.get_buf_check(p.id_patch);
        sycl::buffer<Tvec> &buf_vel_xp = vel_n_xp.get_buf_check(p.id_patch);
        sycl::buffer<Tvec> &buf_vel_yp = vel_n_yp.get_buf_check(p.id_patch);
        sycl::buffer<Tvec> &buf_vel_zp = vel_n_zp.get_buf_check(p.id_patch);

        sycl::buffer<Tvec> &q_AV_buf = storage.q_AV.get().get_buf_check(p.id_patch);

        shamsys::instance::get_compute_queue().submit([&](sycl::handler &cgh) {
            sycl::accessor cell_min{buf_cell_min, cgh, sycl::read_only};
            sycl::accessor cell_max{buf_cell_max, cgh, sycl::read_only};

            sycl::accessor rho{buf_rho, cgh, sycl::read_only};
            sycl::accessor vel{buf_vel, cgh, sycl::read_only};
            sycl::accessor vel_xp{buf_vel_xp, cgh, sycl::read_only};
            sycl::accessor vel_yp{buf_vel_yp, cgh, sycl::read_only};
            sycl::accessor vel_zp{buf_vel_zp, cgh, sycl::read_only};

            sycl::accessor q_AV{q_AV_buf, cgh, sycl::write_only, sycl::no_init};

            shambase::parralel_for(
                cgh, mpdat.total_elements * Block::block_size, "compute AV", [=](u64 id_a) {
                    u32 block_id = id_a / Block::block_size;
                    Tvec d_cell =
                        (cell_max[block_id] - cell_min[block_id]).template convert<Tscal>() *
                        coord_conv_fact;

                    // clang-format off
                Tscal rho_i_j_k   = rho[id_a];

                Tvec vel_i_j_k = vel[id_a];
                Tvec vel_ip1_j_k = vel_xp[id_a];
                Tvec vel_i_jp1_k = vel_yp[id_a];
                Tvec vel_i_j_kp1 = vel_zp[id_a];

                Tvec dv = {
                    vel_ip1_j_k.x() - vel_i_j_k.x(),
                    vel_i_jp1_k.y() - vel_i_j_k.y(),
                    vel_i_j_kp1.z() - vel_i_j_k.z()
                };

                dv = shambase::sycl_utils::negative_part(dv);

                constexpr Tscal C2 = 3;

                q_AV[id_a] = C2*rho_i_j_k*(dv*dv);
                    // clang-format on
                });
        });
    });
}

template<class Tvec, class TgridVec>
void Module<Tvec, TgridVec>::apply_AV(Tscal dt) {
    StackEntry stack_loc{};

    using namespace shamrock::patch;
    using namespace shamrock;
    using namespace shammath;
    using MergedPDat = shamrock::MergedPatchData;

    using Block = typename Config::AMRBlock;

    ComputeField<Tvec> &q_AV_n    = storage.q_AV.get();
    ComputeField<Tvec> &q_AV_n_xm = storage.q_AV_n_xm.get();
    ComputeField<Tvec> &q_AV_n_ym = storage.q_AV_n_ym.get();
    ComputeField<Tvec> &q_AV_n_zm = storage.q_AV_n_zm.get();

    ComputeField<Tscal> &rho_xm = storage.rho_n_xm.get();
    ComputeField<Tscal> &rho_ym = storage.rho_n_ym.get();
    ComputeField<Tscal> &rho_zm = storage.rho_n_zm.get();

    ComputeField<Tvec> &vel_n    = storage.vel_n.get();
    ComputeField<Tvec> &vel_n_xp = storage.vel_n_xp.get();
    ComputeField<Tvec> &vel_n_yp = storage.vel_n_yp.get();
    ComputeField<Tvec> &vel_n_zp = storage.vel_n_zp.get();

    shamrock::patch::PatchDataLayout &ghost_layout = storage.ghost_layout.get();
    u32 irho_interf                                = ghost_layout.get_field_idx<Tscal>("rho");
    u32 ieint_interf                               = ghost_layout.get_field_idx<Tscal>("eint");
    u32 ivel_interf                                = ghost_layout.get_field_idx<Tvec>("vel");

    scheduler().for_each_patchdata_nonempty([&](Patch p, PatchData &pdat) {
        MergedPDat &mpdat = storage.merged_patchdata_ghost.get().get(p.id_patch);

        sycl::buffer<TgridVec> &buf_cell_min = mpdat.pdat.get_field_buf_ref<TgridVec>(0);
        sycl::buffer<TgridVec> &buf_cell_max = mpdat.pdat.get_field_buf_ref<TgridVec>(1);

        Tscal coord_conv_fact = solver_config.grid_coord_to_pos_fact/Block::block_size;

        sycl::buffer<Tscal> &buf_rho = mpdat.pdat.get_field_buf_ref<Tscal>(irho_interf);
        sycl::buffer<Tvec> &buf_vel  = mpdat.pdat.get_field_buf_ref<Tvec>(ivel_interf);

        sycl::buffer<Tvec> &buf_vel_xp = vel_n_xp.get_buf_check(p.id_patch);
        sycl::buffer<Tvec> &buf_vel_yp = vel_n_yp.get_buf_check(p.id_patch);
        sycl::buffer<Tvec> &buf_vel_zp = vel_n_zp.get_buf_check(p.id_patch);

        sycl::buffer<Tscal> &buf_rho_xm = rho_xm.get_buf_check(p.id_patch);
        sycl::buffer<Tscal> &buf_rho_ym = rho_ym.get_buf_check(p.id_patch);
        sycl::buffer<Tscal> &buf_rho_zm = rho_zm.get_buf_check(p.id_patch);

        sycl::buffer<Tvec> &q_AV_buf      = storage.q_AV.get().get_buf_check(p.id_patch);
        sycl::buffer<Tvec> &buf_q_AV_n_xm = q_AV_n_xm.get_buf_check(p.id_patch);
        sycl::buffer<Tvec> &buf_q_AV_n_ym = q_AV_n_ym.get_buf_check(p.id_patch);
        sycl::buffer<Tvec> &buf_q_AV_n_zm = q_AV_n_zm.get_buf_check(p.id_patch);

        shamsys::instance::get_compute_queue().submit([&](sycl::handler &cgh) {
            sycl::accessor cell_min{buf_cell_min, cgh, sycl::read_only};
            sycl::accessor cell_max{buf_cell_max, cgh, sycl::read_only};

            sycl::accessor rho{buf_rho, cgh, sycl::read_only};

            sycl::accessor rho_xm{buf_rho_xm, cgh, sycl::read_only};
            sycl::accessor rho_ym{buf_rho_ym, cgh, sycl::read_only};
            sycl::accessor rho_zm{buf_rho_zm, cgh, sycl::read_only};

            sycl::accessor q_AV{q_AV_buf, cgh, sycl::read_only};
            sycl::accessor q_AV_xm{buf_q_AV_n_xm, cgh, sycl::read_only};
            sycl::accessor q_AV_ym{buf_q_AV_n_ym, cgh, sycl::read_only};
            sycl::accessor q_AV_zm{buf_q_AV_n_zm, cgh, sycl::read_only};

            sycl::accessor vel{buf_vel, cgh, sycl::read_write};

            shambase::parralel_for(
                cgh, mpdat.total_elements * Block::block_size, "add vel AV", [=](u64 id_a) {
                    u32 block_id = id_a / Block::block_size;
                    Tvec d_cell =
                        (cell_max[block_id] - cell_min[block_id]).template convert<Tscal>() *
                        coord_conv_fact;

                    // clang-format off
                Tscal rho_i_j_k   = rho[id_a];
                Tscal rho_im1_j_k = rho_xm[id_a];
                Tscal rho_i_jm1_k = rho_ym[id_a];
                Tscal rho_i_j_km1 = rho_zm[id_a];

                Tvec q_i_j_k   = q_AV[id_a];
                Tvec q_im1_j_k = q_AV_xm[id_a];
                Tvec q_i_jm1_k = q_AV_ym[id_a];
                Tvec q_i_j_km1 = q_AV_zm[id_a];

                Tvec avg_rho =
                    Tvec{
                        rho_i_j_k + rho_im1_j_k, 
                        rho_i_j_k + rho_i_jm1_k, 
                        rho_i_j_k + rho_i_j_km1
                        } * Tscal{0.5};
                
                Tvec dq = {
                    q_i_j_k.x()-q_im1_j_k.x(),
                    q_i_j_k.y()-q_i_jm1_k.y(),
                    q_i_j_k.z()-q_i_j_km1.z()
                };

                vel[id_a] += - dt*(dq)/ (avg_rho * d_cell);
                    // clang-format on
                });
        });
    });

    scheduler().for_each_patchdata_nonempty([&](Patch p, PatchData &pdat) {
        MergedPDat &mpdat = storage.merged_patchdata_ghost.get().get(p.id_patch);

        sycl::buffer<TgridVec> &buf_cell_min = mpdat.pdat.get_field_buf_ref<TgridVec>(0);
        sycl::buffer<TgridVec> &buf_cell_max = mpdat.pdat.get_field_buf_ref<TgridVec>(1);

        Tscal coord_conv_fact = solver_config.grid_coord_to_pos_fact/Block::block_size;

        sycl::buffer<Tscal> &buf_eint = mpdat.pdat.get_field_buf_ref<Tscal>(ieint_interf);

        sycl::buffer<Tvec> &buf_vel    = vel_n.get_buf_check(p.id_patch);
        sycl::buffer<Tvec> &buf_vel_xp = vel_n_xp.get_buf_check(p.id_patch);
        sycl::buffer<Tvec> &buf_vel_yp = vel_n_yp.get_buf_check(p.id_patch);
        sycl::buffer<Tvec> &buf_vel_zp = vel_n_zp.get_buf_check(p.id_patch);

        sycl::buffer<Tvec> &q_AV_buf      = storage.q_AV.get().get_buf_check(p.id_patch);
        sycl::buffer<Tvec> &buf_q_AV_n_xm = q_AV_n_xm.get_buf_check(p.id_patch);
        sycl::buffer<Tvec> &buf_q_AV_n_ym = q_AV_n_ym.get_buf_check(p.id_patch);
        sycl::buffer<Tvec> &buf_q_AV_n_zm = q_AV_n_zm.get_buf_check(p.id_patch);

        shamsys::instance::get_compute_queue().submit([&](sycl::handler &cgh) {
            sycl::accessor cell_min{buf_cell_min, cgh, sycl::read_only};
            sycl::accessor cell_max{buf_cell_max, cgh, sycl::read_only};

            sycl::accessor vel{buf_vel, cgh, sycl::read_only};
            sycl::accessor vel_xp{buf_vel_xp, cgh, sycl::read_only};
            sycl::accessor vel_yp{buf_vel_yp, cgh, sycl::read_only};
            sycl::accessor vel_zp{buf_vel_zp, cgh, sycl::read_only};

            sycl::accessor q_AV{q_AV_buf, cgh, sycl::read_only};
            sycl::accessor q_AV_xm{buf_q_AV_n_xm, cgh, sycl::read_only};
            sycl::accessor q_AV_ym{buf_q_AV_n_ym, cgh, sycl::read_only};
            sycl::accessor q_AV_zm{buf_q_AV_n_zm, cgh, sycl::read_only};

            sycl::accessor eint{buf_eint, cgh, sycl::read_write};

            shambase::parralel_for(cgh, pdat.get_obj_cnt() * Block::block_size, "add eint AV", [=](u64 id_a) {
                u32 block_id = id_a / Block::block_size;
                Tvec d_cell =
                    (cell_max[block_id] - cell_min[block_id]).template convert<Tscal>() *
                    coord_conv_fact;

                // clang-format off
                Tvec vel_i_j_k = vel[id_a];
                Tvec vel_ip1_j_k = vel_xp[id_a];
                Tvec vel_i_jp1_k = vel_yp[id_a];
                Tvec vel_i_j_kp1 = vel_zp[id_a];

                Tvec q_i_j_k   = q_AV[id_a];

                Tvec dv = {
                    vel_ip1_j_k.x() - vel_i_j_k.x(),
                    vel_i_jp1_k.y() - vel_i_j_k.y(),
                    vel_i_j_kp1.z() - vel_i_j_k.z()
                };

                eint[id_a] += -dt*sycl::dot(q_i_j_k,dv/ d_cell);
                // clang-format on
            });
        });
    });
}

template<class Tvec, class TgridVec>
void Module<Tvec, TgridVec>::compute_div_v(){
    StackEntry stack_loc{};
    using namespace shamrock::patch;
    using namespace shamrock;
    using namespace shammath;
    using MergedPDat = shamrock::MergedPatchData;

    using Flagger = FaceFlagger<Tvec, TgridVec>;

    using Block = typename Config::AMRBlock;

    shamrock::SchedulerUtility utility(scheduler());
    storage.div_v_n.set(utility.make_compute_field<Tscal>("div_v_n", Block::block_size, [&](u64 id) {
        return storage.merged_patchdata_ghost.get().get(id).total_elements;
    }));


    ComputeField<Tvec> &vel_n    = storage.vel_n.get();
    ComputeField<Tvec> &vel_n_xp = storage.vel_n_xp.get();
    ComputeField<Tvec> &vel_n_yp = storage.vel_n_yp.get();
    ComputeField<Tvec> &vel_n_zp = storage.vel_n_zp.get();

    ComputeField<Tscal> & div_v = storage.div_v_n.get();

    scheduler().for_each_patchdata_nonempty([&](Patch p, PatchData &pdat) {
        MergedPDat &mpdat = storage.merged_patchdata_ghost.get().get(p.id_patch);

        sycl::buffer<TgridVec> &buf_cell_min = mpdat.pdat.get_field_buf_ref<TgridVec>(0);
        sycl::buffer<TgridVec> &buf_cell_max = mpdat.pdat.get_field_buf_ref<TgridVec>(1);

        Tscal coord_conv_fact = solver_config.grid_coord_to_pos_fact/Block::block_size;

        sycl::buffer<Tvec> &buf_vel    = vel_n.get_buf_check(p.id_patch);
        sycl::buffer<Tvec> &buf_vel_xp = vel_n_xp.get_buf_check(p.id_patch);
        sycl::buffer<Tvec> &buf_vel_yp = vel_n_yp.get_buf_check(p.id_patch);
        sycl::buffer<Tvec> &buf_vel_zp = vel_n_zp.get_buf_check(p.id_patch);

        sycl::buffer<Tscal> &buf_div_v      = div_v.get_buf_check(p.id_patch);

        shamsys::instance::get_compute_queue().submit([&](sycl::handler &cgh) {
            sycl::accessor cell_min{buf_cell_min, cgh, sycl::read_only};
            sycl::accessor cell_max{buf_cell_max, cgh, sycl::read_only};

            sycl::accessor vel{buf_vel, cgh, sycl::read_only};
            sycl::accessor vel_xp{buf_vel_xp, cgh, sycl::read_only};
            sycl::accessor vel_yp{buf_vel_yp, cgh, sycl::read_only};
            sycl::accessor vel_zp{buf_vel_zp, cgh, sycl::read_only};

            sycl::accessor divv{buf_div_v, cgh, sycl::write_only, sycl::no_init};

            shambase::parralel_for(cgh, pdat.get_obj_cnt() * Block::block_size, "add eint AV", [=](u64 id_a) {
                u32 block_id = id_a / Block::block_size;
                Tvec d_cell =
                    (cell_max[block_id] - cell_min[block_id]).template convert<Tscal>() *
                    coord_conv_fact;

                // clang-format off
                Tvec vel_i_j_k = vel[id_a];
                Tvec vel_ip1_j_k = vel_xp[id_a];
                Tvec vel_i_jp1_k = vel_yp[id_a];
                Tvec vel_i_j_kp1 = vel_zp[id_a];

                Tvec dv = {
                    vel_ip1_j_k.x() - vel_i_j_k.x(),
                    vel_i_jp1_k.y() - vel_i_j_k.y(),
                    vel_i_j_kp1.z() - vel_i_j_k.z()
                };

                divv[id_a] += sycl::dot(dv,Tvec{1,1,1}/ d_cell);
                // clang-format on
            });
        });
    });


} 


template<class Tvec, class TgridVec>
void Module<Tvec, TgridVec>::update_eint_eos(Tscal dt){
    StackEntry stack_loc{};

    using namespace shamrock::patch;
    using namespace shamrock;
    using namespace shammath;
    using MergedPDat = shamrock::MergedPatchData;

    using Block = typename Config::AMRBlock;

    ComputeField<Tscal> & div_v = storage.div_v_n.get();


    shamrock::patch::PatchDataLayout &ghost_layout = storage.ghost_layout.get();
    u32 irho_interf                                = ghost_layout.get_field_idx<Tscal>("rho");
    u32 ieint_interf                               = ghost_layout.get_field_idx<Tscal>("eint");
    u32 ivel_interf                                = ghost_layout.get_field_idx<Tvec>("vel");

    scheduler().for_each_patchdata_nonempty([&](Patch p, PatchData &pdat) {
        MergedPDat &mpdat = storage.merged_patchdata_ghost.get().get(p.id_patch);

        sycl::buffer<TgridVec> &buf_cell_min = mpdat.pdat.get_field_buf_ref<TgridVec>(0);
        sycl::buffer<TgridVec> &buf_cell_max = mpdat.pdat.get_field_buf_ref<TgridVec>(1);

        Tscal coord_conv_fact = solver_config.grid_coord_to_pos_fact/Block::block_size;

        sycl::buffer<Tscal> &buf_eint = mpdat.pdat.get_field_buf_ref<Tscal>(ieint_interf);

        sycl::buffer<Tscal> &buf_divv    = div_v.get_buf_check(p.id_patch);

        shamsys::instance::get_compute_queue().submit([&](sycl::handler &cgh) {
            sycl::accessor cell_min{buf_cell_min, cgh, sycl::read_only};
            sycl::accessor cell_max{buf_cell_max, cgh, sycl::read_only};

            sycl::accessor divv{buf_divv, cgh, sycl::read_only};
            sycl::accessor eint{buf_eint, cgh, sycl::read_write};

            Tscal fact = (dt/2)*(solver_config.eos_gamma -1);

            shambase::parralel_for(cgh, pdat.get_obj_cnt() * Block::block_size, "add eint AV", [=](u64 id_a) {
                u32 block_id = id_a / Block::block_size;
                Tvec d_cell =
                    (cell_max[block_id] - cell_min[block_id]).template convert<Tscal>() *
                    coord_conv_fact;

                Tscal factdivv = divv[id_a]*fact;

                eint[id_a] *= (1-factdivv)/(1+factdivv);
            });
        });
    });

}

template class shammodels::zeus::modules::SourceStep<f64_3, i64_3>;