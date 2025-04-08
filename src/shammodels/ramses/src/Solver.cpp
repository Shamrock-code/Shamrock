// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright (c) 2021-2024 Timothée David--Cléris <tim.shamrock@proton.me>
// SPDX-License-Identifier: CeCILL Free Software License Agreement v2.1
// Shamrock is licensed under the CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

/**
 * @file Solver.cpp
 * @author Timothée David--Cléris (timothee.david--cleris@ens-lyon.fr)
 * @brief
 *
 */

#include "shammodels/ramses/Solver.hpp"
#include "shamcomm/collectives.hpp"
#include "shammodels/common/timestep_report.hpp"
#include "shammodels/ramses/modules/AMRGraphGen.hpp"
#include "shammodels/ramses/modules/AMRGridRefinementHandler.hpp"
#include "shammodels/ramses/modules/AMRTree.hpp"
#include "shammodels/ramses/modules/ComputeCFL.hpp"
#include "shammodels/ramses/modules/ComputeCellInfos.hpp"
#include "shammodels/ramses/modules/ComputeFlux.hpp"
#include "shammodels/ramses/modules/ComputeGradient.hpp"
#include "shammodels/ramses/modules/ComputeTimeDerivative.hpp"
#include "shammodels/ramses/modules/ConsToPrim.hpp"
#include "shammodels/ramses/modules/DragIntegrator.hpp"
#include "shammodels/ramses/modules/FaceInterpolate.hpp"
#include "shammodels/ramses/modules/FluxDivergence.hpp"
#include "shammodels/ramses/modules/GhostZones.hpp"
#include "shammodels/ramses/modules/StencilGenerator.hpp"
#include "shammodels/ramses/modules/TimeIntegrator.hpp"
#include "shamrock/io/LegacyVtkWritter.hpp"
#include "shamrock/scheduler/SchedulerUtility.hpp"

template<class Tvec, class TgridVec>
void shammodels::basegodunov::Solver<Tvec, TgridVec>::get_old_fields() {

    StackEntry stack_loc{};

    using MergedPDat = shamrock::MergedPatchData;

    shamrock::SchedulerUtility utility(scheduler());

    shamrock::ComputeField<Tvec> rhovel_old
        = utility.make_compute_field<Tvec>("rhovel", AMRBlock::block_size, [&](u64 id) {
              return storage.merged_patchdata_ghost.get().get(id).total_elements;
          });

    shamrock::ComputeField<Tscal> rhoetot_old
        = utility.make_compute_field<Tscal>("rhoetot", AMRBlock::block_size, [&](u64 id) {
              return storage.merged_patchdata_ghost.get().get(id).total_elements;
          });

    shamrock::ComputeField<Tscal> rho_old
        = utility.make_compute_field<Tscal>("rho", AMRBlock::block_size, [&](u64 id) {
              return storage.merged_patchdata_ghost.get().get(id).total_elements;
          });

    shamrock::patch::PatchDataLayout &ghost_layout = storage.ghost_layout.get();
    u32 irho_ghost                                 = ghost_layout.get_field_idx<Tscal>("rho");
    u32 irhov_ghost                                = ghost_layout.get_field_idx<Tvec>("rhovel");
    u32 irhoe_ghost                                = ghost_layout.get_field_idx<Tscal>("rhoetot");

    storage.merged_patchdata_ghost.get().for_each([&](u64 id, MergedPDat &mpdat) {
        sham::DeviceQueue &q = shamsys::instance::get_compute_scheduler().get_queue();

        sham::DeviceBuffer<TgridVec> &buf_block_min = mpdat.pdat.get_field_buf_ref<TgridVec>(0);
        sham::DeviceBuffer<TgridVec> &buf_block_max = mpdat.pdat.get_field_buf_ref<TgridVec>(1);

        sham::DeviceBuffer<Tscal> &buf_rho  = mpdat.pdat.get_field_buf_ref<Tscal>(irho_ghost);
        sham::DeviceBuffer<Tvec> &buf_rhov  = mpdat.pdat.get_field_buf_ref<Tvec>(irhov_ghost);
        sham::DeviceBuffer<Tscal> &buf_rhoe = mpdat.pdat.get_field_buf_ref<Tscal>(irhoe_ghost);

        sham::DeviceBuffer<Tvec> &buf_rhovel_old   = rhovel_old.get_buf(id);
        sham::DeviceBuffer<Tscal> &buf_rhoetot_old = rhoetot_old.get_buf(id);
        sham::DeviceBuffer<Tscal> &buf_rho_old     = rho_old.get_buf(id);

        sham::EventList depends_list;

        auto rho    = buf_rho.get_read_access(depends_list);
        auto rhovel = buf_rhov.get_read_access(depends_list);
        auto rhoe   = buf_rhoe.get_read_access(depends_list);

        auto rhovel_old  = buf_rhovel_old.get_write_access(depends_list);
        auto rhoetot_old = buf_rhoetot_old.get_write_access(depends_list);
        auto rho_old     = buf_rho_old.get_write_access(depends_list);

        auto e = q.submit(depends_list, [&](sycl::handler &cgh) {
            u32 cell_count = (mpdat.total_elements) * AMRBlock::block_size;

            shambase::parralel_for(cgh, cell_count, "get old state", [=](u64 gid) {
                rho_old[gid]     = rho[gid];
                rhovel_old[gid]  = rhovel[gid];
                rhoetot_old[gid] = rhoe[gid];
            });
        });

        buf_rho.complete_event_state(e);
        buf_rhov.complete_event_state(e);
        buf_rhoe.complete_event_state(e);
        buf_rho_old.complete_event_state(e);
        buf_rhovel_old.complete_event_state(e);
        buf_rhoetot_old.complete_event_state(e);
    });

    storage.rho_old.set(std::move(rho_old));
    storage.rhovel_old.set(std::move(rhovel_old));
    storage.rhoetot_old.set(std::move(rhoetot_old));

    if (solver_config.is_dust_on()) {
        u32 ndust                                      = solver_config.dust_config.ndust;
        shamrock::patch::PatchDataLayout &ghost_layout = storage.ghost_layout.get();

        shamrock::ComputeField<Tvec> rhovel_dust_old = utility.make_compute_field<Tvec>(
            "rhovel_dust", ndust * AMRBlock::block_size, [&](u64 id) {
                return storage.merged_patchdata_ghost.get().get(id).total_elements;
            });

        shamrock::ComputeField<Tscal> rho_dust_old = utility.make_compute_field<Tscal>(
            "rho_dust", ndust * AMRBlock::block_size, [&](u64 id) {
                return storage.merged_patchdata_ghost.get().get(id).total_elements;
            });
        u32 irho_dust_ghost  = ghost_layout.get_field_idx<Tscal>("rho_dust");
        u32 irhov_dust_ghost = ghost_layout.get_field_idx<Tvec>("rhovel_dust");

        storage.merged_patchdata_ghost.get().for_each([&](u64 id, MergedPDat &mpdat) {
            sham::DeviceQueue &q = shamsys::instance::get_compute_scheduler().get_queue();

            sham::DeviceBuffer<TgridVec> &buf_block_min = mpdat.pdat.get_field_buf_ref<TgridVec>(0);
            sham::DeviceBuffer<TgridVec> &buf_block_max = mpdat.pdat.get_field_buf_ref<TgridVec>(1);

            sham::DeviceBuffer<Tscal> &buf_rho_dust
                = mpdat.pdat.get_field_buf_ref<Tscal>(irho_dust_ghost);
            sham::DeviceBuffer<Tvec> &buf_rhov_dust
                = mpdat.pdat.get_field_buf_ref<Tvec>(irhov_dust_ghost);

            sham::DeviceBuffer<Tvec> &buf_rhovel_dust_old = rhovel_dust_old.get_buf(id);
            sham::DeviceBuffer<Tscal> &buf_rho_dust_old   = rho_dust_old.get_buf(id);

            sham::EventList depends_list;

            auto rho_dust    = buf_rho_dust.get_read_access(depends_list);
            auto rhovel_dust = buf_rhov_dust.get_read_access(depends_list);

            auto rhovel_dust_old = buf_rhovel_dust_old.get_write_access(depends_list);
            auto rho_dust_old    = buf_rho_dust_old.get_write_access(depends_list);

            auto e = q.submit(depends_list, [&](sycl::handler &cgh) {
                u32 cell_count = (mpdat.total_elements) * AMRBlock::block_size;

                u32 nvar_dust = ndust;
                shambase::parralel_for(
                    cgh, nvar_dust * cell_count, "get old state dust", [=](u64 gid) {
                        rho_dust_old[gid]    = rho_dust[gid];
                        rhovel_dust_old[gid] = rhovel_dust[gid];
                    });
            });

            buf_rho_dust.complete_event_state(e);
            buf_rhov_dust.complete_event_state(e);
            buf_rho_dust_old.complete_event_state(e);
            buf_rhovel_dust_old.complete_event_state(e);
        });
        storage.rho_dust_old.set(std::move(rho_dust_old));
        storage.rhovel_dust_old.set(std::move(rhovel_dust_old));
    }
}

template<class Tvec, class TgridVec>
void shammodels::basegodunov::Solver<Tvec, TgridVec>::set_old_fields() {

    StackEntry stack_loc{};
    using namespace shamrock::patch;
    using namespace shamrock;
    using namespace shammath;

    shamrock::ComputeField<Tscal> &cfield_rho_old  = storage.rho_old.get();
    shamrock::ComputeField<Tvec> &cfield_rhov_old  = storage.rhovel_old.get();
    shamrock::ComputeField<Tscal> &cfield_rhoe_old = storage.rhoetot_old.get();

    // load layout info
    PatchDataLayout &pdl = scheduler().pdl;

    const u32 icell_min = pdl.get_field_idx<TgridVec>("cell_min");
    const u32 icell_max = pdl.get_field_idx<TgridVec>("cell_max");
    const u32 irho      = pdl.get_field_idx<Tscal>("rho");
    const u32 irhoetot  = pdl.get_field_idx<Tscal>("rhoetot");
    const u32 irhovel   = pdl.get_field_idx<Tvec>("rhovel");

    scheduler().for_each_patchdata_nonempty(
        [&](const shamrock::patch::Patch p, shamrock::patch::PatchData &pdat) {
            logger::debug_ln("[AMR Flux]", "set old fields patch", p.id_patch);

            sham::DeviceQueue &q = shamsys::instance::get_compute_scheduler().get_queue();
            u32 id               = p.id_patch;

            sham::DeviceBuffer<Tscal> &rho_old_patch  = cfield_rho_old.get_buf_check(id);
            sham::DeviceBuffer<Tvec> &rhov_old_patch  = cfield_rhov_old.get_buf_check(id);
            sham::DeviceBuffer<Tscal> &rhoe_old_patch = cfield_rhoe_old.get_buf_check(id);

            u32 cell_count                      = pdat.get_obj_cnt() * AMRBlock::block_size;
            sham::DeviceBuffer<Tscal> &buf_rho  = pdat.get_field_buf_ref<Tscal>(irho);
            sham::DeviceBuffer<Tvec> &buf_rhov  = pdat.get_field_buf_ref<Tvec>(irhovel);
            sham::DeviceBuffer<Tscal> &buf_rhoe = pdat.get_field_buf_ref<Tscal>(irhoetot);

            sham::EventList depends_list;

            auto rho  = buf_rho.get_write_access(depends_list);
            auto rhov = buf_rhov.get_write_access(depends_list);
            auto rhoe = buf_rhoe.get_write_access(depends_list);

            auto rho_old  = rho_old_patch.get_read_access(depends_list);
            auto rhov_old = rhov_old_patch.get_read_access(depends_list);
            auto rhoe_old = rhoe_old_patch.get_read_access(depends_list);

            auto e = q.submit(depends_list, [&](sycl::handler &cgh) {
                shambase::parralel_for(cgh, cell_count, "set old fields", [=](u32 id_a) {
                    rho[id_a]  = rho_old[id_a];
                    rhov[id_a] = rhov[id_a];
                    rhoe[id_a] = rhoe_old[id_a];
                });
            });

            rho_old_patch.complete_event_state(e);
            rhov_old_patch.complete_event_state(e);
            rhoe_old_patch.complete_event_state(e);

            buf_rho.complete_event_state(e);
            buf_rhov.complete_event_state(e);
            buf_rhoe.complete_event_state(e);
        });

    if (solver_config.is_dust_on()) {
        shamrock::ComputeField<Tscal> &cfield_rho_dust_old = storage.rho_dust_old.get();
        shamrock::ComputeField<Tvec> &cfield_rhov_dust_old = storage.rhovel_dust_old.get();

        const u32 irho_dust    = pdl.get_field_idx<Tscal>("rho_dust");
        const u32 irhovel_dust = pdl.get_field_idx<Tvec>("rhovel_dust");

        scheduler().for_each_patchdata_nonempty([&](const shamrock::patch::Patch p,
                                                    shamrock::patch::PatchData &pdat) {
            logger::debug_ln("[AMR Flux]", "set old fields dust patch [dust]", p.id_patch);

            sham::DeviceQueue &q = shamsys::instance::get_compute_scheduler().get_queue();
            u32 id               = p.id_patch;

            sham::DeviceBuffer<Tscal> &rho_dust_old_patch = cfield_rho_dust_old.get_buf_check(id);
            sham::DeviceBuffer<Tvec> &rhov_dust_old_patch = cfield_rhov_dust_old.get_buf_check(id);

            u32 cell_count                          = pdat.get_obj_cnt() * AMRBlock::block_size;
            sham::DeviceBuffer<Tscal> &buf_rho_dust = pdat.get_field_buf_ref<Tscal>(irho_dust);
            sham::DeviceBuffer<Tvec> &buf_rhov_dust = pdat.get_field_buf_ref<Tvec>(irhovel_dust);

            sham::EventList depends_list;

            auto rho_dust  = buf_rho_dust.get_write_access(depends_list);
            auto rhov_dust = buf_rhov_dust.get_write_access(depends_list);

            auto rho_old_dust  = rho_dust_old_patch.get_read_access(depends_list);
            auto rhov_old_dust = rhov_dust_old_patch.get_read_access(depends_list);

            auto e = q.submit(depends_list, [&](sycl::handler &cgh) {
                shambase::parralel_for(cgh, ndust * cell_count, "accumulate fluxes", [=](u32 id_a) {
                    rho_dust[id_a]  = rho_old_dust[id_a];
                    rhov_dust[id_a] = rhov_old_dust[id_a];
                });
            });

            rho_dust_old_patch.complete_event_state(e);
            rhov_dust_old_patch.complete_event_state(e);

            buf_rho_dust.complete_event_state(e);
            buf_rhov_dust.complete_event_state(e);
        });
    }
}

template<class Tvec, class TgridVec>
void shammodels::basegodunov::Solver<Tvec, TgridVec>::reset_old_fields() {

    StackEntry stack_loc{};

    storage.rho_old.reset();
    storage.rhovel_old.reset();
    storage.rhoetot_old.reset();

    if (solver_config.is_dust_on()) {
        storage.rho_dust_old.reset();
        storage.rhovel_dust_old.reset();
    }

    if (solver_config.drag_config.drag_solver_config != DragSolverMode::NoDrag) {
        storage.rho_next_no_drag.reset();
        storage.rhov_next_no_drag.reset();
        storage.rhoe_next_no_drag.reset();
        storage.rho_d_next_no_drag.reset();
        storage.rhov_d_next_no_drag.reset();
    }
}

template<class Tvec, class TgridVec>
void shammodels::basegodunov::Solver<Tvec, TgridVec>::evolve_once() {

    StackEntry stack_loc{};

    sham::MemPerfInfos mem_perf_infos_start = sham::details::get_mem_perf_info();

    Tscal t_current = solver_config.get_time();
    Tscal dt_input  = solver_config.get_dt();

    if (shamcomm::world_rank() == 0) {
        logger::normal_ln("amr::Godunov", shambase::format("t = {}, dt = {}", t_current, dt_input));
    }

    shambase::Timer tstep;
    tstep.start();

    // Scheduler step
    auto update_load_val = [&]() {
        logger::debug_ln("ComputeLoadBalanceValue", "update load balancing");
        scheduler().update_local_load_value([&](shamrock::patch::Patch p) {
            return scheduler().patch_data.owned_data.get(p.id_patch).get_obj_cnt();
        });
    };
    update_load_val();
    scheduler().scheduler_step(true, true);
    update_load_val();
    scheduler().scheduler_step(false, false);

    SerialPatchTree<TgridVec> _sptree = SerialPatchTree<TgridVec>::build(scheduler());
    _sptree.attach_buf();
    storage.serial_patch_tree.set(std::move(_sptree));

    // ghost zone exchange
    modules::GhostZones gz(context, solver_config, storage);
    gz.build_ghost_cache();

    gz.exchange_ghost();

    modules::ComputeCellInfos comp_cell_infos(context, solver_config, storage);
    comp_cell_infos.compute_aabb();

    // compute bound received
    // round to next pow of 2
    // build radix trees
    modules::AMRTree amrtree(context, solver_config, storage);
    amrtree.build_trees();

    amrtree.correct_bounding_box();

    // modules::StencilGenerator stencil_gen(context,solver_config,storage);
    // stencil_gen.make_stencil();

    modules::AMRGraphGen graph_gen(context, solver_config, storage);
    auto block_oriented_graph = graph_gen.find_AMR_block_graph_links_common_face();

    graph_gen.lower_AMR_block_graph_to_cell_common_face_graph(block_oriented_graph);

    // get old fields from shamrock::MergedPatchData
    get_old_fields();

    // time integration and drag integration steps

    if (solver_config.drag_config.drag_solver_config == DragSolverMode::NoDrag) {

        if (solver_config.time_integrator_config.time_integrator == TimeIntegratorMode::MUSCL
            || solver_config.time_integrator_config.time_integrator == TimeIntegratorMode::RK1) {
            modules::FluxDivergence flux_op(context, solver_config, storage);
            flux_op.eval_flux_divergence_hydro_fields();
            if (solver_config.is_dust_on())
                flux_op.eval_flux_divergence_dust_fields();

            modules::TimeIntegrator dt_integ(context, solver_config, storage);
            dt_integ.evolve_old_fields(dt_input, 1.0);

            flux_op.reset_storage_buffers_hydro();
            if (solver_config.is_dust_on()) {
                flux_op.reset_storage_buffers_dust();
            }
        }

        else if (solver_config.time_integrator_config.time_integrator == TimeIntegratorMode::RK2) {
            /***************** First stage::start **************/
            modules::FluxDivergence flux_op(context, solver_config, storage);
            flux_op.eval_flux_divergence_hydro_fields();
            if (solver_config.is_dust_on())
                flux_op.eval_flux_divergence_dust_fields();

            modules::TimeIntegrator dt_integ(context, solver_config, storage);
            dt_integ.evolve_old_fields(dt_input, 1.0);

            /***************** First stage::end **************/

            // ================================================

            // have a reset call here before next flux div call

            flux_op.reset_storage_buffers_hydro();
            if (solver_config.is_dust_on()) {
                flux_op.reset_storage_buffers_dust();
            }

            // ================================================

            /***************** Second stage::start **************/
            flux_op.eval_flux_divergence_hydro_fields();
            if (solver_config.is_dust_on())
                flux_op.eval_flux_divergence_dust_fields();

            dt_integ.evolve_intermediate(dt_input, 0.5, 0.5);
            /***************** Second stage::end **************/

            flux_op.reset_storage_buffers_hydro();
            if (solver_config.is_dust_on()) {
                flux_op.reset_storage_buffers_dust();
            }
        }

        else if (solver_config.time_integrator_config.time_integrator == TimeIntegratorMode::RK3) {
            /***************** First stage::start **************/
            modules::FluxDivergence flux_op(context, solver_config, storage);
            flux_op.eval_flux_divergence_hydro_fields();

            if (solver_config.is_dust_on())
                flux_op.eval_flux_divergence_dust_fields();

            modules::TimeIntegrator dt_integ(context, solver_config, storage);
            dt_integ.evolve_old_fields(dt_input, 1.0);

            /***************** First stage::end **************/

            // ================================================

            // have a reset call here before next flux div call

            flux_op.reset_storage_buffers_hydro();
            if (solver_config.is_dust_on()) {
                flux_op.reset_storage_buffers_dust();
            }

            // ================================================

            /***************** Second stage::start **************/

            flux_op.eval_flux_divergence_hydro_fields();
            if (solver_config.is_dust_on())
                flux_op.eval_flux_divergence_dust_fields();

            dt_integ.evolve_intermediate(dt_input, 3. / 4., 1. / 4.);
            /***************** Second stage::end **************/

            // ================================================

            // have a reset call here before next flux div call

            flux_op.reset_storage_buffers_hydro();
            if (solver_config.is_dust_on()) {
                flux_op.reset_storage_buffers_dust();
            }

            // ================================================

            /***************** Third stage::start **************/

            flux_op.eval_flux_divergence_hydro_fields();
            if (solver_config.is_dust_on())
                flux_op.eval_flux_divergence_dust_fields();

            dt_integ.evolve_intermediate(dt_input, 1. / 3., 2. / 3.);
            /***************** Thrid stage::end **************/

            flux_op.reset_storage_buffers_hydro();
            if (solver_config.is_dust_on()) {
                flux_op.reset_storage_buffers_dust();
            }

        }

        else if (solver_config.time_integrator_config.time_integrator == TimeIntegratorMode::VL2) {
            /***************** First stage::start **************/
            modules::FluxDivergence flux_op(context, solver_config, storage);
            flux_op.eval_flux_divergence_hydro_fields();
            if (solver_config.is_dust_on())
                flux_op.eval_flux_divergence_dust_fields();

            modules::TimeIntegrator dt_integ(context, solver_config, storage);
            dt_integ.evolve_old_fields(dt_input, 0.5);

            /***************** First stage::end **************/

            // ================================================

            // have a reset call here before next flux div call
            flux_op.reset_storage_buffers_hydro();
            if (solver_config.is_dust_on()) {
                flux_op.reset_storage_buffers_dust();
            }

            // ================================================

            /***************** Second stage::start **************/
            flux_op.eval_flux_divergence_hydro_fields();
            if (solver_config.is_dust_on())
                flux_op.eval_flux_divergence_dust_fields();

            dt_integ.evolve_old_fields(dt_input, 1.0);
            /***************** Second stage::end **************/

            flux_op.reset_storage_buffers_hydro();
            if (solver_config.is_dust_on()) {
                flux_op.reset_storage_buffers_dust();
            }
        }

        else {
            shambase::throw_unimplemented();
        }

        // set old fields in patch data. This give the next steps fields after transport step
        set_old_fields();

    }

    else if (solver_config.drag_config.drag_solver_config == DragSolverMode::IRK1) {
        modules::DragIntegrator drag_integ(context, solver_config, storage);
        drag_integ.involve_with_no_src(dt_input);
        drag_integ.enable_irk1_drag_integrator(dt_input);
    } else {
        shambase::throw_unimplemented();
    }

    // reset old fields members of storage
    reset_old_fields();

    /*
        // compute prim variable
        modules::ConsToPrim ctop(context, solver_config, storage);
        ctop.cons_to_prim();

        // compute & limit gradients /
        modules::Slopes slopes(context, solver_config, storage);
        slopes.slope_rho();
        slopes.slope_v();
        slopes.slope_P();
        if (solver_config.is_dust_on()) {
            slopes.slope_rho_dust();
            slopes.slope_v_dust();
        }

        // shift values
        modules::FaceInterpolate face_interpolator(context, solver_config, storage);
        // Tscal dt_face_interp = 0;
        // if (solver_config.face_half_time_interpolation) {
        //     dt_face_interp = dt_input / 2.0;
        // }
        bool is_muscl = solver_config.is_muscl_scheme();

        face_interpolator.interpolate_rho_to_face(dt_input, is_muscl);
        face_interpolator.interpolate_v_to_face(dt_input, is_muscl);
        face_interpolator.interpolate_P_to_face(dt_input, is_muscl);

        if (solver_config.is_dust_on()) {
            face_interpolator.interpolate_rho_dust_to_face(dt_input,is_muscl);
            face_interpolator.interpolate_v_dust_to_face(dt_input,is_muscl);
        }

        // flux
        modules::ComputeFlux flux_compute(context, solver_config, storage);
        flux_compute.compute_flux();
        if (solver_config.is_dust_on()) {
            flux_compute.compute_flux_dust();
        }
        // compute dt fields
        modules::ComputeTimeDerivative dt_compute(context, solver_config, storage);
        dt_compute.compute_dt_fields();
        if (solver_config.is_dust_on()) {
            dt_compute.compute_dt_dust_fields();
        }


        if (solver_config.drag_config.drag_solver_config == DragSolverMode::NoDrag)
        {
            if(solver_config.time_integrator_config.time_integrator == TimeIntegratorMode::MUSCL ||
       solver_config.time_integrator_config.time_integrator == TimeIntegratorMode::RK1)
            {
                modules::TimeIntegrator dt_integ(context, solver_config, storage);
                dt_integ.forward_euler(dt_input);
            }

            else if (solver_config.time_integrator == TimeIntegratorMode::RK2)
            {

            }
        }



        // RK2 + flux lim
        if (solver_config.drag_config.drag_solver_config == DragSolverMode::NoDrag) {
            modules::TimeIntegrator dt_integ(context, solver_config, storage);
            dt_integ.forward_euler(dt_input);
        } else if (solver_config.drag_config.drag_solver_config == DragSolverMode::IRK1) {
            modules::DragIntegrator drag_integ(context, solver_config, storage);
            drag_integ.involve_with_no_src(dt_input);
            drag_integ.enable_irk1_drag_integrator(dt_input);
        } else {
            shambase::throw_unimplemented();
        }

    */

    modules::AMRGridRefinementHandler refinement(context, solver_config, storage);
    refinement.update_refinement();

    modules::ComputeCFL cfl_compute(context, solver_config, storage);
    f64 new_dt = cfl_compute.compute_cfl();

    // if new physics like dust is added then use the smallest dt
    if (solver_config.is_dust_on())
        new_dt = std::min(new_dt, cfl_compute.compute_dust_cfl());

    solver_config.set_next_dt(new_dt);
    solver_config.set_time(t_current + dt_input);

    storage.cell_infos.reset();
    storage.cell_link_graph.reset();

    storage.trees.reset();
    storage.merge_patch_bounds.reset();

    storage.merged_patchdata_ghost.reset();
    storage.ghost_layout.reset();
    storage.ghost_zone_infos.reset();

    storage.serial_patch_tree.reset();

    tstep.end();

    sham::MemPerfInfos mem_perf_infos_end = sham::details::get_mem_perf_info();

    f64 t_dev_alloc
        = (mem_perf_infos_end.time_alloc_device - mem_perf_infos_start.time_alloc_device)
          + (mem_perf_infos_end.time_free_device - mem_perf_infos_start.time_free_device);

    u64 rank_count = scheduler().get_rank_count() * AMRBlock::block_size;
    f64 rate       = f64(rank_count) / tstep.elasped_sec();

    std::string log_step = report_perf_timestep(
        rate,
        rank_count,
        tstep.elasped_sec(),
        storage.timings_details.interface,
        t_dev_alloc,
        mem_perf_infos_end.max_allocated_byte_device);

    if (shamcomm::world_rank() == 0) {
        logger::info_ln("amr::RAMSES", log_step);
        logger::info_ln(
            "amr::RAMSES",
            "estimated rate :",
            dt_input * (3600 / tstep.elasped_sec()),
            "(tsim/hr)");
    }

    storage.timings_details.reset();
}

template<class Tvec, class TgridVec>
void shammodels::basegodunov::Solver<Tvec, TgridVec>::do_debug_vtk_dump(std::string filename) {

    StackEntry stack_loc{};
    shamrock::LegacyVtkWritter writer(filename, true, shamrock::UnstructuredGrid);

    PatchScheduler &sched = shambase::get_check_ref(context.sched);

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
    writer.add_field_data_section(11);

    std::unique_ptr<sycl::buffer<Tscal>> fields_rho = sched.rankgather_field<Tscal>(2);
    writer.write_field("rho", fields_rho, num_obj * block_size);

    std::unique_ptr<sycl::buffer<Tvec>> fields_vel = sched.rankgather_field<Tvec>(3);
    writer.write_field("rhovel", fields_vel, num_obj * block_size);

    std::unique_ptr<sycl::buffer<Tscal>> fields_eint = sched.rankgather_field<Tscal>(4);
    writer.write_field("rhoetot", fields_eint, num_obj * block_size);

    std::unique_ptr<sycl::buffer<Tvec>> grad_rho
        = storage.grad_rho.get().rankgather_computefield(sched);
    writer.write_field("grad_rho", grad_rho, num_obj * block_size);

    std::unique_ptr<sycl::buffer<Tvec>> dx_v = storage.dx_v.get().rankgather_computefield(sched);
    writer.write_field("dx_v", dx_v, num_obj * block_size);

    std::unique_ptr<sycl::buffer<Tvec>> dy_v = storage.dy_v.get().rankgather_computefield(sched);
    writer.write_field("dy_v", dy_v, num_obj * block_size);

    std::unique_ptr<sycl::buffer<Tvec>> dz_v = storage.dz_v.get().rankgather_computefield(sched);
    writer.write_field("dz_v", dz_v, num_obj * block_size);

    std::unique_ptr<sycl::buffer<Tvec>> grad_P
        = storage.grad_P.get().rankgather_computefield(sched);
    writer.write_field("grad_P", grad_P, num_obj * block_size);

    std::unique_ptr<sycl::buffer<Tscal>> dtrho = storage.dtrho.get().rankgather_computefield(sched);
    writer.write_field("dtrho", dtrho, num_obj * block_size);

    std::unique_ptr<sycl::buffer<Tvec>> dtrhov
        = storage.dtrhov.get().rankgather_computefield(sched);
    writer.write_field("dtrhov", dtrhov, num_obj * block_size);

    std::unique_ptr<sycl::buffer<Tscal>> dtrhoe
        = storage.dtrhoe.get().rankgather_computefield(sched);
    writer.write_field("dtrhoe", dtrhoe, num_obj * block_size);
}

template class shammodels::basegodunov::Solver<f64_3, i64_3>;
