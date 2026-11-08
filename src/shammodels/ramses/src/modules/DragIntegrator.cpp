// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright (c) 2021-2026 Timothée David--Cléris <tim.shamrock@proton.me>
// SPDX-License-Identifier: CeCILL Free Software License Agreement v2.1
// Shamrock is licensed under the CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

/**
 * @file DragIntegrator.cpp
 * @author Anass Serhani (anass.serhani@cnrs.fr)
 * @author Léodasce Sewanou (leodasce.sewanou@ens-lyon.fr)
 * @author Timothée David--Cléris (tim.shamrock@proton.me)
 * @brief
 *
 */
#include "shambase/exception.hpp"
#include "shambase/string.hpp"
#include "shambackends/DeviceBuffer.hpp"
#include "shambackends/DeviceScheduler.hpp"
#include "shambackends/EventList.hpp"
#include "shambackends/kernel_call.hpp"
#include "shammath/matrix_exponential.hpp"
#include "shammodels/ramses/modules/ApplyDragExpo.hpp"
#include "shammodels/ramses/modules/DragIntegrator.hpp"
#include "shamrock/scheduler/SchedulerUtility.hpp"
#include "shamrock/solvergraph/FieldRefs.hpp"
#include "shamrock/solvergraph/GetFieldRefFromLayer.hpp"
#include "shamrock/solvergraph/ScalarEdge.hpp"
#include "shamsolvergraph/node/OperationSequence.hpp"
#include "shamsys/NodeInstance.hpp"
#include <memory>
#include <stdexcept>

template<class Tvec, class TgridVec>
void shammodels::basegodunov::modules::DragIntegrator<Tvec, TgridVec>::involve_with_no_src(
    Tscal dt) {

    StackEntry stack_lock{};

    using namespace shamrock::patch;
    using namespace shamrock;
    using namespace shammath;

    const u32 ndust = solver_config.dust_config.ndust;

    SchedulerUtility utility(scheduler());
    shamrock::ComputeField<Tscal> cfield_rho_next_bf_drag
        = utility.make_compute_field<Tscal>("rho_next_bf_drag", AMRBlock::block_size);
    shamrock::ComputeField<Tvec> cfield_rhov_next_bf_drag
        = utility.make_compute_field<Tvec>("rhov_next_bf_drag", AMRBlock::block_size);
    shamrock::ComputeField<Tscal> cfield_rhoe_next_bf_drag
        = utility.make_compute_field<Tscal>("rhoe_next_bf_drag", AMRBlock::block_size);
    shamrock::ComputeField<Tscal> cfield_rho_d_next_bf_drag
        = utility.make_compute_field<Tscal>("rho_d_next_bf_drag", ndust * AMRBlock::block_size);
    shamrock::ComputeField<Tvec> cfield_rhov_d_next_bf_drag
        = utility.make_compute_field<Tvec>("rhov_d_next_bf_drag", ndust * AMRBlock::block_size);

    shamrock::solvergraph::Field<Tscal> &cfield_dtrho  = shambase::get_check_ref(storage.dtrho);
    shamrock::solvergraph::Field<Tvec> &cfield_dtrhov  = shambase::get_check_ref(storage.dtrhov);
    shamrock::solvergraph::Field<Tscal> &cfield_dtrhoe = shambase::get_check_ref(storage.dtrhoe);
    shamrock::solvergraph::Field<Tscal> &cfield_dtrho_d
        = shambase::get_check_ref(storage.dtrho_dust);
    shamrock::solvergraph::Field<Tvec> &cfield_dtrhov_d
        = shambase::get_check_ref(storage.dtrhov_dust);

    // load layout info
    PatchDataLayerLayout &pdl = scheduler().pdl_old();

    const u32 icell_min = pdl.get_field_idx<TgridVec>("cell_min");
    const u32 icell_max = pdl.get_field_idx<TgridVec>("cell_max");
    const u32 irho      = pdl.get_field_idx<Tscal>("rho");
    const u32 irhoetot  = pdl.get_field_idx<Tscal>("rhoetot");
    const u32 irhovel   = pdl.get_field_idx<Tvec>("rhovel");
    const u32 irho_d    = pdl.get_field_idx<Tscal>("rho_dust");
    const u32 irhovel_d = pdl.get_field_idx<Tvec>("rhovel_dust");

    scheduler().for_each_patchdata_nonempty([&, dt, ndust](
                                                const shamrock::patch::Patch p,
                                                shamrock::patch::PatchDataLayer &pdat) {
        shamlog_debug_ln(
            "[AMR evolve time step before drag ]", "evolve field with no drag patch", p.id_patch);

        sham::DeviceQueue &q = shamsys::instance::get_compute_scheduler().get_queue();
        u32 id               = p.id_patch;

        sham::DeviceBuffer<Tscal> &dt_rho_patch   = cfield_dtrho.get_buf(id);
        sham::DeviceBuffer<Tvec> &dt_rhov_patch   = cfield_dtrhov.get_buf(id);
        sham::DeviceBuffer<Tscal> &dt_rhoe_patch  = cfield_dtrhoe.get_buf(id);
        sham::DeviceBuffer<Tscal> &dt_rho_d_patch = cfield_dtrho_d.get_buf(id);
        sham::DeviceBuffer<Tvec> &dt_rhov_d_patch = cfield_dtrhov_d.get_buf(id);

        sham::DeviceBuffer<Tscal> &buf_rho   = pdat.get_field_buf_ref<Tscal>(irho);
        sham::DeviceBuffer<Tvec> &buf_rhov   = pdat.get_field_buf_ref<Tvec>(irhovel);
        sham::DeviceBuffer<Tscal> &buf_rhoe  = pdat.get_field_buf_ref<Tscal>(irhoetot);
        sham::DeviceBuffer<Tscal> &buf_rho_d = pdat.get_field_buf_ref<Tscal>(irho_d);
        sham::DeviceBuffer<Tvec> &buf_rhov_d = pdat.get_field_buf_ref<Tvec>(irhovel_d);

        sham::DeviceBuffer<Tscal> &rho_patch   = cfield_rho_next_bf_drag.get_buf_check(id);
        sham::DeviceBuffer<Tvec> &rhov_patch   = cfield_rhov_next_bf_drag.get_buf_check(id);
        sham::DeviceBuffer<Tscal> &rhoe_patch  = cfield_rhoe_next_bf_drag.get_buf_check(id);
        sham::DeviceBuffer<Tscal> &rho_d_patch = cfield_rho_d_next_bf_drag.get_buf_check(id);
        sham::DeviceBuffer<Tvec> &rhov_d_patch = cfield_rhov_d_next_bf_drag.get_buf_check(id);

        u32 cell_count = pdat.get_obj_cnt() * AMRBlock::block_size;

        sham::kernel_call(
            q,
            sham::MultiRef{dt_rho_patch, dt_rhov_patch, dt_rhoe_patch, buf_rho, buf_rhov, buf_rhoe},
            sham::MultiRef{rho_patch, rhov_patch, rhoe_patch},
            cell_count,
            [dt](
                u32 id_a,
                const Tscal *__restrict acc_dt_rho_patch,
                const Tvec *__restrict acc_dt_rhov_patch,
                const Tscal *__restrict acc_dt_rhoe_patch,
                const Tscal *__restrict rho,
                const Tvec *__restrict rhov,
                const Tscal *__restrict rhoe,
                Tscal *__restrict acc_rho,
                Tvec *__restrict acc_rhov,
                Tscal *__restrict acc_rhoe) {
                acc_rho[id_a]  = rho[id_a] + dt * acc_dt_rho_patch[id_a];
                acc_rhov[id_a] = rhov[id_a] + dt * acc_dt_rhov_patch[id_a];
                acc_rhoe[id_a] = rhoe[id_a] + dt * acc_dt_rhoe_patch[id_a];
            });

        sham::kernel_call(
            q,
            sham::MultiRef{dt_rho_d_patch, dt_rhov_d_patch, buf_rho_d, buf_rhov_d},
            sham::MultiRef{rho_d_patch, rhov_d_patch},
            ndust * cell_count,
            [dt](
                u32 id_a,
                const Tscal *__restrict acc_dt_rho_d_patch,
                const Tvec *__restrict acc_dt_rhov_d_patch,
                const Tscal *__restrict rho_d,
                const Tvec *__restrict rhov_d,
                Tscal *__restrict acc_rho_d,
                Tvec *__restrict acc_rhov_d) {
                acc_rho_d[id_a]  = rho_d[id_a] + dt * acc_dt_rho_d_patch[id_a];
                acc_rhov_d[id_a] = rhov_d[id_a] + dt * acc_dt_rhov_d_patch[id_a];
            });
    });

    storage.rho_next_no_drag.set(std::move(cfield_rho_next_bf_drag));
    storage.rhov_next_no_drag.set(std::move(cfield_rhov_next_bf_drag));
    storage.rhoe_next_no_drag.set(std::move(cfield_rhoe_next_bf_drag));
    storage.rho_d_next_no_drag.set(std::move(cfield_rho_d_next_bf_drag));
    storage.rhov_d_next_no_drag.set(std::move(cfield_rhov_d_next_bf_drag));
}

template<class Tvec, class TgridVec>
void shammodels::basegodunov::modules::DragIntegrator<Tvec, TgridVec>::enable_irk1_drag_integrator(
    Tscal dt) {
    StackEntry stack_lock{};

    using namespace shamrock::patch;
    using namespace shamrock;
    using namespace shammath;

    shamrock::ComputeField<Tscal> &cfield_rho_new   = storage.rho_next_no_drag.get();
    shamrock::ComputeField<Tvec> &cfield_rhov_new   = storage.rhov_next_no_drag.get();
    shamrock::ComputeField<Tscal> &cfield_rhoe_new  = storage.rhoe_next_no_drag.get();
    shamrock::ComputeField<Tscal> &cfield_rho_d_new = storage.rho_d_next_no_drag.get();
    shamrock::ComputeField<Tvec> &cfield_rhov_d_new = storage.rhov_d_next_no_drag.get();

    // load layout info
    PatchDataLayerLayout &pdl = scheduler().pdl_old();

    const u32 icell_min = pdl.get_field_idx<TgridVec>("cell_min");
    const u32 icell_max = pdl.get_field_idx<TgridVec>("cell_max");
    const u32 irho      = pdl.get_field_idx<Tscal>("rho");
    const u32 irhoetot  = pdl.get_field_idx<Tscal>("rhoetot");
    const u32 irhovel   = pdl.get_field_idx<Tvec>("rhovel");
    const u32 irho_d    = pdl.get_field_idx<Tscal>("rho_dust");
    const u32 irhovel_d = pdl.get_field_idx<Tvec>("rhovel_dust");

    const u32 ndust = solver_config.dust_config.ndust;
    // alphas are dust collision rates
    auto alphas_vector = solver_config.drag_config.alphas;
    std::vector<Tscal> inv_dt_alphas(ndust);
    bool enable_frictional_heating = solver_config.drag_config.enable_frictional_heating;
    u32 friction_control           = (enable_frictional_heating == false) ? 1 : 0;

    scheduler().for_each_patchdata_nonempty([&, dt, ndust, friction_control](
                                                const shamrock::patch::Patch p,
                                                shamrock::patch::PatchDataLayer &pdat) {
        shamlog_debug_ln("[AMR enable drag ]", "irk1 drag patch", p.id_patch);

        sham::DeviceQueue &q = shamsys::instance::get_compute_scheduler().get_queue();
        u32 id               = p.id_patch;
        u32 cell_count       = pdat.get_obj_cnt() * AMRBlock::block_size;

        sham::DeviceBuffer<Tscal> &rho_new_patch   = cfield_rho_new.get_buf_check(id);
        sham::DeviceBuffer<Tvec> &rhov_new_patch   = cfield_rhov_new.get_buf_check(id);
        sham::DeviceBuffer<Tscal> &rhoe_new_patch  = cfield_rhoe_new.get_buf_check(id);
        sham::DeviceBuffer<Tscal> &rho_d_new_patch = cfield_rho_d_new.get_buf_check(id);
        sham::DeviceBuffer<Tvec> &rhov_d_new_patch = cfield_rhov_d_new.get_buf_check(id);

        sham::DeviceBuffer<Tscal> &rho_old   = pdat.get_field_buf_ref<Tscal>(irho);
        sham::DeviceBuffer<Tvec> &rhov_old   = pdat.get_field_buf_ref<Tvec>(irhovel);
        sham::DeviceBuffer<Tscal> &rhoe_old  = pdat.get_field_buf_ref<Tscal>(irhoetot);
        sham::DeviceBuffer<Tscal> &rho_d_old = pdat.get_field_buf_ref<Tscal>(irho_d);
        sham::DeviceBuffer<Tvec> &rhov_d_old = pdat.get_field_buf_ref<Tvec>(irhovel_d);

        sham::DeviceBuffer<Tscal> alphas_buf(ndust, shamsys::instance::get_compute_scheduler_ptr());

        alphas_buf.copy_from_stdvec(alphas_vector);

        sham::EventList depend_list;
        auto acc_rho_new_patch    = rho_new_patch.get_read_access(depend_list);
        auto acc_rhov_new_patch   = rhov_new_patch.get_read_access(depend_list);
        auto acc_rhoe_new_patch   = rhoe_new_patch.get_read_access(depend_list);
        auto acc_rho_d_new_patch  = rho_d_new_patch.get_read_access(depend_list);
        auto acc_rhov_d_new_patch = rhov_d_new_patch.get_read_access(depend_list);

        auto acc_rho_old    = rho_old.get_write_access(depend_list);
        auto acc_rhov_old   = rhov_old.get_write_access(depend_list);
        auto acc_rhoe_old   = rhoe_old.get_write_access(depend_list);
        auto acc_rho_d_old  = rho_d_old.get_write_access(depend_list);
        auto acc_rhov_d_old = rhov_d_old.get_write_access(depend_list);

        auto acc_alphas = alphas_buf.get_read_access(depend_list);

        auto e = q.submit(depend_list, [&, dt, ndust, friction_control](sycl::handler &cgh) {
            shambase::parallel_for(cgh, cell_count, "add_drag [irk1]", [=](u32 id_a) {
                Tvec tmp_mom_1 = acc_rhov_new_patch[id_a];
                Tscal tmp_rho  = acc_rho_old[id_a];

                for (u32 i = 0; i < ndust; i++) {
                    const Tscal inv_dt_alphas = 1.0 / (1.0 + acc_alphas[i] * dt);
                    const Tscal dt_alphas     = dt * acc_alphas[i];

                    tmp_mom_1
                        = tmp_mom_1
                          + dt_alphas * inv_dt_alphas * acc_rhov_d_new_patch[id_a * ndust + i];
                    tmp_rho = tmp_rho + dt_alphas * inv_dt_alphas * acc_rho_d_old[id_a * ndust + i];
                }

                Tscal tmp_inv_rho = 1.0 / tmp_rho;
                Tvec tmp_vel      = tmp_inv_rho * tmp_mom_1;
                Tscal Eg          = 0.0;

                Tscal inv_rho_g = 1.0 / acc_rho_new_patch[id_a];
                Tvec vg_bf      = inv_rho_g * acc_rhov_new_patch[id_a];
                Tvec vg_af      = inv_rho_g * acc_rho_old[id_a] * tmp_vel;
                ;
                Tscal work_drag
                    = 0.5
                      * ((acc_rho_old[id_a] * tmp_vel[0] - acc_rhov_new_patch[id_a][0])
                             * (vg_bf[0] + vg_af[0])
                         + (acc_rho_old[id_a] * tmp_vel[1] - acc_rhov_new_patch[id_a][1])
                               * (vg_bf[1] + vg_af[1])
                         + (acc_rho_old[id_a] * tmp_vel[2] - acc_rhov_new_patch[id_a][2])
                               * (vg_bf[2] + vg_af[2]));
                Tscal dissipation = 0.0;
                for (u32 i = 0; i < ndust; i++) {
                    const Tscal inv_dt_alphas = 1.0 / (1.0 + acc_alphas[i] * dt);
                    const Tscal dt_alphas     = dt * acc_alphas[i];
                    Tscal inv_rho_d           = 1.0 / acc_rho_d_new_patch[id_a * ndust + i];
                    Tvec vd_bf                = inv_rho_d * acc_rhov_d_new_patch[id_a * ndust + i];
                    Tvec vd_af = inv_rho_d * inv_dt_alphas
                                 * (acc_rhov_d_new_patch[id_a * ndust + i]
                                    + dt_alphas * acc_rho_d_old[id_a * ndust + i] * tmp_vel);
                    dissipation += 0.5 * dt_alphas * inv_dt_alphas
                                   * ((acc_rho_d_old[id_a * ndust + i] * tmp_vel[0]
                                       - acc_rhov_d_new_patch[id_a * ndust + i][0])
                                          * (vd_af[0] + vd_bf[0])
                                      + (acc_rho_d_old[id_a * ndust + i] * tmp_vel[1]
                                         - acc_rhov_d_new_patch[id_a * ndust + i][1])
                                            * (vd_af[1] + vd_bf[1])
                                      + (acc_rho_d_old[id_a * ndust + i] * tmp_vel[2]
                                         - acc_rhov_d_new_patch[id_a * ndust + i][2])
                                            * (vd_af[2] + vd_bf[2]));
                }

                Eg += acc_rhoe_new_patch[id_a] + (1 - friction_control) * work_drag
                      - friction_control * dissipation;
                acc_rhov_old[id_a] = tmp_vel * acc_rho_old[id_a];
                acc_rhoe_old[id_a] = Eg;
                acc_rho_old[id_a]  = acc_rho_new_patch[id_a];
                for (u32 i = 0; i < ndust; i++) {
                    const Tscal inv_dt_alphas = 1.0 / (1.0 + acc_alphas[i] * dt);
                    const Tscal dt_alphas     = dt * acc_alphas[i];
                    acc_rhov_d_old[id_a * ndust + i]
                        = inv_dt_alphas
                          * (acc_rhov_d_new_patch[id_a * ndust + i]
                             + dt_alphas * acc_rho_d_old[id_a * ndust + i] * tmp_vel);
                    acc_rho_d_old[id_a * ndust + i] = acc_rho_d_new_patch[id_a * ndust + i];
                }
            });
        });

        rho_new_patch.complete_event_state(e);
        rhov_new_patch.complete_event_state(e);
        rhoe_new_patch.complete_event_state(e);
        rho_d_new_patch.complete_event_state(e);
        rhov_d_new_patch.complete_event_state(e);

        rho_old.complete_event_state(e);
        rhov_old.complete_event_state(e);
        rhoe_old.complete_event_state(e);
        rho_d_old.complete_event_state(e);
        rhov_d_old.complete_event_state(e);

        alphas_buf.complete_event_state(e);
    });
}

template<class Tvec, class TgridVec>
void shammodels::basegodunov::modules::DragIntegrator<Tvec, TgridVec>::enable_expo_drag_integrator(
    Tscal dt) {
    StackEntry stack_lock{};

    using namespace shamrock::solvergraph;

    const u32 ndust = solver_config.dust_config.ndust;

    // load layout info
    shamrock::patch::PatchDataLayerLayout &pdl = scheduler().pdl_old();

    auto dt_edge   = std::make_shared<ScalarEdge<Tscal>>("dt", "\\Delta t");
    dt_edge->value = dt;

    std::vector<std::shared_ptr<INode>> sequence;

    // refs to the fields of the patch data without ghosts (the ones that are integrated)
    auto attach_source_field
        = [&]<class T>(std::shared_ptr<FieldRefs<T>> &refs, const std::string &field_name) {
              GetFieldRefFromLayer<T> attach{pdl, field_name};
              attach.set_edges(storage.source_patches, refs);
              sequence.push_back(std::make_shared<decltype(attach)>(std::move(attach)));
          };

    // refs to the state at t^{n+1} without drag
    auto refs_from_compute_field =
        [&]<class T>(std::shared_ptr<FieldRefs<T>> &refs, shamrock::ComputeField<T> &field) {
            refs->set_refs(field.field_data.template map<std::reference_wrapper<PatchDataField<T>>>(
                [](u64 id_patch, PatchDataField<T> &pdf) {
                    return std::ref(pdf);
                }));
        };

    auto refs_rho      = std::make_shared<FieldRefs<Tscal>>("rho_source", "\\rho");
    auto refs_rhov     = std::make_shared<FieldRefs<Tvec>>("rhovel_source", "(\\rho \\mathbf{v})");
    auto refs_rhoe     = std::make_shared<FieldRefs<Tscal>>("rhoetot_source", "(\\rho E)");
    auto refs_rho_dust = std::make_shared<FieldRefs<Tscal>>("rho_dust_source", "\\rho_{\\rm dust}");
    auto refs_rhov_dust = std::make_shared<FieldRefs<Tvec>>(
        "rhovel_dust_source", "(\\rho_{\\rm dust} \\mathbf{v}_{\\rm dust})");

    attach_source_field(refs_rho, "rho");
    attach_source_field(refs_rhov, "rhovel");
    attach_source_field(refs_rhoe, "rhoetot");
    attach_source_field(refs_rho_dust, "rho_dust");
    attach_source_field(refs_rhov_dust, "rhovel_dust");

    auto refs_rho_next
        = std::make_shared<FieldRefs<Tscal>>("rho_next_no_drag", "\\rho^{n+1}_{\\rm no\\,drag}");
    auto refs_rhov_next = std::make_shared<FieldRefs<Tvec>>(
        "rhov_next_no_drag", "(\\rho \\mathbf{v})^{n+1}_{\\rm no\\,drag}");
    auto refs_rhoe_next = std::make_shared<FieldRefs<Tscal>>(
        "rhoe_next_no_drag", "(\\rho E)^{n+1}_{\\rm no\\,drag}");
    auto refs_rho_d_next = std::make_shared<FieldRefs<Tscal>>(
        "rho_d_next_no_drag", "\\rho^{n+1}_{\\rm dust,\\,no\\,drag}");
    auto refs_rhov_d_next = std::make_shared<FieldRefs<Tvec>>(
        "rhov_d_next_no_drag",
        "(\\rho_{\\rm dust} \\mathbf{v}_{\\rm dust})^{n+1}_{\\rm no\\,drag}");

    refs_from_compute_field(refs_rho_next, storage.rho_next_no_drag.get());
    refs_from_compute_field(refs_rhov_next, storage.rhov_next_no_drag.get());
    refs_from_compute_field(refs_rhoe_next, storage.rhoe_next_no_drag.get());
    refs_from_compute_field(refs_rho_d_next, storage.rho_d_next_no_drag.get());
    refs_from_compute_field(refs_rhov_d_next, storage.rhov_d_next_no_drag.get());

    {
        // alphas are dust collision rates
        const auto &alphas = solver_config.drag_config.alphas;

        ApplyDragExpo<Tvec> node{
            AMRBlock::block_size,
            ndust,
            std::vector<Tscal>(alphas.begin(), alphas.end()),
            solver_config.drag_config.enable_frictional_heating};
        node.set_edges(
            dt_edge,
            storage.block_counts,
            refs_rho_next,
            refs_rhov_next,
            refs_rhoe_next,
            refs_rho_d_next,
            refs_rhov_d_next,
            refs_rho,
            refs_rhov,
            refs_rhoe,
            refs_rho_dust,
            refs_rhov_dust);
        sequence.push_back(std::make_shared<decltype(node)>(std::move(node)));
    }

    OperationSequence seq("Drag expo", std::move(sequence));
    seq.evaluate();
}

template class shammodels::basegodunov::modules::DragIntegrator<f64_3, i64_3>;
