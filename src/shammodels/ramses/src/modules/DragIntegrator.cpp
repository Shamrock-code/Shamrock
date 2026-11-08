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
#include "shammodels/ramses/modules/DragIntegrator.hpp"
#include "shammodels/ramses/modules/ForwardEulerDust.hpp"
#include "shammodels/ramses/modules/ForwardEulerGas.hpp"
#include "shamrock/scheduler/SchedulerUtility.hpp"
#include "shamrock/solvergraph/CopyPatchDataField.hpp"
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

    using namespace shamrock::solvergraph;

    const u32 ndust = solver_config.dust_config.ndust;

    // load layout info
    PatchDataLayerLayout &pdl = scheduler().pdl_old();

    storage.rho_next_no_drag = std::make_shared<Field<Tscal>>(
        AMRBlock::block_size, "rho_next_bf_drag", "\\rho^{n+1}_{\\rm no\\,drag}");
    storage.rhov_next_no_drag = std::make_shared<Field<Tvec>>(
        AMRBlock::block_size, "rhov_next_bf_drag", "(\\rho \\mathbf{v})^{n+1}_{\\rm no\\,drag}");
    storage.rhoe_next_no_drag = std::make_shared<Field<Tscal>>(
        AMRBlock::block_size, "rhoe_next_bf_drag", "(\\rho E)^{n+1}_{\\rm no\\,drag}");
    storage.rho_d_next_no_drag = std::make_shared<Field<Tscal>>(
        ndust * AMRBlock::block_size, "rho_d_next_bf_drag", "\\rho^{n+1}_{\\rm dust,\\,no\\,drag}");
    storage.rhov_d_next_no_drag = std::make_shared<Field<Tvec>>(
        ndust * AMRBlock::block_size,
        "rhov_d_next_bf_drag",
        "(\\rho_{\\rm dust} \\mathbf{v}_{\\rm dust})^{n+1}_{\\rm no\\,drag}");

    auto dt_edge   = std::make_shared<ScalarEdge<Tscal>>("dt", "\\Delta t");
    dt_edge->value = dt;

    std::vector<std::shared_ptr<INode>> sequence;

    // U^{n+1} starts as a copy of the state at t^n (patch data without ghosts), then gets the
    // forward Euler update in place
    auto copy_source_field = [&]<class T>(
                                 const std::string &field_name,
                                 const std::string &label,
                                 const std::string &tex,
                                 const std::shared_ptr<Field<T>> &target) {
        auto refs = std::make_shared<FieldRefs<T>>(label, tex);

        GetFieldRefFromLayer<T> attach{pdl, field_name};
        attach.set_edges(storage.source_patches, refs);
        sequence.push_back(std::make_shared<decltype(attach)>(std::move(attach)));

        CopyPatchDataField<T> copy{};
        copy.set_edges(refs, target);
        sequence.push_back(std::make_shared<decltype(copy)>(std::move(copy)));
    };

    copy_source_field("rho", "rho_source", "\\rho", storage.rho_next_no_drag);
    copy_source_field("rhovel", "rhovel_source", "(\\rho \\mathbf{v})", storage.rhov_next_no_drag);
    copy_source_field("rhoetot", "rhoetot_source", "(\\rho E)", storage.rhoe_next_no_drag);
    copy_source_field(
        "rho_dust", "rho_dust_source", "\\rho_{\\rm dust}", storage.rho_d_next_no_drag);
    copy_source_field(
        "rhovel_dust",
        "rhovel_dust_source",
        "(\\rho_{\\rm dust} \\mathbf{v}_{\\rm dust})",
        storage.rhov_d_next_no_drag);

    {
        ForwardEulerGas<Tvec> node{AMRBlock::block_size};
        node.set_edges(
            dt_edge,
            storage.block_counts,
            storage.dtrho,
            storage.dtrhov,
            storage.dtrhoe,
            storage.rho_next_no_drag,
            storage.rhov_next_no_drag,
            storage.rhoe_next_no_drag);
        sequence.push_back(std::make_shared<decltype(node)>(std::move(node)));
    }

    {
        ForwardEulerDust<Tvec> node{AMRBlock::block_size, ndust};
        node.set_edges(
            dt_edge,
            storage.block_counts,
            storage.dtrho_dust,
            storage.dtrhov_dust,
            storage.rho_d_next_no_drag,
            storage.rhov_d_next_no_drag);
        sequence.push_back(std::make_shared<decltype(node)>(std::move(node)));
    }

    OperationSequence seq("Forward Euler before drag", std::move(sequence));
    seq.evaluate();
}

template<class Tvec, class TgridVec>
void shammodels::basegodunov::modules::DragIntegrator<Tvec, TgridVec>::enable_irk1_drag_integrator(
    Tscal dt) {
    StackEntry stack_lock{};

    using namespace shamrock::patch;
    using namespace shamrock;
    using namespace shammath;

    shamrock::ComputeField<Tscal> &cfield_rho_new
        = shambase::get_check_ref(storage.rho_next_no_drag).get_native();
    shamrock::ComputeField<Tvec> &cfield_rhov_new
        = shambase::get_check_ref(storage.rhov_next_no_drag).get_native();
    shamrock::ComputeField<Tscal> &cfield_rhoe_new
        = shambase::get_check_ref(storage.rhoe_next_no_drag).get_native();
    shamrock::ComputeField<Tscal> &cfield_rho_d_new
        = shambase::get_check_ref(storage.rho_d_next_no_drag).get_native();
    shamrock::ComputeField<Tvec> &cfield_rhov_d_new
        = shambase::get_check_ref(storage.rhov_d_next_no_drag).get_native();

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

    using namespace shamrock::patch;
    using namespace shamrock;
    using namespace shammath;

    shamrock::ComputeField<Tscal> &cfield_rho_new
        = shambase::get_check_ref(storage.rho_next_no_drag).get_native();
    shamrock::ComputeField<Tvec> &cfield_rhov_new
        = shambase::get_check_ref(storage.rhov_next_no_drag).get_native();
    shamrock::ComputeField<Tscal> &cfield_rhoe_new
        = shambase::get_check_ref(storage.rhoe_next_no_drag).get_native();
    shamrock::ComputeField<Tscal> &cfield_rho_d_new
        = shambase::get_check_ref(storage.rho_d_next_no_drag).get_native();
    shamrock::ComputeField<Tvec> &cfield_rhov_d_new
        = shambase::get_check_ref(storage.rhov_d_next_no_drag).get_native();

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
        shamlog_debug_ln("[Ramses]", "expo drag on patch", p.id_patch);

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

        size_t mat_size         = ndust + 1;
        size_t mat_size_squared = mat_size * mat_size;
        size_t group_size
            = (q.get_device_prop().local_mem_size) / (5 * mat_size_squared * sizeof(Tscal));
        size_t loc_acc_size = mat_size_squared * group_size;

        size_t loc_mem_size = 5 * sizeof(Tscal) * loc_acc_size;

        if (group_size < 8) {
            sham::DeviceBuffer<Tscal> scratch_expo(
                5 * mat_size_squared * cell_count, shamsys::instance::get_compute_scheduler_ptr());
            Tscal *exp_scratch_ptr_base = scratch_expo.get_write_access(depend_list);
            auto e = q.submit(depend_list, [&, dt, ndust, friction_control](sycl::handler &cgh) {
                shambase::parallel_for(
                    cgh, cell_count, "add_drag [expo-global-mem]", [=](u32 id_a) {
                        // sparse jacobian matrix
                        auto get_jacobian =
                            [=](u32 id,
                                std::mdspan<
                                    Tscal,
                                    std::extents<size_t, std::dynamic_extent, std::dynamic_extent>>
                                    &jacobian) {
                                mat_set_nul<Tscal>(jacobian);
                                // fill first row
                                for (auto j = 1; j < jacobian.extent(1); j++)
                                    jacobian(0, j) = acc_alphas[j - 1];
                                // fil first column
                                for (auto i = 1; i < jacobian.extent(0); i++) {
                                    jacobian(i, 0) = acc_alphas[i - 1]
                                                     * (acc_rho_d_new_patch[id * ndust + (i - 1)]
                                                        / acc_rho_new_patch[id]);
                                    jacobian(0, 0) -= jacobian(i, 0);
                                }
                                // fill diagonal from (i,j)=(1,1)
                                for (auto i = 1; i < jacobian.extent(0); i++)
                                    jacobian(i, i) = -acc_alphas[i - 1];
                                // the rest of the buffer is set to zero
                            };
                        Tscal mu = 0;
                        for (auto i = 0; i < ndust; i++) {
                            mu += (1
                                   + (acc_rho_d_new_patch[id_a * ndust + i]
                                      / acc_rho_new_patch[id_a]))
                                  * acc_alphas[i];
                        }
                        mu *= (-dt / (ndust + 1));

                        // get ptr to datas
                        Tscal *ptr_A  = exp_scratch_ptr_base + (id_a * 5 * mat_size_squared);
                        Tscal *ptr_B  = exp_scratch_ptr_base + (id_a * 5 * mat_size_squared)
                                        + mat_size_squared;
                        Tscal *ptr_F  = exp_scratch_ptr_base + (id_a * 5 * mat_size_squared)
                                        + 2 * mat_size_squared;
                        Tscal *ptr_I  = exp_scratch_ptr_base + (id_a * 5 * mat_size_squared)
                                        + 3 * mat_size_squared;
                        Tscal *ptr_Id = exp_scratch_ptr_base + (id_a * 5 * mat_size_squared)
                                        + 4 * mat_size_squared;

                        // create mdspan(s)
                        std::mdspan<
                            Tscal,
                            std::extents<size_t, std::dynamic_extent, std::dynamic_extent>>
                            mdspan_A(ptr_A, mat_size, mat_size);
                        std::mdspan<
                            Tscal,
                            std::extents<size_t, std::dynamic_extent, std::dynamic_extent>>
                            mdspan_B(ptr_B, mat_size, mat_size);
                        std::mdspan<
                            Tscal,
                            std::extents<size_t, std::dynamic_extent, std::dynamic_extent>>
                            mdspan_F(ptr_F, mat_size, mat_size);
                        std::mdspan<
                            Tscal,
                            std::extents<size_t, std::dynamic_extent, std::dynamic_extent>>
                            mdspan_I(ptr_I, mat_size, mat_size);
                        std::mdspan<
                            Tscal,
                            std::extents<size_t, std::dynamic_extent, std::dynamic_extent>>
                            mdspan_Id(ptr_Id, mat_size, mat_size);

                        get_jacobian(id_a, mdspan_A);

                        // pre-processing step
                        shammath::mat_set_identity<Tscal>(mdspan_Id);
                        shammath::mat_axpy_beta<Tscal, Tscal>(-mu, mdspan_Id, dt, mdspan_A);

                        // compute matrix exponential
                        const i32 K_exp = 9;
                        shammath::mat_exp<Tscal, Tscal>(
                            K_exp, mdspan_A, mdspan_F, mdspan_B, mdspan_I, mdspan_Id, ndust + 1);

                        // post-processing step
                        shammath::mat_mul_scalar<Tscal>(mdspan_A, sycl::exp(mu));

                        // use the matrix exponential to for to updates momemtum
                        Tvec r = {0., 0., 0.}, dd = {0., 0., 0.};
                        r += mdspan_A(0, 0) * acc_rhov_new_patch[id_a];

                        for (auto j = 1; j < ndust + 1; j++) {
                            r += mdspan_A(0, j) * acc_rhov_d_new_patch[id_a * ndust + (j - 1)];
                        }

                        dd = r - acc_rhov_new_patch[id_a];

                        Tscal dissipation = 0, drag_work = 0;

                        // compute work of drag terms
                        Tscal inv_rho = 1.0 / (acc_rho_new_patch[id_a]);

                        Tvec v_bf = inv_rho * acc_rhov_new_patch[id_a];
                        Tvec v_af = inv_rho * r;

                        drag_work = 0.5
                                    * (dd[0] * (v_bf[0] + v_af[0]) + dd[1] * (v_bf[1] + v_af[1])
                                       + dd[2] * (v_bf[2] + v_af[2]));

                        // save gas momentum back
                        acc_rhov_old[id_a] = r;
                        acc_rho_old[id_a]  = acc_rho_new_patch[id_a];

                        for (auto d_id = 1; d_id <= ndust; d_id++) {
                            r *= 0;
                            r += mdspan_A(d_id, 0) * acc_rhov_new_patch[id_a];

                            for (auto j = 1; j <= ndust; j++) {

                                r += mdspan_A(d_id, j)
                                     * acc_rhov_d_new_patch[id_a * ndust + (j - 1)];
                            }

                            dd = r - acc_rhov_d_new_patch[id_a * ndust + (d_id - 1)];

                            inv_rho = 1.0 / (acc_rho_d_new_patch[id_a * ndust + (d_id - 1)]);

                            v_bf = inv_rho * acc_rhov_d_new_patch[id_a * ndust + (d_id - 1)];

                            v_af = inv_rho * r;

                            // compute dissipaation by id-th dust
                            dissipation
                                += 0.5
                                   * (dd[0] * (v_bf[0] + v_af[0]) + dd[1] * (v_bf[1] + v_af[1])
                                      + dd[2] * (v_bf[2] + v_af[2]));

                            // save dust momentum back
                            acc_rhov_d_old[id_a * ndust + (d_id - 1)] = r;
                            acc_rho_d_old[id_a * ndust + (d_id - 1)]
                                = acc_rho_d_new_patch[id_a * ndust + (d_id - 1)];
                        }

                        // updates energy
                        acc_rhoe_old[id_a] = acc_rhoe_new_patch[id_a]
                                             + (1 - friction_control) * drag_work
                                             - friction_control * dissipation;
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
            scratch_expo.complete_event_state(e);

        } else {

            if (loc_mem_size > q.get_device_prop().local_mem_size) {
                shambase::throw_with_loc<std::runtime_error>(sham::format(
                    "not enough local memory for expo drag integrator:\n"
                    "loc_mem_size: {} > max_local_mem: {}\n"
                    "loc_acc_size: {}\n"
                    "group_size: {}\n"
                    "ndust: {}\n",
                    loc_mem_size,
                    q.get_device_prop().local_mem_size,
                    loc_acc_size,
                    group_size,
                    ndust));
            }

            auto e = q.submit(depend_list, [&, dt, ndust, friction_control](sycl::handler &cgh) {
                // local/shared memory alloc for each work-item
                sycl::local_accessor<Tscal> local_A(loc_acc_size, cgh);
                sycl::local_accessor<Tscal> local_B(loc_acc_size, cgh);
                sycl::local_accessor<Tscal> local_F(loc_acc_size, cgh);
                sycl::local_accessor<Tscal> local_I(loc_acc_size, cgh);
                sycl::local_accessor<Tscal> local_Id(loc_acc_size, cgh);

                logger::debug_sycl_ln(
                    "SYCL", sham::format("parallel_for add_drag [expo-shared-mem]"));
                cgh.parallel_for(
                    shambase::make_range(cell_count, group_size), [=](sycl::nd_item<1> id) {
                        u32 loc_id = id.get_local_id();
                        u32 id_a   = id.get_global_id();
                        if (id_a >= cell_count)
                            return;

                        // sparse jacobian matrix
                        auto get_jacobian =
                            [=](u32 id,
                                std::mdspan<
                                    Tscal,
                                    std::extents<size_t, std::dynamic_extent, std::dynamic_extent>>
                                    &jacobian) {
                                mat_set_nul<Tscal>(jacobian);
                                // fill first row
                                for (auto j = 1; j < jacobian.extent(1); j++)
                                    jacobian(0, j) = acc_alphas[j - 1];
                                // fil first column
                                for (auto i = 1; i < jacobian.extent(0); i++) {
                                    jacobian(i, 0) = acc_alphas[i - 1]
                                                     * (acc_rho_d_new_patch[id * ndust + (i - 1)]
                                                        / acc_rho_new_patch[id]);
                                    jacobian(0, 0) -= jacobian(i, 0);
                                }
                                // fill diagonal from (i,j)=(1,1)
                                for (auto i = 1; i < jacobian.extent(0); i++)
                                    jacobian(i, i) = -acc_alphas[i - 1];
                                // the rest of the buffer is set to zero
                            };

                        Tscal mu = 0;
                        for (auto i = 0; i < ndust; i++) {
                            mu += (1
                                   + (acc_rho_d_new_patch[id_a * ndust + i]
                                      / acc_rho_new_patch[id_a]))
                                  * acc_alphas[i];
                        }
                        mu *= (-dt / (ndust + 1));

                        // get ptr to datas
                        Tscal *ptr_loc_A  = &(local_A[0]) + mat_size_squared * loc_id;
                        Tscal *ptr_loc_B  = &(local_B[0]) + mat_size_squared * loc_id;
                        Tscal *ptr_loc_F  = &(local_F[0]) + mat_size_squared * loc_id;
                        Tscal *ptr_loc_I  = &(local_I[0]) + mat_size_squared * loc_id;
                        Tscal *ptr_loc_Id = &(local_Id[0]) + mat_size_squared * loc_id;

                        // create mdspan(s)
                        std::mdspan<
                            Tscal,
                            std::extents<size_t, std::dynamic_extent, std::dynamic_extent>>
                            mdspan_A(ptr_loc_A, mat_size, mat_size);
                        std::mdspan<
                            Tscal,
                            std::extents<size_t, std::dynamic_extent, std::dynamic_extent>>
                            mdspan_B(ptr_loc_B, mat_size, mat_size);
                        std::mdspan<
                            Tscal,
                            std::extents<size_t, std::dynamic_extent, std::dynamic_extent>>
                            mdspan_F(ptr_loc_F, mat_size, mat_size);
                        std::mdspan<
                            Tscal,
                            std::extents<size_t, std::dynamic_extent, std::dynamic_extent>>
                            mdspan_I(ptr_loc_I, mat_size, mat_size);
                        std::mdspan<
                            Tscal,
                            std::extents<size_t, std::dynamic_extent, std::dynamic_extent>>
                            mdspan_Id(ptr_loc_Id, mat_size, mat_size);

                        // get local Jacobian matrix

                        get_jacobian(id_a, mdspan_A);

                        // pre-processing step
                        shammath::mat_set_identity<Tscal>(mdspan_Id);
                        shammath::mat_axpy_beta<Tscal, Tscal>(-mu, mdspan_Id, dt, mdspan_A);

                        // compute matrix exponential
                        const i32 K_exp = 9;
                        shammath::mat_exp<Tscal, Tscal>(
                            K_exp, mdspan_A, mdspan_F, mdspan_B, mdspan_I, mdspan_Id, ndust + 1);

                        // post-processing step
                        shammath::mat_mul_scalar<Tscal>(mdspan_A, sycl::exp(mu));

                        // use the matrix exponential to for to updates momemtum
                        Tvec r = {0., 0., 0.}, dd = {0., 0., 0.};
                        r += mdspan_A(0, 0) * acc_rhov_new_patch[id_a];

                        for (auto j = 1; j < ndust + 1; j++) {
                            r += mdspan_A(0, j) * acc_rhov_d_new_patch[id_a * ndust + (j - 1)];
                        }

                        dd = r - acc_rhov_new_patch[id_a];

                        Tscal dissipation = 0, drag_work = 0;

                        // compute work of drag terms
                        Tscal inv_rho = 1.0 / (acc_rho_new_patch[id_a]);

                        Tvec v_bf = inv_rho * acc_rhov_new_patch[id_a];
                        Tvec v_af = inv_rho * r;

                        drag_work = 0.5
                                    * (dd[0] * (v_bf[0] + v_af[0]) + dd[1] * (v_bf[1] + v_af[1])
                                       + dd[2] * (v_bf[2] + v_af[2]));

                        // save gas momentum back
                        acc_rhov_old[id_a] = r;
                        acc_rho_old[id_a]  = acc_rho_new_patch[id_a];

                        for (auto d_id = 1; d_id <= ndust; d_id++) {
                            r *= 0;
                            r += mdspan_A(d_id, 0) * acc_rhov_new_patch[id_a];

                            for (auto j = 1; j <= ndust; j++) {

                                r += mdspan_A(d_id, j)
                                     * acc_rhov_d_new_patch[id_a * ndust + (j - 1)];
                            }

                            dd = r - acc_rhov_d_new_patch[id_a * ndust + (d_id - 1)];

                            inv_rho = 1.0 / (acc_rho_d_new_patch[id_a * ndust + (d_id - 1)]);

                            v_bf = inv_rho * acc_rhov_d_new_patch[id_a * ndust + (d_id - 1)];

                            v_af = inv_rho * r;

                            // compute dissipaation by id-th dust
                            dissipation
                                += 0.5
                                   * (dd[0] * (v_bf[0] + v_af[0]) + dd[1] * (v_bf[1] + v_af[1])
                                      + dd[2] * (v_bf[2] + v_af[2]));

                            // save dust momentum back
                            acc_rhov_d_old[id_a * ndust + (d_id - 1)] = r;
                            acc_rho_d_old[id_a * ndust + (d_id - 1)]
                                = acc_rho_d_new_patch[id_a * ndust + (d_id - 1)];
                        }

                        // updates energy
                        acc_rhoe_old[id_a] = acc_rhoe_new_patch[id_a]
                                             + (1 - friction_control) * drag_work
                                             - friction_control * dissipation;
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
        }
    });
}
template class shammodels::basegodunov::modules::DragIntegrator<f64_3, i64_3>;
