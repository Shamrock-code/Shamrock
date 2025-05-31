// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright (c) 2021-2024 Timothée David--Cléris <tim.shamrock@proton.me>
// SPDX-License-Identifier: CeCILL Free Software License Agreement v2.1
// Shamrock is licensed under the CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

/**
 * @file FaceInterpolate.cpp
 * @author Timothée David--Cléris (timothee.david--cleris@ens-lyon.fr)
 * @brief
 *
 */

#include "shambase/memory.hpp"
#include "shammodels/common/amr/NeighGraph.hpp"
#include "shammodels/common/amr/NeighGraphLinkField.hpp"
#include "shammodels/ramses/modules/FaceInterpolate.hpp"
#include "shamrock/patch/PatchDataFieldSpan.hpp"
#include "shamsys/NodeInstance.hpp"
#include <array>

namespace {

    template<class Tvec, class TgridVec, class AMRBlock>
    class GetShift {

        public:
        using Tscal     = shambase::VecComponent<Tvec>;
        using Tgridscal = shambase::VecComponent<TgridVec>;

        const Tvec *acc_aabb_block_lower;
        const Tscal *acc_aabb_cell_size;

        GetShift(const Tvec *aabb_block_lower, const Tscal *aabb_cell_size)
            : acc_aabb_block_lower{aabb_block_lower}, acc_aabb_cell_size{aabb_cell_size} {}

        shammath::AABB<Tvec> get_cell_aabb(u32 id) const {

            const u32 cell_global_id = (u32) id;

            const u32 block_id    = cell_global_id / AMRBlock::block_size;
            const u32 cell_loc_id = cell_global_id % AMRBlock::block_size;

            // fetch current block info
            const Tvec cblock_min  = acc_aabb_block_lower[block_id];
            const Tscal delta_cell = acc_aabb_cell_size[block_id];

            std::array<u32, 3> lcoord_arr = AMRBlock::get_coord(cell_loc_id);
            Tvec offset = Tvec{lcoord_arr[0], lcoord_arr[1], lcoord_arr[2]} * delta_cell;

            Tvec aabb_min = cblock_min + offset;
            Tvec aabb_max = aabb_min + delta_cell;

            return {aabb_min, aabb_max};
        }

        std::pair<Tvec, Tvec> get_shifts(u32 id_a, u32 id_b) const {

            shammath::AABB<Tvec> aabb_cell_a = get_cell_aabb(id_a);
            shammath::AABB<Tvec> aabb_cell_b = get_cell_aabb(id_b);

            shammath::AABB<Tvec> face_aabb = aabb_cell_a.get_intersect(aabb_cell_b);

            Tvec face_center = face_aabb.get_center();

            Tvec shift_a = face_center - aabb_cell_a.get_center();
            Tvec shift_b = face_center - aabb_cell_b.get_center();

            return {shift_a, shift_b};
        }
    };

    template<class Tvec, class TgridVec, class AMRBlock>
    class RhoInterpolate {
        using Tscal = shambase::VecComponent<Tvec>;

        public:
        shamrock::PatchDataFieldSpanPointer<Tvec> aabb_block_lower;
        shamrock::PatchDataFieldSpanPointer<Tscal> aabb_cell_size;
        shamrock::PatchDataFieldSpanPointer<Tscal> rho_cell;
        shamrock::PatchDataFieldSpanPointer<Tvec> grad_rho_cell;
        // For time interpolation
        Tscal dt_interp;
        shamrock::PatchDataFieldSpanPointer<Tvec> vel_cell;
        shamrock::PatchDataFieldSpanPointer<Tvec> dx_v_cell;
        shamrock::PatchDataFieldSpanPointer<Tvec> dy_v_cell;
        shamrock::PatchDataFieldSpanPointer<Tvec> dz_v_cell;

        class acc {
            public:
            GetShift<Tvec, TgridVec, AMRBlock> shift_get;

            const Tscal *acc_rho_cell;
            const Tvec *acc_grad_rho_cell;

            // For time interpolation
            const Tvec *acc_vel_cell;
            const Tvec *acc_dx_v_cell;
            const Tvec *acc_dy_v_cell;
            const Tvec *acc_dz_v_cell;

            Tscal dt_interp;

            acc(const Tvec *aabb_block_lower,
                const Tscal *aabb_cell_size,
                const Tscal *rho_cell,
                const Tvec *grad_rho_cell,
                // For time interpolation
                Tscal dt_interp,
                const Tvec *vel_cell,
                const Tvec *dx_v_cell,
                const Tvec *dy_v_cell,
                const Tvec *dz_v_cell)
                : shift_get(aabb_block_lower, aabb_cell_size), acc_rho_cell{rho_cell},
                  acc_grad_rho_cell{grad_rho_cell}, dt_interp(dt_interp), acc_vel_cell{vel_cell},
                  acc_dx_v_cell{dx_v_cell}, acc_dy_v_cell{dy_v_cell}, acc_dz_v_cell{dz_v_cell} {}

            Tscal
            get_dt_rho(Tscal rho, Tvec v, Tvec grad_rho, Tvec dx_v, Tvec dy_v, Tvec dz_v) const {
                return -(sham::dot(v, grad_rho) + rho * (dx_v[0] + dy_v[1] + dz_v[2]));
            }

            std::array<Tscal, 2> get_link_field_val(u32 id_a, u32 id_b) const {

                auto [shift_a, shift_b] = shift_get.get_shifts(id_a, id_b);

                Tscal rho_a     = acc_rho_cell[id_a];
                Tvec grad_rho_a = acc_grad_rho_cell[id_a];
                Tscal rho_b     = acc_rho_cell[id_b];
                Tvec grad_rho_b = acc_grad_rho_cell[id_b];

                Tvec vel_a  = acc_vel_cell[id_a];
                Tvec dx_v_a = acc_dx_v_cell[id_a];
                Tvec dy_v_a = acc_dy_v_cell[id_a];
                Tvec dz_v_a = acc_dz_v_cell[id_a];
                Tvec vel_b  = acc_vel_cell[id_b];
                Tvec dx_v_b = acc_dx_v_cell[id_b];
                Tvec dy_v_b = acc_dy_v_cell[id_b];
                Tvec dz_v_b = acc_dz_v_cell[id_b];

                // Spatial interpolate
                Tscal rho_face_a = rho_a + sycl::dot(grad_rho_a, shift_a);
                Tscal rho_face_b = rho_b + sycl::dot(grad_rho_b, shift_b);

                // Interpolate also to half a timestep
                rho_face_a
                    += get_dt_rho(rho_a, vel_a, grad_rho_a, dx_v_a, dy_v_a, dz_v_a) * dt_interp;
                rho_face_b
                    += get_dt_rho(rho_b, vel_b, grad_rho_b, dx_v_b, dy_v_b, dz_v_b) * dt_interp;

                return {rho_face_a, rho_face_b};
            }
        };

        inline acc get_read_access(sham::EventList &deps) {
            return acc(
                aabb_block_lower.get_read_access(deps),
                aabb_cell_size.get_read_access(deps),
                rho_cell.get_read_access(deps),
                grad_rho_cell.get_read_access(deps),
                // For time interpolation
                dt_interp,
                vel_cell.get_read_access(deps),
                dx_v_cell.get_read_access(deps),
                dy_v_cell.get_read_access(deps),
                dz_v_cell.get_read_access(deps));
        }

        inline void complete_event_state(sycl::event e) {
            aabb_block_lower.complete_event_state(e);
            aabb_cell_size.complete_event_state(e);
            rho_cell.complete_event_state(e);
            grad_rho_cell.complete_event_state(e);
            vel_cell.complete_event_state(e);
            dx_v_cell.complete_event_state(e);
            dy_v_cell.complete_event_state(e);
            dz_v_cell.complete_event_state(e);
        }
    };

    template<class Tvec, class TgridVec, class AMRBlock>
    class VelInterpolate {
        using Tscal = shambase::VecComponent<Tvec>;

        public:
        shamrock::PatchDataFieldSpanPointer<Tvec> aabb_block_lower;
        shamrock::PatchDataFieldSpanPointer<Tscal> aabb_cell_size;

        shamrock::PatchDataFieldSpanPointer<Tvec> vel_cell;
        shamrock::PatchDataFieldSpanPointer<Tvec> dx_v_cell;
        shamrock::PatchDataFieldSpanPointer<Tvec> dy_v_cell;
        shamrock::PatchDataFieldSpanPointer<Tvec> dz_v_cell;
        // For time interpolation
        Tscal dt_interp;
        shamrock::PatchDataFieldSpanPointer<Tscal> rho_cell;
        shamrock::PatchDataFieldSpanPointer<Tvec> grad_P_cell;

        class acc {
            public:
            GetShift<Tvec, TgridVec, AMRBlock> shift_get;

            const Tvec *acc_vel_cell;
            const Tvec *acc_dx_v_cell;
            const Tvec *acc_dy_v_cell;
            const Tvec *acc_dz_v_cell;

            // For time interpolation
            const Tscal *acc_rho_cell;
            const Tvec *acc_grad_P_cell;

            Tscal dt_interp;

            acc(const Tvec *aabb_block_lower,
                const Tscal *aabb_cell_size,
                const Tvec *vel_cell,
                const Tvec *dx_v_cell,
                const Tvec *dy_v_cell,
                const Tvec *dz_v_cell,
                // For time interpolation
                Tscal dt_interp,
                const Tscal *rho_cell,
                const Tvec *grad_P_cell)
                : shift_get(aabb_block_lower, aabb_cell_size), acc_vel_cell{vel_cell},
                  acc_dx_v_cell{dx_v_cell}, acc_dy_v_cell{dy_v_cell}, acc_dz_v_cell{dz_v_cell},
                  dt_interp(dt_interp), acc_rho_cell{rho_cell}, acc_grad_P_cell{grad_P_cell} {}

            Tvec get_dt_v(Tvec v, Tvec dx_v, Tvec dy_v, Tvec dz_v, Tscal rho, Tvec grad_P) const {
                return -(v[0] * dx_v + v[1] * dy_v + v[2] * dz_v + grad_P / rho);
            }

            std::array<Tvec, 2> get_link_field_val(u32 id_a, u32 id_b) const {

                auto [shift_a, shift_b] = shift_get.get_shifts(id_a, id_b);

                Tvec v_a      = acc_vel_cell[id_a];
                Tvec dx_vel_a = acc_dx_v_cell[id_a];
                Tvec dy_vel_a = acc_dy_v_cell[id_a];
                Tvec dz_vel_a = acc_dz_v_cell[id_a];

                Tvec v_b      = acc_vel_cell[id_b];
                Tvec dx_vel_b = acc_dx_v_cell[id_b];
                Tvec dy_vel_b = acc_dy_v_cell[id_b];
                Tvec dz_vel_b = acc_dz_v_cell[id_b];

                Tscal rho_a   = acc_rho_cell[id_a];
                Tvec grad_P_a = acc_grad_P_cell[id_a];
                Tscal rho_b   = acc_rho_cell[id_b];
                Tvec grad_P_b = acc_grad_P_cell[id_b];

                Tvec dx_v_a_dot_shift
                    = shift_a.x() * dx_vel_a + shift_a.y() * dy_vel_a + shift_a.z() * dz_vel_a;
                Tvec dx_v_b_dot_shift
                    = shift_b.x() * dx_vel_b + shift_b.y() * dy_vel_b + shift_b.z() * dz_vel_b;

                Tvec dt_v_a = get_dt_v(v_a, dx_vel_a, dy_vel_a, dz_vel_a, rho_a, grad_P_a);
                Tvec dt_v_b = get_dt_v(v_b, dx_vel_b, dy_vel_b, dz_vel_b, rho_b, grad_P_b);

                Tvec vel_face_a = v_a + dx_v_a_dot_shift + dt_v_a * dt_interp;
                Tvec vel_face_b = v_b + dx_v_b_dot_shift + dt_v_b * dt_interp;

                return {vel_face_a, vel_face_b};
            }
        };

        inline acc get_read_access(sham::EventList &deps) {
            return acc(
                aabb_block_lower.get_read_access(deps),
                aabb_cell_size.get_read_access(deps),
                vel_cell.get_read_access(deps),
                dx_v_cell.get_read_access(deps),
                dy_v_cell.get_read_access(deps),
                dz_v_cell.get_read_access(deps),
                // For time interpolation
                dt_interp,
                rho_cell.get_read_access(deps),
                grad_P_cell.get_read_access(deps));
        }

        inline void complete_event_state(sycl::event e) {
            aabb_block_lower.complete_event_state(e);
            aabb_cell_size.complete_event_state(e);
            vel_cell.complete_event_state(e);
            dx_v_cell.complete_event_state(e);
            dy_v_cell.complete_event_state(e);
            dz_v_cell.complete_event_state(e);
            rho_cell.complete_event_state(e);
            grad_P_cell.complete_event_state(e);
        }
    };

    template<class Tvec, class TgridVec, class AMRBlock>
    class PressInterpolate {
        using Tscal = shambase::VecComponent<Tvec>;

        public:
        shamrock::PatchDataFieldSpanPointer<Tvec> aabb_block_lower;
        shamrock::PatchDataFieldSpanPointer<Tscal> aabb_cell_size;
        shamrock::PatchDataFieldSpanPointer<Tscal> P_cell;
        shamrock::PatchDataFieldSpanPointer<Tvec> grad_P_cell;
        // For time interpolation
        Tscal dt_interp;
        Tscal gamma;
        shamrock::PatchDataFieldSpanPointer<Tvec> vel_cell;
        shamrock::PatchDataFieldSpanPointer<Tvec> dx_v_cell;
        shamrock::PatchDataFieldSpanPointer<Tvec> dy_v_cell;
        shamrock::PatchDataFieldSpanPointer<Tvec> dz_v_cell;

        class acc {
            public:
            GetShift<Tvec, TgridVec, AMRBlock> shift_get;

            const Tscal *acc_P_cell;
            const Tvec *acc_grad_P_cell;

            // For time interpolation
            const Tvec *acc_vel_cell;
            const Tvec *acc_dx_v_cell;
            const Tvec *acc_dy_v_cell;
            const Tvec *acc_dz_v_cell;

            Tscal gamma;
            Tscal dt_interp;

            acc(const Tvec *aabb_block_lower,
                const Tscal *aabb_cell_size,
                const Tscal *P_cell,
                const Tvec *grad_P_cell,
                // For time interpolation
                Tscal dt_interp,
                Tscal gamma,
                const Tvec *vel_cell,
                const Tvec *dx_v_cell,
                const Tvec *dy_v_cell,
                const Tvec *dz_v_cell)
                : shift_get(aabb_block_lower, aabb_cell_size), acc_P_cell{P_cell},
                  acc_grad_P_cell{grad_P_cell}, dt_interp(dt_interp), gamma(gamma),
                  acc_vel_cell{vel_cell}, acc_dx_v_cell{dx_v_cell}, acc_dy_v_cell{dy_v_cell},
                  acc_dz_v_cell{dz_v_cell} {}

            Tscal get_dt_P(
                Tscal P, Tvec grad_P, Tvec v, Tvec dx_v, Tvec dy_v, Tvec dz_v, Tscal gamma) const {
                return -(gamma * P * (dx_v[0] + dy_v[1] + dz_v[2]) + sham::dot(v, grad_P));
            }

            std::array<Tscal, 2> get_link_field_val(u32 id_a, u32 id_b) const {

                auto [shift_a, shift_b] = shift_get.get_shifts(id_a, id_b);

                Tscal P_a     = acc_P_cell[id_a];
                Tvec grad_P_a = acc_grad_P_cell[id_a];
                Tscal P_b     = acc_P_cell[id_b];
                Tvec grad_P_b = acc_grad_P_cell[id_b];

                Tvec v_a    = acc_vel_cell[id_a];
                Tvec dx_v_a = acc_dx_v_cell[id_a];
                Tvec dy_v_a = acc_dy_v_cell[id_a];
                Tvec dz_v_a = acc_dz_v_cell[id_a];
                Tvec v_b    = acc_vel_cell[id_b];
                Tvec dx_v_b = acc_dx_v_cell[id_b];
                Tvec dy_v_b = acc_dy_v_cell[id_b];
                Tvec dz_v_b = acc_dz_v_cell[id_b];

                Tscal dtP_cell_a = get_dt_P(P_a, grad_P_a, v_a, dx_v_a, dy_v_a, dz_v_a, gamma);
                Tscal dtP_cell_b = get_dt_P(P_b, grad_P_b, v_b, dx_v_b, dy_v_b, dz_v_b, gamma);

                Tscal P_face_a = P_a + sycl::dot(grad_P_a, shift_a) + dtP_cell_a * dt_interp;
                Tscal P_face_b = P_b + sycl::dot(grad_P_b, shift_b) + dtP_cell_b * dt_interp;

                return {P_face_a, P_face_b};
            }
        };

        inline acc get_read_access(sham::EventList &deps) {
            return acc(
                aabb_block_lower.get_read_access(deps),
                aabb_cell_size.get_read_access(deps),
                P_cell.get_read_access(deps),
                grad_P_cell.get_read_access(deps),
                dt_interp,
                gamma,
                vel_cell.get_read_access(deps),
                dx_v_cell.get_read_access(deps),
                dy_v_cell.get_read_access(deps),
                dz_v_cell.get_read_access(deps));
        }

        inline void complete_event_state(sycl::event e) {
            aabb_block_lower.complete_event_state(e);
            aabb_cell_size.complete_event_state(e);
            P_cell.complete_event_state(e);
            grad_P_cell.complete_event_state(e);
            vel_cell.complete_event_state(e);
            dx_v_cell.complete_event_state(e);
            dy_v_cell.complete_event_state(e);
            dz_v_cell.complete_event_state(e);
        }
    };

} // namespace

template<class Tvec, class TgridVec>
void shammodels::basegodunov::modules::FaceInterpolate<Tvec, TgridVec>::interpolate_rho_to_face(
    Tscal dt_interp) {

    StackEntry stack_loc{};

    using MergedPDat = shamrock::MergedPatchData;

    solvergraph::NeighGrapkLinkFieldEdge<std::array<Tscal, 2>> &rho_face_xp
        = shambase::get_check_ref(storage.rho_face_xp);
    solvergraph::NeighGrapkLinkFieldEdge<std::array<Tscal, 2>> &rho_face_xm
        = shambase::get_check_ref(storage.rho_face_xm);
    solvergraph::NeighGrapkLinkFieldEdge<std::array<Tscal, 2>> &rho_face_yp
        = shambase::get_check_ref(storage.rho_face_yp);
    solvergraph::NeighGrapkLinkFieldEdge<std::array<Tscal, 2>> &rho_face_ym
        = shambase::get_check_ref(storage.rho_face_ym);
    solvergraph::NeighGrapkLinkFieldEdge<std::array<Tscal, 2>> &rho_face_zp
        = shambase::get_check_ref(storage.rho_face_zp);
    solvergraph::NeighGrapkLinkFieldEdge<std::array<Tscal, 2>> &rho_face_zm
        = shambase::get_check_ref(storage.rho_face_zm);

    rho_face_xp.resize_according_to(
        shambase::get_check_ref(storage.cell_graph_edge).get_refs_dir(OrientedAMRGraph::xp));
    rho_face_xm.resize_according_to(
        shambase::get_check_ref(storage.cell_graph_edge).get_refs_dir(OrientedAMRGraph::xm));
    rho_face_yp.resize_according_to(
        shambase::get_check_ref(storage.cell_graph_edge).get_refs_dir(OrientedAMRGraph::yp));
    rho_face_ym.resize_according_to(
        shambase::get_check_ref(storage.cell_graph_edge).get_refs_dir(OrientedAMRGraph::ym));
    rho_face_zp.resize_according_to(
        shambase::get_check_ref(storage.cell_graph_edge).get_refs_dir(OrientedAMRGraph::zp));
    rho_face_zm.resize_according_to(
        shambase::get_check_ref(storage.cell_graph_edge).get_refs_dir(OrientedAMRGraph::zm));

    auto spans_block_cell_sizes = shambase::get_check_ref(storage.block_cell_sizes).get_spans();
    auto spans_cell0block_aabb_lower
        = shambase::get_check_ref(storage.cell0block_aabb_lower).get_spans();
    auto spans_rhos     = shambase::get_check_ref(storage.refs_rho).get_spans();
    auto spans_grad_rho = shambase::get_check_ref(storage.grad_rho).get_spans();
    auto spans_vel      = shambase::get_check_ref(storage.vel).get_spans();
    auto spans_dx_vel   = shambase::get_check_ref(storage.dx_v).get_spans();
    auto spans_dy_vel   = shambase::get_check_ref(storage.dy_v).get_spans();
    auto spans_dz_vel   = shambase::get_check_ref(storage.dz_v).get_spans();

    using Interp = RhoInterpolate<Tvec, TgridVec, AMRBlock>;
    auto interpolators
        = spans_block_cell_sizes.template map<Interp>([&](u64 id, auto &csize) -> Interp {
              return {
                  spans_cell0block_aabb_lower.get(id),
                  spans_block_cell_sizes.get(id),
                  spans_rhos.get(id),
                  spans_grad_rho.get(id),
                  dt_interp,
                  spans_vel.get(id),
                  spans_dx_vel.get(id),
                  spans_dy_vel.get(id),
                  spans_dz_vel.get(id)};
          });

    auto graphs_xp
        = shambase::get_check_ref(storage.cell_graph_edge).get_refs_dir(OrientedAMRGraph::xp);
    auto graphs_xm
        = shambase::get_check_ref(storage.cell_graph_edge).get_refs_dir(OrientedAMRGraph::xm);
    auto graphs_yp
        = shambase::get_check_ref(storage.cell_graph_edge).get_refs_dir(OrientedAMRGraph::yp);
    auto graphs_ym
        = shambase::get_check_ref(storage.cell_graph_edge).get_refs_dir(OrientedAMRGraph::ym);
    auto graphs_zp
        = shambase::get_check_ref(storage.cell_graph_edge).get_refs_dir(OrientedAMRGraph::zp);
    auto graphs_zm
        = shambase::get_check_ref(storage.cell_graph_edge).get_refs_dir(OrientedAMRGraph::zm);

    shambase::DistributedData<u32> counts_xp
        = graphs_xp.template map<u32>([&](u64 id, auto &graph) {
              return graph.get().obj_cnt;
          });
    shambase::DistributedData<u32> counts_xm
        = graphs_xm.template map<u32>([&](u64 id, auto &graph) {
              return graph.get().obj_cnt;
          });
    shambase::DistributedData<u32> counts_yp
        = graphs_yp.template map<u32>([&](u64 id, auto &graph) {
              return graph.get().obj_cnt;
          });
    shambase::DistributedData<u32> counts_ym
        = graphs_ym.template map<u32>([&](u64 id, auto &graph) {
              return graph.get().obj_cnt;
          });
    shambase::DistributedData<u32> counts_zp
        = graphs_zp.template map<u32>([&](u64 id, auto &graph) {
              return graph.get().obj_cnt;
          });
    shambase::DistributedData<u32> counts_zm
        = graphs_zm.template map<u32>([&](u64 id, auto &graph) {
              return graph.get().obj_cnt;
          });

    sham::distributed_data_kernel_call(
        shamsys::instance::get_compute_scheduler_ptr(),
        sham::DDMultiRef{graphs_xp, interpolators},
        sham::DDMultiRef{rho_face_xp.link_fields},
        counts_xp,
        [](u32 id_a, auto link_iter, auto compute, auto acc_link_field) {
            link_iter.for_each_object_link_id(id_a, [&](u32 id_b, u32 link_id) {
                acc_link_field[link_id] = compute.get_link_field_val(id_a, id_b);
            });
        });
    sham::distributed_data_kernel_call(
        shamsys::instance::get_compute_scheduler_ptr(),
        sham::DDMultiRef{graphs_xm, interpolators},
        sham::DDMultiRef{rho_face_xm.link_fields},
        counts_xm,
        [](u32 id_a, auto link_iter, auto compute, auto acc_link_field) {
            link_iter.for_each_object_link_id(id_a, [&](u32 id_b, u32 link_id) {
                acc_link_field[link_id] = compute.get_link_field_val(id_a, id_b);
            });
        });
    sham::distributed_data_kernel_call(
        shamsys::instance::get_compute_scheduler_ptr(),
        sham::DDMultiRef{graphs_yp, interpolators},
        sham::DDMultiRef{rho_face_yp.link_fields},
        counts_yp,
        [](u32 id_a, auto link_iter, auto compute, auto acc_link_field) {
            link_iter.for_each_object_link_id(id_a, [&](u32 id_b, u32 link_id) {
                acc_link_field[link_id] = compute.get_link_field_val(id_a, id_b);
            });
        });
    sham::distributed_data_kernel_call(
        shamsys::instance::get_compute_scheduler_ptr(),
        sham::DDMultiRef{graphs_ym, interpolators},
        sham::DDMultiRef{rho_face_ym.link_fields},
        counts_ym,
        [](u32 id_a, auto link_iter, auto compute, auto acc_link_field) {
            link_iter.for_each_object_link_id(id_a, [&](u32 id_b, u32 link_id) {
                acc_link_field[link_id] = compute.get_link_field_val(id_a, id_b);
            });
        });
    sham::distributed_data_kernel_call(
        shamsys::instance::get_compute_scheduler_ptr(),
        sham::DDMultiRef{graphs_zp, interpolators},
        sham::DDMultiRef{rho_face_zp.link_fields},
        counts_zp,
        [](u32 id_a, auto link_iter, auto compute, auto acc_link_field) {
            link_iter.for_each_object_link_id(id_a, [&](u32 id_b, u32 link_id) {
                acc_link_field[link_id] = compute.get_link_field_val(id_a, id_b);
            });
        });
    sham::distributed_data_kernel_call(
        shamsys::instance::get_compute_scheduler_ptr(),
        sham::DDMultiRef{graphs_zm, interpolators},
        sham::DDMultiRef{rho_face_zm.link_fields},
        counts_zm,
        [](u32 id_a, auto link_iter, auto compute, auto acc_link_field) {
            link_iter.for_each_object_link_id(id_a, [&](u32 id_b, u32 link_id) {
                acc_link_field[link_id] = compute.get_link_field_val(id_a, id_b);
            });
        });
}

template<class Tvec, class TgridVec>
void shammodels::basegodunov::modules::FaceInterpolate<Tvec, TgridVec>::interpolate_v_to_face(
    Tscal dt_interp) {

    StackEntry stack_loc{};

    using MergedPDat = shamrock::MergedPatchData;

    solvergraph::NeighGrapkLinkFieldEdge<std::array<Tvec, 2>> &vel_face_xp
        = shambase::get_check_ref(storage.vel_face_xp);
    solvergraph::NeighGrapkLinkFieldEdge<std::array<Tvec, 2>> &vel_face_xm
        = shambase::get_check_ref(storage.vel_face_xm);
    solvergraph::NeighGrapkLinkFieldEdge<std::array<Tvec, 2>> &vel_face_yp
        = shambase::get_check_ref(storage.vel_face_yp);
    solvergraph::NeighGrapkLinkFieldEdge<std::array<Tvec, 2>> &vel_face_ym
        = shambase::get_check_ref(storage.vel_face_ym);
    solvergraph::NeighGrapkLinkFieldEdge<std::array<Tvec, 2>> &vel_face_zp
        = shambase::get_check_ref(storage.vel_face_zp);
    solvergraph::NeighGrapkLinkFieldEdge<std::array<Tvec, 2>> &vel_face_zm
        = shambase::get_check_ref(storage.vel_face_zm);

    vel_face_xp.resize_according_to(
        shambase::get_check_ref(storage.cell_graph_edge).get_refs_dir(OrientedAMRGraph::xp));
    vel_face_xm.resize_according_to(
        shambase::get_check_ref(storage.cell_graph_edge).get_refs_dir(OrientedAMRGraph::xm));
    vel_face_yp.resize_according_to(
        shambase::get_check_ref(storage.cell_graph_edge).get_refs_dir(OrientedAMRGraph::yp));
    vel_face_ym.resize_according_to(
        shambase::get_check_ref(storage.cell_graph_edge).get_refs_dir(OrientedAMRGraph::ym));
    vel_face_zp.resize_according_to(
        shambase::get_check_ref(storage.cell_graph_edge).get_refs_dir(OrientedAMRGraph::zp));
    vel_face_zm.resize_according_to(
        shambase::get_check_ref(storage.cell_graph_edge).get_refs_dir(OrientedAMRGraph::zm));

    auto spans_block_cell_sizes = shambase::get_check_ref(storage.block_cell_sizes).get_spans();
    auto spans_cell0block_aabb_lower
        = shambase::get_check_ref(storage.cell0block_aabb_lower).get_spans();
    auto spans_vel    = shambase::get_check_ref(storage.vel).get_spans();
    auto spans_dx_vel = shambase::get_check_ref(storage.dx_v).get_spans();
    auto spans_dy_vel = shambase::get_check_ref(storage.dy_v).get_spans();
    auto spans_dz_vel = shambase::get_check_ref(storage.dz_v).get_spans();
    auto spans_rhos   = shambase::get_check_ref(storage.refs_rho).get_spans();
    auto spans_grad_P = shambase::get_check_ref(storage.grad_P).get_spans();

    using Interp = VelInterpolate<Tvec, TgridVec, AMRBlock>;
    auto interpolators
        = spans_block_cell_sizes.template map<Interp>([&](u64 id, auto &csize) -> Interp {
              return {
                  spans_cell0block_aabb_lower.get(id),
                  spans_block_cell_sizes.get(id),
                  spans_vel.get(id),
                  spans_dx_vel.get(id),
                  spans_dy_vel.get(id),
                  spans_dz_vel.get(id),
                  // For time interpolation
                  dt_interp,
                  spans_rhos.get(id),
                  spans_grad_P.get(id)};
          });

    auto graphs_xp
        = shambase::get_check_ref(storage.cell_graph_edge).get_refs_dir(OrientedAMRGraph::xp);
    auto graphs_xm
        = shambase::get_check_ref(storage.cell_graph_edge).get_refs_dir(OrientedAMRGraph::xm);
    auto graphs_yp
        = shambase::get_check_ref(storage.cell_graph_edge).get_refs_dir(OrientedAMRGraph::yp);
    auto graphs_ym
        = shambase::get_check_ref(storage.cell_graph_edge).get_refs_dir(OrientedAMRGraph::ym);
    auto graphs_zp
        = shambase::get_check_ref(storage.cell_graph_edge).get_refs_dir(OrientedAMRGraph::zp);
    auto graphs_zm
        = shambase::get_check_ref(storage.cell_graph_edge).get_refs_dir(OrientedAMRGraph::zm);

    shambase::DistributedData<u32> counts_xp
        = graphs_xp.template map<u32>([&](u64 id, auto &graph) {
              return graph.get().obj_cnt;
          });
    shambase::DistributedData<u32> counts_xm
        = graphs_xm.template map<u32>([&](u64 id, auto &graph) {
              return graph.get().obj_cnt;
          });
    shambase::DistributedData<u32> counts_yp
        = graphs_yp.template map<u32>([&](u64 id, auto &graph) {
              return graph.get().obj_cnt;
          });
    shambase::DistributedData<u32> counts_ym
        = graphs_ym.template map<u32>([&](u64 id, auto &graph) {
              return graph.get().obj_cnt;
          });
    shambase::DistributedData<u32> counts_zp
        = graphs_zp.template map<u32>([&](u64 id, auto &graph) {
              return graph.get().obj_cnt;
          });
    shambase::DistributedData<u32> counts_zm
        = graphs_zm.template map<u32>([&](u64 id, auto &graph) {
              return graph.get().obj_cnt;
          });

    sham::distributed_data_kernel_call(
        shamsys::instance::get_compute_scheduler_ptr(),
        sham::DDMultiRef{graphs_xp, interpolators},
        sham::DDMultiRef{vel_face_xp.link_fields},
        counts_xp,
        [](u32 id_a, auto link_iter, auto compute, auto acc_link_field) {
            link_iter.for_each_object_link_id(id_a, [&](u32 id_b, u32 link_id) {
                acc_link_field[link_id] = compute.get_link_field_val(id_a, id_b);
            });
        });
    sham::distributed_data_kernel_call(
        shamsys::instance::get_compute_scheduler_ptr(),
        sham::DDMultiRef{graphs_xm, interpolators},
        sham::DDMultiRef{vel_face_xm.link_fields},
        counts_xm,
        [](u32 id_a, auto link_iter, auto compute, auto acc_link_field) {
            link_iter.for_each_object_link_id(id_a, [&](u32 id_b, u32 link_id) {
                acc_link_field[link_id] = compute.get_link_field_val(id_a, id_b);
            });
        });
    sham::distributed_data_kernel_call(
        shamsys::instance::get_compute_scheduler_ptr(),
        sham::DDMultiRef{graphs_yp, interpolators},
        sham::DDMultiRef{vel_face_yp.link_fields},
        counts_yp,
        [](u32 id_a, auto link_iter, auto compute, auto acc_link_field) {
            link_iter.for_each_object_link_id(id_a, [&](u32 id_b, u32 link_id) {
                acc_link_field[link_id] = compute.get_link_field_val(id_a, id_b);
            });
        });
    sham::distributed_data_kernel_call(
        shamsys::instance::get_compute_scheduler_ptr(),
        sham::DDMultiRef{graphs_ym, interpolators},
        sham::DDMultiRef{vel_face_ym.link_fields},
        counts_ym,
        [](u32 id_a, auto link_iter, auto compute, auto acc_link_field) {
            link_iter.for_each_object_link_id(id_a, [&](u32 id_b, u32 link_id) {
                acc_link_field[link_id] = compute.get_link_field_val(id_a, id_b);
            });
        });
    sham::distributed_data_kernel_call(
        shamsys::instance::get_compute_scheduler_ptr(),
        sham::DDMultiRef{graphs_zp, interpolators},
        sham::DDMultiRef{vel_face_zp.link_fields},
        counts_zp,
        [](u32 id_a, auto link_iter, auto compute, auto acc_link_field) {
            link_iter.for_each_object_link_id(id_a, [&](u32 id_b, u32 link_id) {
                acc_link_field[link_id] = compute.get_link_field_val(id_a, id_b);
            });
        });
    sham::distributed_data_kernel_call(
        shamsys::instance::get_compute_scheduler_ptr(),
        sham::DDMultiRef{graphs_zm, interpolators},
        sham::DDMultiRef{vel_face_zm.link_fields},
        counts_zm,
        [](u32 id_a, auto link_iter, auto compute, auto acc_link_field) {
            link_iter.for_each_object_link_id(id_a, [&](u32 id_b, u32 link_id) {
                acc_link_field[link_id] = compute.get_link_field_val(id_a, id_b);
            });
        });
}

template<class Tvec, class TgridVec>
void shammodels::basegodunov::modules::FaceInterpolate<Tvec, TgridVec>::interpolate_P_to_face(
    Tscal dt_interp) {

    Tscal gamma = solver_config.eos_gamma;

    StackEntry stack_loc{};

    using MergedPDat = shamrock::MergedPatchData;

    solvergraph::NeighGrapkLinkFieldEdge<std::array<Tscal, 2>> &press_face_xp
        = shambase::get_check_ref(storage.press_face_xp);
    solvergraph::NeighGrapkLinkFieldEdge<std::array<Tscal, 2>> &press_face_xm
        = shambase::get_check_ref(storage.press_face_xm);
    solvergraph::NeighGrapkLinkFieldEdge<std::array<Tscal, 2>> &press_face_yp
        = shambase::get_check_ref(storage.press_face_yp);
    solvergraph::NeighGrapkLinkFieldEdge<std::array<Tscal, 2>> &press_face_ym
        = shambase::get_check_ref(storage.press_face_ym);
    solvergraph::NeighGrapkLinkFieldEdge<std::array<Tscal, 2>> &press_face_zp
        = shambase::get_check_ref(storage.press_face_zp);
    solvergraph::NeighGrapkLinkFieldEdge<std::array<Tscal, 2>> &press_face_zm
        = shambase::get_check_ref(storage.press_face_zm);

    press_face_xp.resize_according_to(
        shambase::get_check_ref(storage.cell_graph_edge).get_refs_dir(OrientedAMRGraph::xp));
    press_face_xm.resize_according_to(
        shambase::get_check_ref(storage.cell_graph_edge).get_refs_dir(OrientedAMRGraph::xm));
    press_face_yp.resize_according_to(
        shambase::get_check_ref(storage.cell_graph_edge).get_refs_dir(OrientedAMRGraph::yp));
    press_face_ym.resize_according_to(
        shambase::get_check_ref(storage.cell_graph_edge).get_refs_dir(OrientedAMRGraph::ym));
    press_face_zp.resize_according_to(
        shambase::get_check_ref(storage.cell_graph_edge).get_refs_dir(OrientedAMRGraph::zp));
    press_face_zm.resize_according_to(
        shambase::get_check_ref(storage.cell_graph_edge).get_refs_dir(OrientedAMRGraph::zm));

    auto spans_block_cell_sizes = shambase::get_check_ref(storage.block_cell_sizes).get_spans();
    auto spans_cell0block_aabb_lower
        = shambase::get_check_ref(storage.cell0block_aabb_lower).get_spans();
    auto spans_press     = shambase::get_check_ref(storage.press).get_spans();
    auto spans_grad_P = shambase::get_check_ref(storage.grad_P).get_spans();
    auto spans_vel      = shambase::get_check_ref(storage.vel).get_spans();
    auto spans_dx_vel   = shambase::get_check_ref(storage.dx_v).get_spans();
    auto spans_dy_vel   = shambase::get_check_ref(storage.dy_v).get_spans();
    auto spans_dz_vel   = shambase::get_check_ref(storage.dz_v).get_spans();

    using Interp = PressInterpolate<Tvec, TgridVec, AMRBlock>;
    auto interpolators
        = spans_block_cell_sizes.template map<Interp>([&](u64 id, auto &csize) -> Interp {
              return {
                  spans_cell0block_aabb_lower.get(id),
                  spans_block_cell_sizes.get(id),
                  spans_rhos.get(id),
                  spans_grad_rho.get(id),
                  dt_interp,
                  spans_vel.get(id),
                  spans_dx_vel.get(id),
                  spans_dy_vel.get(id),
                  spans_dz_vel.get(id)};
          });

    auto graphs_xp
        = shambase::get_check_ref(storage.cell_graph_edge).get_refs_dir(OrientedAMRGraph::xp);
    auto graphs_xm
        = shambase::get_check_ref(storage.cell_graph_edge).get_refs_dir(OrientedAMRGraph::xm);
    auto graphs_yp
        = shambase::get_check_ref(storage.cell_graph_edge).get_refs_dir(OrientedAMRGraph::yp);
    auto graphs_ym
        = shambase::get_check_ref(storage.cell_graph_edge).get_refs_dir(OrientedAMRGraph::ym);
    auto graphs_zp
        = shambase::get_check_ref(storage.cell_graph_edge).get_refs_dir(OrientedAMRGraph::zp);
    auto graphs_zm
        = shambase::get_check_ref(storage.cell_graph_edge).get_refs_dir(OrientedAMRGraph::zm);

    shambase::DistributedData<u32> counts_xp
        = graphs_xp.template map<u32>([&](u64 id, auto &graph) {
              return graph.get().obj_cnt;
          });
    shambase::DistributedData<u32> counts_xm
        = graphs_xm.template map<u32>([&](u64 id, auto &graph) {
              return graph.get().obj_cnt;
          });
    shambase::DistributedData<u32> counts_yp
        = graphs_yp.template map<u32>([&](u64 id, auto &graph) {
              return graph.get().obj_cnt;
          });
    shambase::DistributedData<u32> counts_ym
        = graphs_ym.template map<u32>([&](u64 id, auto &graph) {
              return graph.get().obj_cnt;
          });
    shambase::DistributedData<u32> counts_zp
        = graphs_zp.template map<u32>([&](u64 id, auto &graph) {
              return graph.get().obj_cnt;
          });
    shambase::DistributedData<u32> counts_zm
        = graphs_zm.template map<u32>([&](u64 id, auto &graph) {
              return graph.get().obj_cnt;
          });

    shambase::get_check_ref(storage.cell_graph_edge)
        .graph.for_each([&](u64 id, OrientedAMRGraph &oriented_cell_graph) {
            sham::DeviceQueue &q = shamsys::instance::get_compute_scheduler().get_queue();
            MergedPDat &mpdat    = storage.merged_patchdata_ghost.get().get(id);

            sham::DeviceBuffer<Tscal> &block_cell_sizes
                = shambase::get_check_ref(storage.block_cell_sizes)
                      .get_refs()
                      .get(id)
                      .get()
                      .get_buf();
            sham::DeviceBuffer<Tvec> &cell0block_aabb_lower
                = shambase::get_check_ref(storage.cell0block_aabb_lower)
                      .get_refs()
                      .get(id)
                      .get()
                      .get_buf();

            sham::DeviceBuffer<Tscal> &buf_press
                = shambase::get_check_ref(storage.press).get_buf(id);
            sham::DeviceBuffer<Tvec> &buf_grad_P
                = shambase::get_check_ref(storage.grad_P).get_buf(id);

            sham::DeviceBuffer<Tvec> &buf_vel = shambase::get_check_ref(storage.vel).get_buf(id);
            sham::DeviceBuffer<Tvec> &buf_dx_vel
                = shambase::get_check_ref(storage.dx_v).get_buf(id);
            sham::DeviceBuffer<Tvec> &buf_dy_vel
                = shambase::get_check_ref(storage.dy_v).get_buf(id);
            sham::DeviceBuffer<Tvec> &buf_dz_vel
                = shambase::get_check_ref(storage.dz_v).get_buf(id);

            // TODO : restore asynchroneousness
            sham::EventList depends_list;
            auto ptr_block_cell_sizes      = block_cell_sizes.get_read_access(depends_list);
            auto ptr_cell0block_aabb_lower = cell0block_aabb_lower.get_read_access(depends_list);

            auto ptr_buf_press  = buf_press.get_read_access(depends_list);
            auto ptr_buf_grad_P = buf_grad_P.get_read_access(depends_list);

            auto ptr_buf_vel    = buf_vel.get_read_access(depends_list);
            auto ptr_buf_dx_vel = buf_dx_vel.get_read_access(depends_list);
            auto ptr_buf_dy_vel = buf_dy_vel.get_read_access(depends_list);
            auto ptr_buf_dz_vel = buf_dz_vel.get_read_access(depends_list);

            sham::EventList resulting_event_list;

            logger::debug_ln("Face Interpolate", "patch", id, "intepolate press");

            update_link_field<PressInterpolate, std::array<Tscal, 2>>(
                q,
                depends_list,
                resulting_event_list,
                press_face_xp.link_fields.get(id),
                shambase::get_check_ref(oriented_cell_graph.graph_links[oriented_cell_graph.xp]),
                ptr_cell0block_aabb_lower,
                ptr_block_cell_sizes,
                ptr_buf_press,
                ptr_buf_grad_P,
                dt_interp,
                gamma,
                ptr_buf_vel,
                ptr_buf_dx_vel,
                ptr_buf_dy_vel,
                ptr_buf_dz_vel);
            update_link_field<PressInterpolate, std::array<Tscal, 2>>(
                q,
                depends_list,
                resulting_event_list,
                press_face_xm.link_fields.get(id),
                shambase::get_check_ref(oriented_cell_graph.graph_links[oriented_cell_graph.xm]),
                ptr_cell0block_aabb_lower,
                ptr_block_cell_sizes,
                ptr_buf_press,
                ptr_buf_grad_P,
                dt_interp,
                gamma,
                ptr_buf_vel,
                ptr_buf_dx_vel,
                ptr_buf_dy_vel,
                ptr_buf_dz_vel);
            update_link_field<PressInterpolate, std::array<Tscal, 2>>(
                q,
                depends_list,
                resulting_event_list,
                press_face_yp.link_fields.get(id),
                shambase::get_check_ref(oriented_cell_graph.graph_links[oriented_cell_graph.yp]),
                ptr_cell0block_aabb_lower,
                ptr_block_cell_sizes,
                ptr_buf_press,
                ptr_buf_grad_P,
                dt_interp,
                gamma,
                ptr_buf_vel,
                ptr_buf_dx_vel,
                ptr_buf_dy_vel,
                ptr_buf_dz_vel);
            update_link_field<PressInterpolate, std::array<Tscal, 2>>(
                q,
                depends_list,
                resulting_event_list,
                press_face_ym.link_fields.get(id),
                shambase::get_check_ref(oriented_cell_graph.graph_links[oriented_cell_graph.ym]),
                ptr_cell0block_aabb_lower,
                ptr_block_cell_sizes,
                ptr_buf_press,
                ptr_buf_grad_P,
                dt_interp,
                gamma,
                ptr_buf_vel,
                ptr_buf_dx_vel,
                ptr_buf_dy_vel,
                ptr_buf_dz_vel);
            update_link_field<PressInterpolate, std::array<Tscal, 2>>(
                q,
                depends_list,
                resulting_event_list,
                press_face_zp.link_fields.get(id),
                shambase::get_check_ref(oriented_cell_graph.graph_links[oriented_cell_graph.zp]),
                ptr_cell0block_aabb_lower,
                ptr_block_cell_sizes,
                ptr_buf_press,
                ptr_buf_grad_P,
                dt_interp,
                gamma,
                ptr_buf_vel,
                ptr_buf_dx_vel,
                ptr_buf_dy_vel,
                ptr_buf_dz_vel);
            update_link_field<PressInterpolate, std::array<Tscal, 2>>(
                q,
                depends_list,
                resulting_event_list,
                press_face_zm.link_fields.get(id),
                shambase::get_check_ref(oriented_cell_graph.graph_links[oriented_cell_graph.zm]),
                ptr_cell0block_aabb_lower,
                ptr_block_cell_sizes,
                ptr_buf_press,
                ptr_buf_grad_P,
                dt_interp,
                gamma,
                ptr_buf_vel,
                ptr_buf_dx_vel,
                ptr_buf_dy_vel,
                ptr_buf_dz_vel);

            block_cell_sizes.complete_event_state(resulting_event_list);
            cell0block_aabb_lower.complete_event_state(resulting_event_list);
            buf_press.complete_event_state(resulting_event_list);
            buf_grad_P.complete_event_state(resulting_event_list);
            buf_vel.complete_event_state(resulting_event_list);
            buf_dx_vel.complete_event_state(resulting_event_list);
            buf_dy_vel.complete_event_state(resulting_event_list);
            buf_dz_vel.complete_event_state(resulting_event_list);
        });
}

template<class Tvec, class TgridVec>
void shammodels::basegodunov::modules::FaceInterpolate<Tvec, TgridVec>::
    interpolate_rho_dust_to_face(Tscal dt_interp) {

    class RhoDustInterpolate {
        public:
        GetShift<Tvec, TgridVec, AMRBlock> shift_get;
        u32 nvar;

        const Tscal *acc_rho_dust_cell;
        const Tvec *acc_grad_rho_dust_cell;

        // For time interpolation
        const Tvec *acc_vel_dust_cell;
        const Tvec *acc_dx_v_dust_cell;
        const Tvec *acc_dy_v_dust_cell;
        const Tvec *acc_dz_v_dust_cell;

        Tscal dt_interp;

        RhoDustInterpolate(
            u32 nvar,
            const Tvec *aabb_block_lower,
            const Tscal *aabb_cell_size,
            const Tscal *rho_dust_cell,
            const Tvec *grad_rho_dust_cell,
            // For time interpolation
            Tscal dt_interp,
            const Tvec *vel_dust_cell,
            const Tvec *dx_v_dust_cell,
            const Tvec *dy_v_dust_cell,
            const Tvec *dz_v_dust_cell)
            : shift_get(aabb_block_lower, aabb_cell_size), nvar(nvar),
              acc_rho_dust_cell{rho_dust_cell}, acc_grad_rho_dust_cell{grad_rho_dust_cell},
              dt_interp(dt_interp), acc_vel_dust_cell{vel_dust_cell},
              acc_dx_v_dust_cell{dx_v_dust_cell}, acc_dy_v_dust_cell{dy_v_dust_cell},
              acc_dz_v_dust_cell{dz_v_dust_cell} {}

        Tscal get_dt_rho_dust(
            Tscal rho_dust,
            Tvec v_dust,
            Tvec grad_rho_dust,
            Tvec dx_v_dust,
            Tvec dy_v_dust,
            Tvec dz_v_dust) const {
            return -(
                sham::dot(v_dust, grad_rho_dust)
                + rho_dust * (dx_v_dust[0] + dy_v_dust[1] + dz_v_dust[2]));
        }

        std::array<Tscal, 2> get_link_field_val(u32 id_a, u32 id_b) const {
            const u32 icell_a = id_a / nvar;
            const u32 icell_b = id_b / nvar;

            auto [shift_a, shift_b] = shift_get.get_shifts(icell_a, icell_b);

            Tscal rho_dust_a     = acc_rho_dust_cell[id_a];
            Tvec grad_rho_dust_a = acc_grad_rho_dust_cell[id_a];
            Tscal rho_dust_b     = acc_rho_dust_cell[id_b];
            Tvec grad_rho_dust_b = acc_grad_rho_dust_cell[id_b];

            Tvec vel_dust_a  = acc_vel_dust_cell[id_a];
            Tvec dx_v_dust_a = acc_dx_v_dust_cell[id_a];
            Tvec dy_v_dust_a = acc_dy_v_dust_cell[id_a];
            Tvec dz_v_dust_a = acc_dz_v_dust_cell[id_a];
            Tvec vel_dust_b  = acc_vel_dust_cell[id_b];
            Tvec dx_v_dust_b = acc_dx_v_dust_cell[id_b];
            Tvec dy_v_dust_b = acc_dy_v_dust_cell[id_b];
            Tvec dz_v_dust_b = acc_dz_v_dust_cell[id_b];

            Tscal rho_dust_face_a = rho_dust_a + sycl::dot(grad_rho_dust_a, shift_a);
            Tscal rho_dust_face_b = rho_dust_b + sycl::dot(grad_rho_dust_b, shift_b);

            rho_dust_face_a += get_dt_rho_dust(
                                   rho_dust_a,
                                   vel_dust_a,
                                   grad_rho_dust_a,
                                   dx_v_dust_a,
                                   dy_v_dust_a,
                                   dz_v_dust_a)
                               * dt_interp;
            rho_dust_face_b += get_dt_rho_dust(
                                   rho_dust_b,
                                   vel_dust_b,
                                   grad_rho_dust_b,
                                   dx_v_dust_b,
                                   dy_v_dust_b,
                                   dz_v_dust_b)
                               * dt_interp;

            return {rho_dust_face_a, rho_dust_face_b};
        }
    };

    StackEntry stack_loc{};

    using MergedPDat = shamrock::MergedPatchData;

    solvergraph::NeighGrapkLinkFieldEdge<std::array<Tscal, 2>> &rho_dust_face_xp
        = shambase::get_check_ref(storage.rho_dust_face_xp);
    solvergraph::NeighGrapkLinkFieldEdge<std::array<Tscal, 2>> &rho_dust_face_xm
        = shambase::get_check_ref(storage.rho_dust_face_xm);
    solvergraph::NeighGrapkLinkFieldEdge<std::array<Tscal, 2>> &rho_dust_face_yp
        = shambase::get_check_ref(storage.rho_dust_face_yp);
    solvergraph::NeighGrapkLinkFieldEdge<std::array<Tscal, 2>> &rho_dust_face_ym
        = shambase::get_check_ref(storage.rho_dust_face_ym);
    solvergraph::NeighGrapkLinkFieldEdge<std::array<Tscal, 2>> &rho_dust_face_zp
        = shambase::get_check_ref(storage.rho_dust_face_zp);
    solvergraph::NeighGrapkLinkFieldEdge<std::array<Tscal, 2>> &rho_dust_face_zm
        = shambase::get_check_ref(storage.rho_dust_face_zm);

    rho_dust_face_xp.resize_according_to(
        shambase::get_check_ref(storage.cell_graph_edge).get_refs_dir(OrientedAMRGraph::xp));
    rho_dust_face_xm.resize_according_to(
        shambase::get_check_ref(storage.cell_graph_edge).get_refs_dir(OrientedAMRGraph::xm));
    rho_dust_face_yp.resize_according_to(
        shambase::get_check_ref(storage.cell_graph_edge).get_refs_dir(OrientedAMRGraph::yp));
    rho_dust_face_ym.resize_according_to(
        shambase::get_check_ref(storage.cell_graph_edge).get_refs_dir(OrientedAMRGraph::ym));
    rho_dust_face_zp.resize_according_to(
        shambase::get_check_ref(storage.cell_graph_edge).get_refs_dir(OrientedAMRGraph::zp));
    rho_dust_face_zm.resize_according_to(
        shambase::get_check_ref(storage.cell_graph_edge).get_refs_dir(OrientedAMRGraph::zm));

    shamrock::patch::PatchDataLayout &ghost_layout = storage.ghost_layout.get();
    u32 irho_dust_ghost                            = ghost_layout.get_field_idx<Tscal>("rho_dust");
    u32 ndust                                      = solver_config.dust_config.ndust;

    shambase::get_check_ref(storage.cell_graph_edge)
        .graph.for_each([&](u64 id, OrientedAMRGraph &oriented_cell_graph) {
            sham::DeviceQueue &q = shamsys::instance::get_compute_scheduler().get_queue();
            MergedPDat &mpdat    = storage.merged_patchdata_ghost.get().get(id);

            sham::DeviceBuffer<Tscal> &block_cell_sizes
                = shambase::get_check_ref(storage.block_cell_sizes)
                      .get_refs()
                      .get(id)
                      .get()
                      .get_buf();
            sham::DeviceBuffer<Tvec> &cell0block_aabb_lower
                = shambase::get_check_ref(storage.cell0block_aabb_lower)
                      .get_refs()
                      .get(id)
                      .get()
                      .get_buf();

            sham::DeviceBuffer<Tscal> &buf_rho_dust
                = mpdat.pdat.get_field_buf_ref<Tscal>(irho_dust_ghost);
            sham::DeviceBuffer<Tvec> &buf_grad_rho_dust
                = shambase::get_check_ref(storage.grad_rho_dust).get_buf(id);

            sham::DeviceBuffer<Tvec> &buf_vel_dust
                = shambase::get_check_ref(storage.vel_dust).get_buf(id);
            sham::DeviceBuffer<Tvec> &buf_dx_vel_dust
                = shambase::get_check_ref(storage.dx_v_dust).get_buf(id);
            sham::DeviceBuffer<Tvec> &buf_dy_vel_dust
                = shambase::get_check_ref(storage.dy_v_dust).get_buf(id);
            sham::DeviceBuffer<Tvec> &buf_dz_vel_dust
                = shambase::get_check_ref(storage.dz_v_dust).get_buf(id);

            // TODO : restore asynchroneousness
            sham::EventList depends_list;
            auto ptr_block_cell_sizes      = block_cell_sizes.get_read_access(depends_list);
            auto ptr_cell0block_aabb_lower = cell0block_aabb_lower.get_read_access(depends_list);

            auto ptr_rho_dust      = buf_rho_dust.get_read_access(depends_list);
            auto ptr_grad_rho_dust = buf_grad_rho_dust.get_read_access(depends_list);

            auto ptr_vel_dust    = buf_vel_dust.get_read_access(depends_list);
            auto ptr_dx_vel_dust = buf_dx_vel_dust.get_read_access(depends_list);
            auto ptr_dy_vel_dust = buf_dy_vel_dust.get_read_access(depends_list);
            auto ptr_dz_vel_dust = buf_dz_vel_dust.get_read_access(depends_list);

            sham::EventList resulting_event_list;

            logger::debug_ln("Face Interpolate", "patch", id, "intepolate rho dust");

            update_link_field_indep_nvar<RhoDustInterpolate, std::array<Tscal, 2>>(
                q,
                depends_list,
                resulting_event_list,
                rho_dust_face_xp.link_fields.get(id),
                shambase::get_check_ref(oriented_cell_graph.graph_links[oriented_cell_graph.xp]),
                ndust,
                ptr_cell0block_aabb_lower,
                ptr_block_cell_sizes,
                ptr_rho_dust,
                ptr_grad_rho_dust,
                dt_interp,
                ptr_vel_dust,
                ptr_dx_vel_dust,
                ptr_dy_vel_dust,
                ptr_dz_vel_dust);
            update_link_field_indep_nvar<RhoDustInterpolate, std::array<Tscal, 2>>(
                q,
                depends_list,
                resulting_event_list,
                rho_dust_face_xm.link_fields.get(id),
                shambase::get_check_ref(oriented_cell_graph.graph_links[oriented_cell_graph.xm]),
                ndust,
                ptr_cell0block_aabb_lower,
                ptr_block_cell_sizes,
                ptr_rho_dust,
                ptr_grad_rho_dust,
                dt_interp,
                ptr_vel_dust,
                ptr_dx_vel_dust,
                ptr_dy_vel_dust,
                ptr_dz_vel_dust);
            update_link_field_indep_nvar<RhoDustInterpolate, std::array<Tscal, 2>>(
                q,
                depends_list,
                resulting_event_list,
                rho_dust_face_yp.link_fields.get(id),
                shambase::get_check_ref(oriented_cell_graph.graph_links[oriented_cell_graph.yp]),
                ndust,
                ptr_cell0block_aabb_lower,
                ptr_block_cell_sizes,
                ptr_rho_dust,
                ptr_grad_rho_dust,
                dt_interp,
                ptr_vel_dust,
                ptr_dx_vel_dust,
                ptr_dy_vel_dust,
                ptr_dz_vel_dust);
            update_link_field_indep_nvar<RhoDustInterpolate, std::array<Tscal, 2>>(
                q,
                depends_list,
                resulting_event_list,
                rho_dust_face_ym.link_fields.get(id),
                shambase::get_check_ref(oriented_cell_graph.graph_links[oriented_cell_graph.ym]),
                ndust,
                ptr_cell0block_aabb_lower,
                ptr_block_cell_sizes,
                ptr_rho_dust,
                ptr_grad_rho_dust,
                dt_interp,
                ptr_vel_dust,
                ptr_dx_vel_dust,
                ptr_dy_vel_dust,
                ptr_dz_vel_dust);
            update_link_field_indep_nvar<RhoDustInterpolate, std::array<Tscal, 2>>(
                q,
                depends_list,
                resulting_event_list,
                rho_dust_face_zp.link_fields.get(id),
                shambase::get_check_ref(oriented_cell_graph.graph_links[oriented_cell_graph.zp]),
                ndust,
                ptr_cell0block_aabb_lower,
                ptr_block_cell_sizes,
                ptr_rho_dust,
                ptr_grad_rho_dust,
                dt_interp,
                ptr_vel_dust,
                ptr_dx_vel_dust,
                ptr_dy_vel_dust,
                ptr_dz_vel_dust);
            update_link_field_indep_nvar<RhoDustInterpolate, std::array<Tscal, 2>>(
                q,
                depends_list,
                resulting_event_list,
                rho_dust_face_zm.link_fields.get(id),
                shambase::get_check_ref(oriented_cell_graph.graph_links[oriented_cell_graph.zm]),
                ndust,
                ptr_cell0block_aabb_lower,
                ptr_block_cell_sizes,
                ptr_rho_dust,
                ptr_grad_rho_dust,
                dt_interp,
                ptr_vel_dust,
                ptr_dx_vel_dust,
                ptr_dy_vel_dust,
                ptr_dz_vel_dust);

            block_cell_sizes.complete_event_state(resulting_event_list);
            cell0block_aabb_lower.complete_event_state(resulting_event_list);
            buf_rho_dust.complete_event_state(resulting_event_list);
            buf_grad_rho_dust.complete_event_state(resulting_event_list);
            buf_vel_dust.complete_event_state(resulting_event_list);
            buf_dx_vel_dust.complete_event_state(resulting_event_list);
            buf_dy_vel_dust.complete_event_state(resulting_event_list);
            buf_dz_vel_dust.complete_event_state(resulting_event_list);
        });
}

template<class Tvec, class TgridVec>
void shammodels::basegodunov::modules::FaceInterpolate<Tvec, TgridVec>::interpolate_v_dust_to_face(
    Tscal dt_interp) {

    class VelDustInterpolate {

        public:
        GetShift<Tvec, TgridVec, AMRBlock> shift_get;
        u32 nvar;

        const Tvec *acc_vel_dust_cell;
        const Tvec *acc_dx_v_dust_cell;
        const Tvec *acc_dy_v_dust_cell;
        const Tvec *acc_dz_v_dust_cell;

        // For time interpolation
        const Tscal *acc_rho_dust_cell;

        Tscal dt_interp;

        VelDustInterpolate(
            u32 nvar,
            const Tvec *aabb_block_lower,
            const Tscal *aabb_cell_size,
            const Tvec *vel_dust_cell,
            const Tvec *dx_v_dust_cell,
            const Tvec *dy_v_dust_cell,
            const Tvec *dz_v_dust_cell,
            // For time interpolation
            Tscal dt_interp,
            const Tscal *rho_dust_cell)
            : shift_get(aabb_block_lower, aabb_cell_size), nvar(nvar),
              acc_vel_dust_cell{vel_dust_cell}, acc_dx_v_dust_cell{dx_v_dust_cell},
              acc_dy_v_dust_cell{dy_v_dust_cell}, acc_dz_v_dust_cell{dz_v_dust_cell},
              dt_interp(dt_interp), acc_rho_dust_cell{rho_dust_cell} {}

        Tvec get_dt_v_dust(Tvec v, Tvec dx_v, Tvec dy_v, Tvec dz_v, Tscal rho) const {
            return -(v[0] * dx_v + v[1] * dy_v + v[2] * dz_v);
        }

        std::array<Tvec, 2> get_link_field_val(u32 id_a, u32 id_b) const {
            const u32 icell_a = id_a / nvar;
            const u32 icell_b = id_b / nvar;

            auto [shift_a, shift_b] = shift_get.get_shifts(icell_a, icell_b);

            Tvec v_dust_a      = acc_vel_dust_cell[id_a];
            Tvec dx_vel_dust_a = acc_dx_v_dust_cell[id_a];
            Tvec dy_vel_dust_a = acc_dy_v_dust_cell[id_a];
            Tvec dz_vel_dust_a = acc_dz_v_dust_cell[id_a];

            Tvec v_dust_b      = acc_vel_dust_cell[id_b];
            Tvec dx_vel_dust_b = acc_dx_v_dust_cell[id_b];
            Tvec dy_vel_dust_b = acc_dy_v_dust_cell[id_b];
            Tvec dz_vel_dust_b = acc_dz_v_dust_cell[id_b];

            Tscal rho_dust_a = acc_rho_dust_cell[id_a];
            Tscal rho_dust_b = acc_rho_dust_cell[id_b];

            Tvec dx_v_dust_a_dot_shift = shift_a.x() * dx_vel_dust_a + shift_a.y() * dy_vel_dust_a
                                         + shift_a.z() * dz_vel_dust_a;
            Tvec dx_v_dust_b_dot_shift = shift_b.x() * dx_vel_dust_b + shift_b.y() * dy_vel_dust_b
                                         + shift_b.z() * dz_vel_dust_b;

            Tvec dt_v_dust_a
                = get_dt_v_dust(v_dust_a, dx_vel_dust_a, dy_vel_dust_a, dz_vel_dust_a, rho_dust_a);
            Tvec dt_v_dust_b
                = get_dt_v_dust(v_dust_b, dx_vel_dust_b, dy_vel_dust_b, dz_vel_dust_b, rho_dust_b);

            Tvec vel_dust_face_a = v_dust_a + dx_v_dust_a_dot_shift + dt_v_dust_a * dt_interp;
            Tvec vel_dust_face_b = v_dust_b + dx_v_dust_b_dot_shift + dt_v_dust_b * dt_interp;

            return {vel_dust_face_a, vel_dust_face_b};
        }
    };

    StackEntry stack_loc{};

    using MergedPDat = shamrock::MergedPatchData;

    solvergraph::NeighGrapkLinkFieldEdge<std::array<Tvec, 2>> &vel_dust_face_xp
        = shambase::get_check_ref(storage.vel_dust_face_xp);
    solvergraph::NeighGrapkLinkFieldEdge<std::array<Tvec, 2>> &vel_dust_face_xm
        = shambase::get_check_ref(storage.vel_dust_face_xm);
    solvergraph::NeighGrapkLinkFieldEdge<std::array<Tvec, 2>> &vel_dust_face_yp
        = shambase::get_check_ref(storage.vel_dust_face_yp);
    solvergraph::NeighGrapkLinkFieldEdge<std::array<Tvec, 2>> &vel_dust_face_ym
        = shambase::get_check_ref(storage.vel_dust_face_ym);
    solvergraph::NeighGrapkLinkFieldEdge<std::array<Tvec, 2>> &vel_dust_face_zp
        = shambase::get_check_ref(storage.vel_dust_face_zp);
    solvergraph::NeighGrapkLinkFieldEdge<std::array<Tvec, 2>> &vel_dust_face_zm
        = shambase::get_check_ref(storage.vel_dust_face_zm);

    vel_dust_face_xp.resize_according_to(
        shambase::get_check_ref(storage.cell_graph_edge).get_refs_dir(OrientedAMRGraph::xp));
    vel_dust_face_xm.resize_according_to(
        shambase::get_check_ref(storage.cell_graph_edge).get_refs_dir(OrientedAMRGraph::xm));
    vel_dust_face_yp.resize_according_to(
        shambase::get_check_ref(storage.cell_graph_edge).get_refs_dir(OrientedAMRGraph::yp));
    vel_dust_face_ym.resize_according_to(
        shambase::get_check_ref(storage.cell_graph_edge).get_refs_dir(OrientedAMRGraph::ym));
    vel_dust_face_zp.resize_according_to(
        shambase::get_check_ref(storage.cell_graph_edge).get_refs_dir(OrientedAMRGraph::zp));
    vel_dust_face_zm.resize_according_to(
        shambase::get_check_ref(storage.cell_graph_edge).get_refs_dir(OrientedAMRGraph::zm));

    shamrock::patch::PatchDataLayout &ghost_layout = storage.ghost_layout.get();
    u32 irho_dust_ghost                            = ghost_layout.get_field_idx<Tscal>("rho_dust");
    u32 ndust                                      = solver_config.dust_config.ndust;
    shambase::get_check_ref(storage.cell_graph_edge)
        .graph.for_each([&](u64 id, OrientedAMRGraph &oriented_cell_graph) {
            sham::DeviceQueue &q = shamsys::instance::get_compute_scheduler().get_queue();
            MergedPDat &mpdat    = storage.merged_patchdata_ghost.get().get(id);

            sham::DeviceBuffer<Tscal> &block_cell_sizes
                = shambase::get_check_ref(storage.block_cell_sizes)
                      .get_refs()
                      .get(id)
                      .get()
                      .get_buf();
            sham::DeviceBuffer<Tvec> &cell0block_aabb_lower
                = shambase::get_check_ref(storage.cell0block_aabb_lower)
                      .get_refs()
                      .get(id)
                      .get()
                      .get_buf();

            sham::DeviceBuffer<Tvec> &buf_vel_dust
                = shambase::get_check_ref(storage.vel_dust).get_buf(id);
            sham::DeviceBuffer<Tvec> &buf_dx_vel_dust
                = shambase::get_check_ref(storage.dx_v_dust).get_buf(id);
            sham::DeviceBuffer<Tvec> &buf_dy_vel_dust
                = shambase::get_check_ref(storage.dy_v_dust).get_buf(id);
            sham::DeviceBuffer<Tvec> &buf_dz_vel_dust
                = shambase::get_check_ref(storage.dz_v_dust).get_buf(id);

            sham::DeviceBuffer<Tscal> &buf_rho_dust
                = mpdat.pdat.get_field_buf_ref<Tscal>(irho_dust_ghost);

            // TODO : restore asynchroneousness
            sham::EventList depends_list;
            auto ptr_block_cell_sizes      = block_cell_sizes.get_read_access(depends_list);
            auto ptr_cell0block_aabb_lower = cell0block_aabb_lower.get_read_access(depends_list);
            auto ptr_buf_vel_dust          = buf_vel_dust.get_read_access(depends_list);
            auto ptr_buf_dx_vel_dust       = buf_dx_vel_dust.get_read_access(depends_list);
            auto ptr_buf_dy_vel_dust       = buf_dy_vel_dust.get_read_access(depends_list);
            auto ptr_buf_dz_vel_dust       = buf_dz_vel_dust.get_read_access(depends_list);
            auto ptr_buf_rho_dust          = buf_rho_dust.get_read_access(depends_list);

            sham::EventList resulting_event_list;

            logger::debug_ln("Face Interpolate", "patch", id, "intepolate vel");

            update_link_field_indep_nvar<VelDustInterpolate, std::array<Tvec, 2>>(
                q,
                depends_list,
                resulting_event_list,
                vel_dust_face_xp.link_fields.get(id),
                shambase::get_check_ref(oriented_cell_graph.graph_links[oriented_cell_graph.xp]),
                ndust,
                ptr_cell0block_aabb_lower,
                ptr_block_cell_sizes,
                ptr_buf_vel_dust,
                ptr_buf_dx_vel_dust,
                ptr_buf_dy_vel_dust,
                ptr_buf_dz_vel_dust,
                dt_interp,
                ptr_buf_rho_dust);
            update_link_field_indep_nvar<VelDustInterpolate, std::array<Tvec, 2>>(
                q,
                depends_list,
                resulting_event_list,
                vel_dust_face_xm.link_fields.get(id),
                shambase::get_check_ref(oriented_cell_graph.graph_links[oriented_cell_graph.xm]),
                ndust,
                ptr_cell0block_aabb_lower,
                ptr_block_cell_sizes,
                ptr_buf_vel_dust,
                ptr_buf_dx_vel_dust,
                ptr_buf_dy_vel_dust,
                ptr_buf_dz_vel_dust,
                dt_interp,
                ptr_buf_rho_dust);
            update_link_field_indep_nvar<VelDustInterpolate, std::array<Tvec, 2>>(
                q,
                depends_list,
                resulting_event_list,
                vel_dust_face_yp.link_fields.get(id),
                shambase::get_check_ref(oriented_cell_graph.graph_links[oriented_cell_graph.yp]),
                ndust,
                ptr_cell0block_aabb_lower,
                ptr_block_cell_sizes,
                ptr_buf_vel_dust,
                ptr_buf_dx_vel_dust,
                ptr_buf_dy_vel_dust,
                ptr_buf_dz_vel_dust,
                dt_interp,
                ptr_buf_rho_dust);
            update_link_field_indep_nvar<VelDustInterpolate, std::array<Tvec, 2>>(
                q,
                depends_list,
                resulting_event_list,
                vel_dust_face_ym.link_fields.get(id),
                shambase::get_check_ref(oriented_cell_graph.graph_links[oriented_cell_graph.ym]),
                ndust,
                ptr_cell0block_aabb_lower,
                ptr_block_cell_sizes,
                ptr_buf_vel_dust,
                ptr_buf_dx_vel_dust,
                ptr_buf_dy_vel_dust,
                ptr_buf_dz_vel_dust,
                dt_interp,
                ptr_buf_rho_dust);
            update_link_field_indep_nvar<VelDustInterpolate, std::array<Tvec, 2>>(
                q,
                depends_list,
                resulting_event_list,
                vel_dust_face_zp.link_fields.get(id),
                shambase::get_check_ref(oriented_cell_graph.graph_links[oriented_cell_graph.zp]),
                ndust,
                ptr_cell0block_aabb_lower,
                ptr_block_cell_sizes,
                ptr_buf_vel_dust,
                ptr_buf_dx_vel_dust,
                ptr_buf_dy_vel_dust,
                ptr_buf_dz_vel_dust,
                dt_interp,
                ptr_buf_rho_dust);
            update_link_field_indep_nvar<VelDustInterpolate, std::array<Tvec, 2>>(
                q,
                depends_list,
                resulting_event_list,
                vel_dust_face_zm.link_fields.get(id),
                shambase::get_check_ref(oriented_cell_graph.graph_links[oriented_cell_graph.zm]),
                ndust,
                ptr_cell0block_aabb_lower,
                ptr_block_cell_sizes,
                ptr_buf_vel_dust,
                ptr_buf_dx_vel_dust,
                ptr_buf_dy_vel_dust,
                ptr_buf_dz_vel_dust,
                dt_interp,
                ptr_buf_rho_dust);

            block_cell_sizes.complete_event_state(resulting_event_list);
            cell0block_aabb_lower.complete_event_state(resulting_event_list);
            buf_vel_dust.complete_event_state(resulting_event_list);
            buf_dx_vel_dust.complete_event_state(resulting_event_list);
            buf_dy_vel_dust.complete_event_state(resulting_event_list);
            buf_dz_vel_dust.complete_event_state(resulting_event_list);
            buf_rho_dust.complete_event_state(resulting_event_list);
        });
}

template class shammodels::basegodunov::modules::FaceInterpolate<f64_3, i64_3>;
