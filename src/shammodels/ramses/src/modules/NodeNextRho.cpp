// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright (c) 2021-2025 Timothée David--Cléris <tim.shamrock@proton.me>
// SPDX-License-Identifier: CeCILL Free Software License Agreement v2.1
// Shamrock is licensed under the CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

/**
 * @file NodeNextRho.cpp
 * @author Léodasce Sewanou (leodasce.sewanou@ens-lyon.fr)
 * @author Timothée David--Cléris (tim.shamrock@proton.me) --no git blame--
 * @brief
 *
 */

#include "shambase/stacktrace.hpp"
#include "shambackends/kernel_call.hpp"
#include "shamcomm/logs.hpp"
#include "shammodels/common/amr/NeighGraph.hpp"
#include "shammodels/ramses/SolverConfig.hpp"
#include "shammodels/ramses/modules/NodeNextRho.hpp"
#include <type_traits>

namespace {

    using Direction = shammodels::basegodunov::modules::Direction;

    template<class Tvec, class TgridVec>
    class KernelNextRho {

        using Tscal            = sham::VecComponent<Tvec>;
        using TgridUint        = typename std::make_unsigned<sham::VecComponent<TgridVec>>::type;
        using OrientedAMRGraph = shammodels::basegodunov::modules::OrientedAMRGraph<Tvec, TgridVec>;
        using AMRGraph         = shammodels::basegodunov::modules::AMRGraph;
        using Edges = typename shammodels::basegodunov::modules::NodeNextRho<Tvec, TgridVec>::Edges;
        using Config   = shammodels::basegodunov::SolverConfig<Tvec, TgridVec>;
        using AMRBlock = typename Config::AMRBlock;

        public:
        inline static void kernel(Edges &edges, u32 block_size) {

            edges.cell_neigh_graph.graph.for_each([&](u64 id,
                                                      const OrientedAMRGraph &oriented_cell_graph) {
                AMRGraph &graph_neigh_xp
                    = shambase::get_check_ref(oriented_cell_graph.graph_links[Direction::xp]);
                AMRGraph &graph_neigh_xm
                    = shambase::get_check_ref(oriented_cell_graph.graph_links[Direction::xm]);
                AMRGraph &graph_neigh_yp
                    = shambase::get_check_ref(oriented_cell_graph.graph_links[Direction::yp]);
                AMRGraph &graph_neigh_ym
                    = shambase::get_check_ref(oriented_cell_graph.graph_links[Direction::ym]);
                AMRGraph &graph_neigh_zp
                    = shambase::get_check_ref(oriented_cell_graph.graph_links[Direction::zp]);
                AMRGraph &graph_neigh_zm
                    = shambase::get_check_ref(oriented_cell_graph.graph_links[Direction::zm]);

                u32 cell_count = (edges.sizes.indexes.get(id)) * block_size;

                sham::DeviceQueue &q = shamsys::instance::get_compute_scheduler().get_queue();

                auto &span_flux_rho_xp
                    = edges.flux_rho_face_xp.link_fields.get(id).link_graph_field;
                auto &span_flux_rho_xm
                    = edges.flux_rho_face_xm.link_fields.get(id).link_graph_field;
                auto &span_flux_rho_yp
                    = edges.flux_rho_face_yp.link_fields.get(id).link_graph_field;
                auto &span_flux_rho_ym
                    = edges.flux_rho_face_ym.link_fields.get(id).link_graph_field;
                auto &span_flux_rho_zp
                    = edges.flux_rho_face_zp.link_fields.get(id).link_graph_field;
                auto &span_flux_rho_zm
                    = edges.flux_rho_face_zm.link_fields.get(id).link_graph_field;

                auto &block_lower_span = edges.spans_cell0block_aabb_lower.get_spans().get(id);
                auto &cell_size_span   = edges.spans_block_cell_sizes.get_spans().get(id);

                auto &span_old_rho  = edges.spans_rho_old.get_spans().get(id);
                auto &span_next_rho = edges.spans_rho_next.get_spans().get(id);
                auto &dt_over2      = edges.dt_over2.value;

                logger::raw_ln("dt_over_2", dt_over2, "\n");

                sham::kernel_call(
                    q,
                    sham::MultiRef{
                        cell_size_span,
                        block_lower_span,
                        span_flux_rho_xp,
                        span_flux_rho_xm,
                        span_flux_rho_yp,
                        span_flux_rho_ym,
                        span_flux_rho_zp,
                        span_flux_rho_zm,
                        graph_neigh_xp,
                        graph_neigh_xm,
                        graph_neigh_yp,
                        graph_neigh_ym,
                        graph_neigh_zp,
                        graph_neigh_zm,
                        span_old_rho},
                    sham::MultiRef{span_next_rho},
                    cell_count,
                    [block_size, dt_over2](
                        i32 i,
                        const Tscal *__restrict cell_size,
                        const Tvec *__restrict aabb_lower,
                        const auto flux_rho_xp,
                        const auto flux_rho_xm,
                        const auto flux_rho_yp,
                        const auto flux_rho_ym,
                        const auto flux_rho_zp,
                        const auto flux_rho_zm,
                        const auto graph_iter_xp,
                        const auto graph_iter_xm,
                        const auto graph_iter_yp,
                        const auto graph_iter_ym,
                        const auto graph_iter_zp,
                        const auto graph_iter_zm,
                        const Tscal *__restrict old_rho,
                        Tscal *__restrict next_rho) {
                        /**/
                        auto get_cell_aabb = [=](u32 id) -> shammath::AABB<Tvec> {
                            const u32 cell_global_id = (u32) id;

                            const u32 block_id    = cell_global_id / block_size;
                            const u32 cell_loc_id = cell_global_id % block_size;

                            // fetch current block info
                            const Tvec cblock_min  = aabb_lower[block_id];
                            const Tscal delta_cell = cell_size[block_id];

                            std::array<u32, 3> lcoord_arr = AMRBlock::get_coord(cell_loc_id);
                            Tvec offset
                                = Tvec{lcoord_arr[0], lcoord_arr[1], lcoord_arr[2]} * delta_cell;

                            Tvec aabb_min = cblock_min + offset;
                            Tvec aabb_max = aabb_min + delta_cell;

                            return {aabb_min, aabb_max};
                        };

                        /**/
                        auto get_face_surface = [=](u32 id_a, u32 id_b) -> Tscal {
                            shammath::AABB<Tvec> aabb_cell_a = get_cell_aabb(id_a);
                            shammath::AABB<Tvec> aabb_cell_b = get_cell_aabb(id_b);

                            shammath::AABB<Tvec> face_aabb = aabb_cell_a.get_intersect(aabb_cell_b);

                            Tvec delta_face = face_aabb.delt();

                            delta_face.x() = (delta_face.x() == 0) ? 1 : delta_face.x();
                            delta_face.y() = (delta_face.y() == 0) ? 1 : delta_face.y();
                            delta_face.z() = (delta_face.z() == 0) ? 1 : delta_face.z();

                            return delta_face.x() * delta_face.y() * delta_face.z();
                        };

                        /**/
                        const u32 block_id    = i / block_size;
                        const u32 cell_loc_id = i % block_size;

                        Tscal V_i = cell_size[block_id];
                        V_i       = V_i * V_i * V_i;

                        Tscal dtrho = 0;

                        graph_iter_xp.for_each_object_link_id(i, [&](u32 id_b, u32 link_id) {
                            Tscal S_ij = get_face_surface(i, id_b);
                            // logger::raw("Sij = ", S_ij, "\n");
                            dtrho -= flux_rho_xp[id_b] * S_ij;
                        });

                        // logger::raw("dtrho = ", dtrho, "\n");
                        graph_iter_xm.for_each_object_link_id(i, [&](u32 id_b, u32 link_id) {
                            Tscal S_ij = get_face_surface(i, id_b);
                            dtrho -= flux_rho_xm[link_id] * S_ij;
                        });
                        graph_iter_yp.for_each_object_link_id(i, [&](u32 id_b, u32 link_id) {
                            Tscal S_ij = get_face_surface(i, id_b);
                            dtrho -= flux_rho_yp[link_id] * S_ij;
                        });
                        graph_iter_ym.for_each_object_link_id(i, [&](u32 id_b, u32 link_id) {
                            Tscal S_ij = get_face_surface(i, id_b);
                            dtrho -= flux_rho_ym[link_id] * S_ij;
                        });
                        graph_iter_zp.for_each_object_link_id(i, [&](u32 id_b, u32 link_id) {
                            Tscal S_ij = get_face_surface(i, id_b);
                            dtrho -= flux_rho_zp[link_id] * S_ij;
                        });
                        graph_iter_zm.for_each_object_link_id(i, [&](u32 id_b, u32 link_id) {
                            Tscal S_ij = get_face_surface(i, id_b);
                            dtrho -= flux_rho_zm[link_id] * S_ij;
                        });

                        dtrho /= V_i;

                        auto dt = 2 * dt_over2;

                        // for(int ii = 4990; ii < 5000; ii++){

                        //     logger::raw_ln("[",i,"] : ", dt, dtrho , old_rho[ii],"\n");
                        // }

                        next_rho[i] = (old_rho[i] + dt * dtrho);
                    });
            });
        }
    };

} // namespace

namespace shammodels::basegodunov::modules {

    template<class Tvec, class TgridVec>
    void NodeNextRho<Tvec, TgridVec>::_impl_evaluate_internal() {
        StackEntry stack_loc{};
        auto edges = get_edges();
        edges.spans_block_cell_sizes.check_sizes(edges.sizes.indexes);
        edges.spans_cell0block_aabb_lower.check_sizes(edges.sizes.indexes);
        edges.spans_rho_old.check_sizes(edges.sizes.indexes);
        edges.spans_rho_next.ensure_sizes(edges.sizes.indexes);

        if (false) {
            for (auto id = 0; id < 1; id++) {
                auto &buf = edges.spans_rho_next.get_buf(id);
                auto vec  = buf.copy_to_stdvec();
                logger::raw_ln(id, "next-rho-sz = ", "--", buf.get_size());
                for (int i = 0; i < buf.get_size(); i++) {
                    logger::raw_ln("rho-next-bf", i, vec[i]);
                }
            }
        }

        KernelNextRho<Tvec, TgridVec>::kernel(edges, block_size);

        if (true) {
            for (auto id = 0; id < 1; id++) {
                auto &buf_rho_next = edges.spans_rho_next.get_buf(id);
                auto &buf_rho_old  = edges.spans_rho_old.get_field(id).get_buf();
                auto vec_old       = buf_rho_old.copy_to_stdvec();
                auto vec_rho_next  = buf_rho_next.copy_to_stdvec();
                logger::raw_ln(id, "next-rho-sz = ", "--", buf_rho_old.get_size());
                for (int i = 0; i < buf_rho_old.get_size(); i++) {
                    logger::raw_ln("[", i, "] : ", vec_old[i], vec_rho_next[i], "\n");
                }
            }
        }
    }

    // template<class Tvec, class TgridVec>
    // void NodeNextRho<Tvec, TgridVec>::_impl_evaluate_internal() {
    //     StackEntry stack_loc{};
    //     auto edges                                                   = get_edges();

    //     edges.flux_rho_face_xp.check_size(edges.cell_neigh_graph.get_refs_dir(Direction::xp));

    //     //
    //     flux_rho_face_xm.resize_according_to(edges.cell_neigh_graph.get_refs_dir(Direction::xm));
    //     //
    //     flux_rho_face_yp.resize_according_to(edges.cell_neigh_graph.get_refs_dir(Direction::yp));
    //     //
    //     flux_rho_face_ym.resize_according_to(edges.cell_neigh_graph.get_refs_dir(Direction::ym));
    //     //
    //     flux_rho_face_zp.resize_according_to(edges.cell_neigh_graph.get_refs_dir(Direction::zp));
    //     //
    //     flux_rho_face_zm.resize_according_to(edges.cell_neigh_graph.get_refs_dir(Direction::zm));
    //     auto spans_block_cell_sizes      = edges.spans_block_cell_sizes.get_spans();
    //     auto spans_cell0block_aabb_lower = edges.spans_cell0block_aabb_lower.get_spans();
    //     edges.spans_rho_old.check_sizes(edges.sizes.indexes);
    //     edges.spans_rho_next_acc.ensure_sizes(edges.sizes.indexes);
    //     edges.spans_rho_next.ensure_sizes(edges.sizes.indexes);
    //     auto spans_rho_old = edges.spans_rho_old.get_spans();
    //     auto spans_rho_next_acc = edges.spans_rho_next_acc.get_spans();
    //     auto spans_rho_next = edges.spans_rho_next.get_spans();

    //     // using Funct_ = RhoNext<Tvec, TgridVec, AMRBlock>;
    //     // auto next_rho_funct
    //     //     = spans_block_cell_sizes.template map<Funct_>([&](u64 id, auto &csize) -> Funct_ {
    //     //           return {spans_cell0block_aabb_lower.get(id),
    //     spans_block_cell_sizes.get(id)};
    //     //       });

    //     auto graphs_xp = edges.cell_neigh_graph.get_refs_dir(Direction::xp);
    //     // auto graphs_xm = edges.cell_neigh_graph.get_refs_dir(Direction::xm);
    //     // auto graphs_yp = edges.cell_neigh_graph.get_refs_dir(Direction::yp);
    //     // auto graphs_ym = edges.cell_neigh_graph.get_refs_dir(Direction::ym);
    //     // auto graphs_zp = edges.cell_neigh_graph.get_refs_dir(Direction::zp);
    //     // auto graphs_zm = edges.cell_neigh_graph.get_refs_dir(Direction::zm);

    //     shambase::DistributedData<u32> counts_xp
    //         = graphs_xp.template map<u32>([&](u64 id, auto &graph) {
    //               return graph.get().link_count;
    //           });
    //     // shambase::DistributedData<u32> counts_xm
    //     //     = graphs_xm.template map<u32>([&](u64 id, auto &graph) {
    //     //           return graph.get().obj_cnt;
    //     //       });
    //     // shambase::DistributedData<u32> counts_yp
    //     //     = graphs_yp.template map<u32>([&](u64 id, auto &graph) {
    //     //           return graph.get().obj_cnt;
    //     //       });
    //     // shambase::DistributedData<u32> counts_ym
    //     //     = graphs_ym.template map<u32>([&](u64 id, auto &graph) {
    //     //           return graph.get().obj_cnt;
    //     //       });
    //     // shambase::DistributedData<u32> counts_zp
    //     //     = graphs_zp.template map<u32>([&](u64 id, auto &graph) {
    //     //           return graph.get().obj_cnt;
    //     //       });
    //     // shambase::DistributedData<u32> counts_zm
    //     //     = graphs_zm.template map<u32>([&](u64 id, auto &graph) {
    //     //           return graph.get().obj_cnt;
    //     //       });

    //     sham::distributed_data_kernel_call(
    //         shamsys::instance::get_compute_scheduler_ptr(),
    //         sham::DDMultiRef{edges.flux_rho_face_xp.link_fields},
    //         sham::DDMultiRef{spans_rho_next_acc},
    //         counts_xp,
    //         [](u32 id_a,

    //            const Tscal* rho_face_xp,

    //            Tscal *__restrict rho_n_acc) {
    //             logger::raw("dtrho = ", rho_face_xp[id_a],"\n");

    //             // link_iter.for_each_object_link_id(id_a, [&](u32 id_b, u32 link_id) {
    //             //     logger::raw("dtrho = ", rho_n_acc[id_a], acc_link_field[id_a],"\n");
    //             //     // rho_n_acc[id_a]
    //             //     //     = compute.get_link_field_val(id_a, id_b, acc_link_field[id_a]);
    //             //     // logger::raw("dtrho = ", rho_n_acc[id_a], acc_link_field[link_id],"\n");
    //             // });
    //         });
    // }

} // namespace shammodels::basegodunov::modules

template class shammodels::basegodunov::modules::NodeNextRho<f64_3, i64_3>;
