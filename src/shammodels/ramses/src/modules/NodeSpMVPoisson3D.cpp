// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright (c) 2021-2025 Timothée David--Cléris <tim.shamrock@proton.me>
// SPDX-License-Identifier: CeCILL Free Software License Agreement v2.1
// Shamrock is licensed under the CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

/**
 * @file NodeSpMVPoisson3D.cpp
 * @author Léodasce Sewanou (leodasce.sewanou@ens-lyon.fr)
 * @author Timothée David--Cléris (tim.shamrock@proton.me) --no git blame--
 * @brief Implementation of matrix-vector product [A*p] for the conjugate gradient solver.
 *
 */
#include "shambase/string.hpp"
#include "shambackends/EventList.hpp"
#include "shambackends/kernel_call.hpp"
#include "shambackends/sycl_utils.hpp"
#include "shambackends/typeAliasVec.hpp"
#include "shambackends/vec.hpp"
#include "shamcomm/logs.hpp"
#include "shammodels/common/amr/NeighGraph.hpp"
#include "shammodels/ramses/modules/CGLaplacianStencil.hpp"
#include "shammodels/ramses/modules/NodeSpMVPoisson3D.hpp"
#include "shamsys/NodeInstance.hpp"
#include <shambackends/sycl.hpp>
#include <type_traits>

namespace {

    template<class Tvec, class TgridVec>
    class _Kernel {
        using Tscal     = shambase::VecComponent<Tvec>;
        using TgridUint = typename std::make_unsigned<shambase::VecComponent<TgridVec>>::type;
        using OrientedAMRGraph = shammodels::basegodunov::modules::OrientedAMRGraph<Tvec, TgridVec>;
        using AMRGraph         = shammodels::basegodunov::modules::AMRGraph;
        using Edges =
            typename shammodels::basegodunov::modules::NodeSpMVPoisson3D<Tvec, TgridVec>::Edges;

        public:
        inline static void kernel(Edges &edges, u32 block_size) {
            edges.cell_neigh_graph.graph.for_each(
                [&](u64 id, const OrientedAMRGraph &oriented_cell_graph) {
                    auto &cell_sizes_span = edges.spans_block_cell_sizes.get_spans().get(id);
                    auto &in_span         = edges.spans_in.get_spans().get(id);
                    auto &out_span        = edges.spans_out.get_spans().get(id);

                    AMRGraph &graph_neigh_xp = shambase::get_check_ref(
                        oriented_cell_graph.graph_links[shammodels::basegodunov::Direction::xp]);
                    AMRGraph &graph_neigh_xm = shambase::get_check_ref(
                        oriented_cell_graph.graph_links[shammodels::basegodunov::Direction::xm]);
                    AMRGraph &graph_neigh_yp = shambase::get_check_ref(
                        oriented_cell_graph.graph_links[shammodels::basegodunov::Direction::yp]);
                    AMRGraph &graph_neigh_ym = shambase::get_check_ref(
                        oriented_cell_graph.graph_links[shammodels::basegodunov::Direction::ym]);
                    AMRGraph &graph_neigh_zp = shambase::get_check_ref(
                        oriented_cell_graph.graph_links[shammodels::basegodunov::Direction::zp]);
                    AMRGraph &graph_neigh_zm = shambase::get_check_ref(
                        oriented_cell_graph.graph_links[shammodels::basegodunov::Direction::zm]);
                    u32 cell_count = (edges.sizes.indexes.get(id)) * block_size;

                    sham::DeviceQueue &q = shamsys::instance::get_compute_scheduler().get_queue();

                    sham::kernel_call(
                        q,
                        sham::MultiRef{
                            cell_sizes_span,
                            in_span,
                            graph_neigh_xp,
                            graph_neigh_xm,
                            graph_neigh_yp,
                            graph_neigh_ym,
                            graph_neigh_zp,
                            graph_neigh_zm},
                        sham::MultiRef{out_span},
                        cell_count,
                        [block_size](
                            i32 cell_global_id,
                            const Tscal *__restrict cell_sizes,
                            const Tscal *__restrict in,
                            const auto graph_iter_xp,
                            const auto graph_iter_xm,
                            const auto graph_iter_yp,
                            const auto graph_iter_ym,
                            const auto graph_iter_zp,
                            const auto graph_iter_zm,
                            Tscal *__restrict out) {
                            const u32 block_id    = cell_global_id / block_size;
                            const u32 cell_loc_id = cell_global_id % block_size;
                            Tscal delta_cell      = cell_sizes[block_id];
                            auto Ap_id = shammodels::basegodunov::laplacian_7pt<Tscal, Tvec>(
                                cell_global_id,
                                delta_cell,
                                graph_iter_xp,
                                graph_iter_xm,
                                graph_iter_yp,
                                graph_iter_ym,
                                graph_iter_zp,
                                graph_iter_zm,
                                [=](u32 id) {
                                    return in[id];
                                });

                            out[cell_global_id] = Ap_id;

                            // if(cell_global_id % 25 == 0)
                            // {
                            //     logger::raw_ln("id_a = [ ", cell_global_id, " ] : ",
                            //     in[cell_global_id], " --- ",out[cell_global_id], "  ","\n");
                            // }

                            // if( (cell_global_id % 150) == 0)
                            // {
                            //     logger::raw_ln("id_a = [ ", cell_global_id, " ] : ",
                            //     out[cell_global_id], "\n");
                            // }
                        });
                });
        }
    };

} // namespace

namespace shammodels::basegodunov::modules {
    template<class Tvec, class TgridVec>
    void NodeSpMVPoisson3D<Tvec, TgridVec>::_impl_evaluate_internal() {
        StackEntry stack_loc{};
        auto edges = get_edges();
        // logger::raw_ln("SPMV:[p, Ap] \t", &edges.spans_in, "-", &edges.spans_out,"\n");
        edges.spans_block_cell_sizes.check_sizes(edges.sizes.indexes);
        edges.spans_in.check_sizes(edges.sizes.indexes);
        edges.spans_out.ensure_sizes(edges.sizes.indexes);

        _Kernel<Tvec, TgridVec>::kernel(edges, block_size);
    }

    template<class Tvec, class TgridVec>
    std::string NodeSpMVPoisson3D<Tvec, TgridVec>::_impl_get_tex() {

        std::string span_in  = get_ro_edge_base(3).get_tex_symbol();
        std::string span_out = get_rw_edge_base(0).get_tex_symbol();

        std::string tex = R"tex(
           SpMV kernel
            \begin{equation}
            \mathbf{result} = \mathbf{A}\mathbf{in}
            \end{equation}
        )tex";

        shambase::replace_all(tex, "{result}", span_out);
        shambase::replace_all(tex, "{in}", span_in);

        return tex;
    }

} // namespace shammodels::basegodunov::modules

template class shammodels::basegodunov::modules::NodeSpMVPoisson3D<f64_3, i64_3>;
