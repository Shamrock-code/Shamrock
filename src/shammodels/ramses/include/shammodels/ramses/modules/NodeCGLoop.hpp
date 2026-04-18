// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright (c) 2021-2026 Timothée David--Cléris <tim.shamrock@proton.me>
// SPDX-License-Identifier: CeCILL Free Software License Agreement v2.1
// Shamrock is licensed under the CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

#pragma once

/**
 * @file NodeCGLoop.hpp
 * @author Léodasce Sewanou (leodasce.sewanou@ens-lyon.fr)
 * @author Timothée David--Cléris (tim.shamrock@proton.me) --no git blame--
 * @brief
 *
 */

#include "shambase/aliases_int.hpp"
#include "shambackends/vec.hpp"
#include "shamcomm/logs.hpp"
#include "shammodels/ramses/SolverConfig.hpp"
#include "shammodels/ramses/modules/CGInit.hpp"
#include "shammodels/ramses/modules/NodeAXPY.hpp"
#include "shammodels/ramses/modules/NodeAYPX.hpp"
#include "shammodels/ramses/modules/NodeHadamardProd.hpp"
#include "shammodels/ramses/modules/NodeSpMVPoisson3D.hpp"
#include "shammodels/ramses/modules/NodeSumReduction.hpp"
#include "shammodels/ramses/modules/ResidualDot.hpp"
#include "shammodels/ramses/modules/SolverStorage.hpp"
#include "shammodels/ramses/solvegraph/OrientedAMRGraphEdge.hpp"
#include "shamrock/scheduler/ShamrockCtx.hpp"
#include "shamrock/solvergraph/CopyPatchDataField.hpp"
#include "shamrock/solvergraph/DDSharedBuffers.hpp"
#include "shamrock/solvergraph/ExchangeGhostField.hpp"
#include "shamrock/solvergraph/ExtractGhostField.hpp"
#include "shamrock/solvergraph/Field.hpp"
#include "shamrock/solvergraph/FieldRefs.hpp"
#include "shamrock/solvergraph/INode.hpp"
#include "shamrock/solvergraph/Indexes.hpp"
#include "shamrock/solvergraph/PatchDataFieldDDShared.hpp"
#include "shamrock/solvergraph/ReplaceGhostField.hpp"
#include "shamrock/solvergraph/ScalarEdge.hpp"
#include "shamrock/solvergraph/ScalarsEdge.hpp"
#include <memory>

// #define NODE_CG_LOOP_EDGES(X_RO, X_RW) \
//     /* inputs */ \
//     X_RO(shamrock::solvergraph::Indexes<u32>, sizes) \
//     X_RO(shamrock::solvergraph::Indexes<u32>, sizes_no_gz) \
//     X_RO(CellGraphEdge, cell_neigh_graph) \
//     X_RO(shamrock::solvergraph::Field<Tscal>, spans_block_cell_sizes) \
//     X_RO(shamrock::solvergraph::IFieldRefs<Tscal>, spans_rho) \
//     X_RO(shamrock::solvergraph::ScalarEdge<Tscal>, mean_rho) \
//     X_RO(shamrock::solvergraph::DDSharedBuffers<u32>, idx_in_ghost) \
//     X_RO(shamrock::solvergraph::ScalarsEdge<u32>, rank_owner) \
//     X_RO(shamrock::solvergraph::ScalarEdge<Tscal>, dt) \
//     /* outputs */ \
//     X_RW(shamrock::solvergraph::IFieldRefs<Tscal>, spans_phi) \
//     X_RW(shamrock::solvergraph::Field<Tscal>, spans_phi_cpy) \
//     X_RW(shamrock::solvergraph::Field<Tscal>, spans_phi_res) \
//     X_RW(shamrock::solvergraph::Field<Tscal>, spans_phi_p) \
//     X_RW(shamrock::solvergraph::Field<Tscal>, spans_phi_Ap) \
//     X_RW(shamrock::solvergraph::Field<Tscal>, spans_phi_hadamard_prod) \
//     X_RW(shamrock::solvergraph::Field<Tscal>, spans_phi_hadamard_prod_cpy) \
//     X_RW(shamrock::solvergraph::ScalarEdge<Tscal>, old_values) \
//     X_RW(shamrock::solvergraph::ScalarEdge<Tscal>, new_values) \
//     X_RW(shamrock::solvergraph::ScalarEdge<Tscal>, e_norm) \
//     X_RW(shamrock::solvergraph::ScalarEdge<Tscal>, alpha) \
//     X_RW(shamrock::solvergraph::ScalarEdge<Tscal>, beta)

namespace shammodels::basegodunov::modules {
    template<class Tvec, class TgridVec>
    class NodeCGLoop : public shamrock::solvergraph::INode {
        using Tscal    = shambase::VecComponent<Tvec>;
        using AMRBlock = typename shammodels::basegodunov::SolverConfig<Tvec, TgridVec>::AMRBlock;
        using CellGraphEdge = solvergraph::OrientedAMRGraphEdge<Tvec, TgridVec>;

        u32 block_size;
        Tscal fourPiG;
        u32 Niter_max;
        Tscal tol;

        public:
        NodeCGLoop(u32 block_size, Tscal fourPiG, u32 Niter_max, Tscal tol)
            : block_size(block_size), fourPiG(fourPiG), Niter_max(Niter_max), tol(tol) {}

        // init node
        modules::CGInit<Tvec, TgridVec> cg_init_node{block_size, fourPiG};
        // ddot node old
        modules::ResidualDot<Tscal> res_ddot_node{block_size};
        // SpMV node
        modules::NodeSpMVPoisson3D<Tvec, TgridVec> spmv_node{block_size};
        // hadamardProd node
        modules::NodeHadamardProd<Tscal> hadamard_prod_node{block_size};
        // A-norm node
        modules::NodeSumReduction<Tscal> a_norm_node{block_size};
        // New-phi node
        modules::NodeAXPY<Tscal> new_potential_node{block_size};
        // New-residual node
        modules::NodeAXPY<Tscal> new_residual_node{block_size};
        // ddot node new
        modules::ResidualDot<Tscal> res_ddot_new_node{block_size};
        // New-A-conjugate vector p node
        modules::NodeAYPX<Tscal> new_p_node{block_size};

        //
        std::shared_ptr<shamrock::solvergraph::PatchDataFieldDDShared<Tscal>> p_ghosts
            = std::make_shared<shamrock::solvergraph::PatchDataFieldDDShared<Tscal>>(
                "p_ghots", "p_ghots");

        /***********************************/
        // Extract ghosts for Field
        shamrock::solvergraph::ExtractGhostField<Tscal> node_gz_p{};

        // Exchange ghosts for field
        shamrock::solvergraph::ExchangeGhostField<Tscal> node_exch_gz_p{};

        // Replace ghosts for field
        shamrock::solvergraph::ReplaceGhostField<Tscal> node_replace_gz_p{};

        struct Edges {
            const shamrock::solvergraph::Indexes<u32> &sizes;
            const shamrock::solvergraph::Indexes<u32> &sizes_no_gz;
            const solvergraph::OrientedAMRGraphEdge<Tvec, TgridVec> &cell_neigh_graph;
            const shamrock::solvergraph::Field<Tscal> &spans_block_cell_sizes;
            const shamrock::solvergraph::IFieldRefs<Tscal> &spans_rho;
            const shamrock::solvergraph::ScalarEdge<Tscal> &mean_rho;
            const shamrock::solvergraph::DDSharedBuffers<u32> &idx_in_ghost;
            const shamrock::solvergraph::RankGetter &rank_owner;
            shamrock::solvergraph::IFieldRefs<Tscal> &spans_phi;
            shamrock::solvergraph::Field<Tscal> &spans_phi_res;
            shamrock::solvergraph::Field<Tscal> &spans_phi_p;
            shamrock::solvergraph::Field<Tscal> &spans_phi_Ap;
            shamrock::solvergraph::Field<Tscal> &spans_phi_hadamard_prod;
            shamrock::solvergraph::ScalarEdge<Tscal> &old_values;
            shamrock::solvergraph::ScalarEdge<Tscal> &new_values;
            shamrock::solvergraph::ScalarEdge<Tscal> &e_norm;
            shamrock::solvergraph::ScalarEdge<Tscal> &alpha;
            shamrock::solvergraph::ScalarEdge<Tscal> &beta;
        };

        inline void set_edges(
            std::shared_ptr<shamrock::solvergraph::Indexes<u32>> sizes,
            std::shared_ptr<shamrock::solvergraph::Indexes<u32>> sizes_no_gz,
            std::shared_ptr<solvergraph::OrientedAMRGraphEdge<Tvec, TgridVec>> cell_neigh_graph,
            std::shared_ptr<shamrock::solvergraph::Field<Tscal>> spans_block_cell_sizes,
            std::shared_ptr<shamrock::solvergraph::IFieldRefs<Tscal>> spans_rho,
            std::shared_ptr<shamrock::solvergraph::ScalarEdge<Tscal>> mean_rho,
            std::shared_ptr<shamrock::solvergraph::DDSharedBuffers<u32>> idx_in_ghost,
            std::shared_ptr<shamrock::solvergraph::RankGetter> rank_owner,
            std::shared_ptr<shamrock::solvergraph::IFieldRefs<Tscal>> spans_phi,
            std::shared_ptr<shamrock::solvergraph::Field<Tscal>> spans_phi_res,
            std::shared_ptr<shamrock::solvergraph::Field<Tscal>> spans_phi_p,
            std::shared_ptr<shamrock::solvergraph::Field<Tscal>> spans_phi_Ap,
            std::shared_ptr<shamrock::solvergraph::Field<Tscal>> spans_phi_hadamard_prod,
            std::shared_ptr<shamrock::solvergraph::ScalarEdge<Tscal>> old_values,
            std::shared_ptr<shamrock::solvergraph::ScalarEdge<Tscal>> new_values,
            std::shared_ptr<shamrock::solvergraph::ScalarEdge<Tscal>> e_norm,
            std::shared_ptr<shamrock::solvergraph::ScalarEdge<Tscal>> alpha,
            std::shared_ptr<shamrock::solvergraph::ScalarEdge<Tscal>> beta

        ) {
            __internal_set_ro_edges(
                {sizes,
                 sizes_no_gz,
                 cell_neigh_graph,
                 spans_block_cell_sizes,
                 spans_rho,
                 mean_rho,
                 idx_in_ghost,
                 rank_owner});

            __internal_set_rw_edges(
                {spans_phi,
                 spans_phi_res,
                 spans_phi_p,
                 spans_phi_Ap,
                 spans_phi_hadamard_prod,
                 old_values,
                 new_values,
                 e_norm,
                 alpha,
                 beta});

            // set cg_init_node edges
            cg_init_node.set_edges(
                sizes,
                cell_neigh_graph,
                spans_block_cell_sizes,
                spans_phi,
                spans_rho,
                mean_rho,
                spans_phi_res,
                spans_phi_p);

            // set res_ddot_node edges
            res_ddot_node.set_edges(sizes_no_gz, spans_phi_res, old_values);

            // set spmv_node edges
            spmv_node.set_edges(
                sizes, cell_neigh_graph, spans_block_cell_sizes, spans_phi_p, spans_phi_Ap);

            // set hadamard_prod_node edges
            hadamard_prod_node.set_edges(sizes, spans_phi_p, spans_phi_Ap, spans_phi_hadamard_prod);

            // set a_norm_node edges
            a_norm_node.set_edges(sizes_no_gz, spans_phi_hadamard_prod, e_norm);

            // set new_potential_node edges
            new_potential_node.set_edges(sizes, spans_phi_p, alpha, spans_phi);

            // set new_residual_node edges
            new_residual_node.set_edges(sizes, spans_phi_Ap, alpha, spans_phi_res);

            // // set res_ddot_new_node edges
            res_ddot_new_node.set_edges(sizes_no_gz, spans_phi_res, new_values);

            // // set new_p_node edges
            new_p_node.set_edges(sizes, spans_phi_res, beta, spans_phi_p);

            // set node_gz edges  for p-vectors
            node_gz_p.set_edges(spans_phi_p, idx_in_ghost, p_ghosts);

            // set node_exch_gz edges for p-vectors
            node_exch_gz_p.set_edges(rank_owner, p_ghosts);

            // replace ghosts for p-vectors
            node_replace_gz_p.set_edges(p_ghosts, spans_phi_p);
        }

        inline Edges get_edges() {
            return Edges{
                get_ro_edge<shamrock::solvergraph::Indexes<u32>>(0),
                get_ro_edge<shamrock::solvergraph::Indexes<u32>>(1),
                get_ro_edge<solvergraph::OrientedAMRGraphEdge<Tvec, TgridVec>>(2),
                get_ro_edge<shamrock::solvergraph::Field<Tscal>>(3),
                get_ro_edge<shamrock::solvergraph::IFieldRefs<Tscal>>(4),
                get_ro_edge<shamrock::solvergraph::ScalarEdge<Tscal>>(5),
                get_ro_edge<shamrock::solvergraph::DDSharedBuffers<u32>>(6),
                get_ro_edge<shamrock::solvergraph::RankGetter>(7),
                get_rw_edge<shamrock::solvergraph::IFieldRefs<Tscal>>(0),
                get_rw_edge<shamrock::solvergraph::Field<Tscal>>(1),
                get_rw_edge<shamrock::solvergraph::Field<Tscal>>(2),
                get_rw_edge<shamrock::solvergraph::Field<Tscal>>(3),
                get_rw_edge<shamrock::solvergraph::Field<Tscal>>(4),
                get_rw_edge<shamrock::solvergraph::ScalarEdge<Tscal>>(5),
                get_rw_edge<shamrock::solvergraph::ScalarEdge<Tscal>>(6),
                get_rw_edge<shamrock::solvergraph::ScalarEdge<Tscal>>(7),
                get_rw_edge<shamrock::solvergraph::ScalarEdge<Tscal>>(8),
                get_rw_edge<shamrock::solvergraph::ScalarEdge<Tscal>>(9)
                //
            };
        }

        void _impl_evaluate_internal();

        inline virtual std::string _impl_get_label() const { return "CGMainLoop"; };

        virtual std::string _impl_get_tex() const { return "TODO"; };
    };

} // namespace shammodels::basegodunov::modules
