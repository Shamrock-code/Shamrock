// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright (c) 2021-2025 Timothée David--Cléris <tim.shamrock@proton.me>
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
#include "shammodels/ramses/SolverConfig.hpp"
#include "shammodels/ramses/modules/CGInit.hpp"
#include "shammodels/ramses/modules/NodeAXPYThreeVectors.hpp"
#include "shammodels/ramses/modules/NodeAXPYTwoVectors.hpp"
#include "shammodels/ramses/modules/NodeAYPXTwoVectors.hpp"
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
#include "shamrock/solvergraph/ReplaceGhostFields.hpp"
#include "shamrock/solvergraph/ScalarEdge.hpp"
#include "shamrock/solvergraph/ScalarsEdge.hpp"
#include <memory>

namespace shammodels::basegodunov::modules {
    template<class Tvec, class TgridVec>
    class NodeCGLoop : public shamrock::solvergraph::INode {
        using Tscal    = shambase::VecComponent<Tvec>;
        using AMRBlock = typename shammodels::basegodunov::SolverConfig<Tvec, TgridVec>::AMRBlock;

        u32 block_size;
        Tscal fourPiG;
        u32 Niter_max;
        Tscal tol;

        public:
        NodeCGLoop(u32 block_size, Tscal fourPiG, u32 Niter_max, Tscal tol)
            : block_size(block_size), fourPiG(fourPiG), Niter_max(Niter_max), tol(tol) {}

        // init node
        modules::CGInit<Tvec, TgridVec> node0{block_size, fourPiG};
        // ddot node old
        modules::ResidualDot<Tscal> node1{};
        // SpMV node
        modules::NodeSpMVPoisson3D<Tvec, TgridVec> node2{block_size};
        // hadamardProd node
        modules::NodeHadamardProd<Tscal> node3{block_size};
        // A-norm node
        modules::NodeSumReduction<Tscal> node4{};
        // New-phi node
        modules::NodeAXPYTwoVectors<Tscal> node5{block_size};
        // New-residual node
        modules::NodeAXPYTwoVectors<Tscal> node6{block_size};
        // ddot node new
        modules::ResidualDot<Tscal> node7{};
        // New-A-conjugate vector p node
        modules::NodeAYPXTwoVectors<Tscal> node8{block_size};

        //
        std::shared_ptr<shamrock::solvergraph::PatchDataFieldDDShared<Tscal>> p_ghosts
            = std::make_shared<shamrock::solvergraph::PatchDataFieldDDShared<Tscal>>(
                "p_ghots", "p_ghots");

        std::shared_ptr<shamrock::solvergraph::PatchDataFieldDDShared<Tscal>> res_ghosts
            = std::make_shared<shamrock::solvergraph::PatchDataFieldDDShared<Tscal>>(
                "res_ghots", "res_ghots");

        // Extract ghosts for Field
        shamrock::solvergraph::ExtractGhostField<Tscal> node_gz{block_size};

        // Exchange ghosts for field
        shamrock::solvergraph::ExchangeGhostField<Tscal> node_exch_gz{};

        // Replace ghosts for field
        shamrock::solvergraph::ReplaceGhostFields<Tscal> node_replace_gz{block_size};

        // Copy original phi_p back after modif node
        shamrock::solvergraph::CopyPatchDataField<Tscal> node_cp_phi_p_back{};

        // Extract ghosts for field (residuals)
        shamrock::solvergraph::ExtractGhostField<Tscal> node_gz_res{block_size};

        // Exchange ghosts for field  (residuals)
        shamrock::solvergraph::ExchangeGhostField<Tscal> node_exch_gz_res{};

        // Replace ghosts for field (residuals)
        shamrock::solvergraph::ReplaceGhostFields<Tscal> node_replace_gz_res{block_size};

        struct Edges {
            const shamrock::solvergraph::Indexes<u32> &sizes;
            const solvergraph::OrientedAMRGraphEdge<Tvec, TgridVec> &cell_neigh_graph;
            const shamrock::solvergraph::Field<Tscal> &spans_block_cell_sizes;
            const shamrock::solvergraph::FieldRefs<Tscal> &spans_rho;
            const shamrock::solvergraph::ScalarEdge<Tscal> &mean_rho;
            const shamrock::solvergraph::DDSharedBuffers<u32> &idx_in_ghost;
            const shamrock::solvergraph::ScalarsEdge<u32> &rank_owner;
            shamrock::solvergraph::FieldRefs<Tscal> &spans_phi;
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
            std::shared_ptr<solvergraph::OrientedAMRGraphEdge<Tvec, TgridVec>> cell_neigh_graph,
            std::shared_ptr<shamrock::solvergraph::Field<Tscal>> spans_block_cell_sizes,
            std::shared_ptr<shamrock::solvergraph::FieldRefs<Tscal>> spans_rho,
            std::shared_ptr<shamrock::solvergraph::ScalarEdge<Tscal>> mean_rho,
            std::shared_ptr<shamrock::solvergraph::DDSharedBuffers<u32>> idx_in_ghost,
            std::shared_ptr<shamrock::solvergraph::ScalarsEdge<u32>> rank_owner,
            std::shared_ptr<shamrock::solvergraph::FieldRefs<Tscal>> spans_phi,
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
                 beta

                });

            // set node0 edges
            node0.set_edges(
                sizes,
                cell_neigh_graph,
                spans_block_cell_sizes,
                spans_phi,
                spans_rho,
                mean_rho,
                spans_phi_res,
                spans_phi_p);

            // set node1 edges
            node1.set_edges(spans_phi_res, old_values);

            // set node2 edges
            node2.set_edges(
                sizes, cell_neigh_graph, spans_block_cell_sizes, spans_phi_p, spans_phi_Ap);

            // set node3 edges
            node3.set_edges(sizes, spans_phi_p, spans_phi_Ap, spans_phi_hadamard_prod);

            // set node4 edges
            node4.set_edges(sizes, spans_phi_hadamard_prod, e_norm);

            // set node5 edges
            node5.set_edges(sizes, spans_phi_p, alpha, spans_phi);

            // set node6 edges
            node6.set_edges(sizes, spans_phi_Ap, alpha, spans_phi_res);

            // set node7 edges
            node7.set_edges(spans_phi_res, new_values);

            // set node8 edges
            node8.set_edges(sizes, spans_phi_res, beta, spans_phi_p);

            // set node_gz edges  for p-vectors
            node_gz.set_edges(spans_phi_p, idx_in_ghost, p_ghosts);

            // set node_exch_gz edges for p-vectors
            node_exch_gz.set_edges(rank_owner, p_ghosts);

            // replace ghosts for p-vectors
            node_replace_gz.set_edges(p_ghosts, spans_phi_p);

            // set node_gz edges for res-vectors
            node_gz_res.set_edges(spans_phi_res, idx_in_ghost, res_ghosts);

            // set node_exch_gz edges for res-vectors
            node_exch_gz_res.set_edges(rank_owner, res_ghosts);

            // replace ghosts for res-vectors
            node_replace_gz_res.set_edges(res_ghosts, spans_phi_res);
        }

        inline Edges get_edges() {
            return Edges{
                get_ro_edge<shamrock::solvergraph::Indexes<u32>>(0),
                get_ro_edge<solvergraph::OrientedAMRGraphEdge<Tvec, TgridVec>>(1),
                get_ro_edge<shamrock::solvergraph::Field<Tscal>>(2),
                get_ro_edge<shamrock::solvergraph::FieldRefs<Tscal>>(3),
                get_ro_edge<shamrock::solvergraph::ScalarEdge<Tscal>>(4),
                get_ro_edge<shamrock::solvergraph::DDSharedBuffers<u32>>(5),
                get_ro_edge<shamrock::solvergraph::ScalarsEdge<u32>>(6),
                get_rw_edge<shamrock::solvergraph::FieldRefs<Tscal>>(0),
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

        inline virtual std::string _impl_get_label() { return "CGMainLoop"; };

        virtual std::string _impl_get_tex();
    };

} // namespace shammodels::basegodunov::modules
