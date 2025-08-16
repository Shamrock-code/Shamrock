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
 * @file NodeBICGSTABLoop.hpp
 * @author Léodasce Sewanou (leodasce.sewanou@ens-lyon.fr)
 * @author Timothée David--Cléris (tim.shamrock@proton.me) --no git blame--
 * @brief
 *
 */

#include "shambase/aliases_int.hpp"
#include "shambackends/vec.hpp"
#include "shammodels/ramses/SolverConfig.hpp"
#include "shammodels/ramses/modules/BICGSTABInit.hpp"
#include "shammodels/ramses/modules/NodeAXPYThreeVectors.hpp"
#include "shammodels/ramses/modules/NodeAXPYTwoVectors.hpp"
#include "shammodels/ramses/modules/NodeAYPXTwoVectors.hpp"
#include "shammodels/ramses/modules/NodeHadamardProd.hpp"
#include "shammodels/ramses/modules/NodeLinCombThreeVectors.hpp"
#include "shammodels/ramses/modules/NodeOverwrite.hpp"
#include "shammodels/ramses/modules/NodeSpMVPoisson3D.hpp"
#include "shammodels/ramses/modules/NodeSumReduction.hpp"
#include "shammodels/ramses/modules/ResidualDot.hpp"
#include "shammodels/ramses/modules/SolverStorage.hpp"
#include "shammodels/ramses/solvegraph/OrientedAMRGraphEdge.hpp"
#include "shamrock/scheduler/ShamrockCtx.hpp"
#include "shamrock/solvergraph/Field.hpp"
#include "shamrock/solvergraph/FieldRefs.hpp"
#include "shamrock/solvergraph/INode.hpp"
#include "shamrock/solvergraph/Indexes.hpp"
#include "shamrock/solvergraph/ScalarEdge.hpp"
#include <memory>

namespace shammodels::basegodunov::modules {
    template<class Tvec, class TgridVec>
    class NodeBICGSTABLoop : public shamrock::solvergraph::INode {
        using Tscal    = shambase::VecComponent<Tvec>;
        using AMRBlock = typename shammodels::basegodunov::SolverConfig<Tvec, TgridVec>::AMRBlock;
        using Config   = SolverConfig<Tvec, TgridVec>;
        /// Alias to the SolverStorage type
        using Storage = SolverStorage<Tvec, TgridVec, u64>;
        /// Reference to the Shamrock context
        ShamrockCtx &context;
        /// Reference to the configuration of the solver
        Config &solver_config;
        /// Reference to the storage of the solver
        Storage &storage;

        u32 block_size;
        Tscal fourPiG;
        u32 Niter_max;
        Tscal tol_cvg;
        Tscal tol_happy_bk;

        public:
        NodeBICGSTABLoop(
            ShamrockCtx &context,
            Config &solver_config,
            Storage &storage,
            u32 block_size,
            Tscal fourPiG,
            u32 Niter_max,
            Tscal tol_cvg,
            Tscal tol_happy_bk)
            : context(context), solver_config(solver_config), storage(storage),
              block_size(block_size), fourPiG(fourPiG), Niter_max(Niter_max), tol_cvg(tol_cvg),
              tol_happy_bk(tol_happy_bk) {}

        // init node
        modules::BICGSTABInit<Tvec, TgridVec> node0_1{block_size, fourPiG};
        // hadamardProd node for dot product <r_0, r'_0>
        modules::NodeHadamardProd<Tscal> node0_2{block_size};
        // dotprod node <r_0, z_0>
        modules::NodeSumReduction<Tscal> node0_3{};

        // SpMV  Ap node
        modules::NodeSpMVPoisson3D<Tvec, TgridVec> node1_1{block_size};
        // hadamardProd node Ap x r'0
        modules::NodeHadamardProd<Tscal> node1_2{block_size};
        // dotprod node <Ap, r'_0>
        modules::NodeSumReduction<Tscal> node1_3{};

        // s_k node
        modules::NodeLinCombThreeVectors<Tscal> node2_1{block_size};
        // ddot <s_k,s_k> node
        modules::ResidualDot<Tscal> node2_2{};
        // phi'_{k} = phi_{k} + alpha_k p_k
        modules::NodeAXPYTwoVectors<Tscal> node2_3{block_size};

        // SpMV  As node
        modules::NodeSpMVPoisson3D<Tvec, TgridVec> node3_1{block_size};
        // hadamardProd node As x s
        modules::NodeHadamardProd<Tscal> node3_2{block_size};
        // dotprod node <As, s>
        modules::NodeSumReduction<Tscal> node3_3{};
        // ddot <As_{k},As_{k}> node
        modules::ResidualDot<Tscal> node3_4{};

        // New-phi node
        modules::NodeAXPYThreeVectors<Tscal> node4{block_size};

        // New-residual node
        modules::NodeLinCombThreeVectors<Tscal> node5_1{block_size};
        // ddot <r_{k+1},r_{k+1}> node;
        modules::ResidualDot<Tscal> node5_2{};

        // hadamardProd node r_{k+1} x r'_0
        modules::NodeHadamardProd<Tscal> node6_1{block_size};
        // dotprod node < r_{k+1}, r'_0>
        modules::NodeSumReduction<Tscal> node6_2{};

        // New-A-conjugate vector p node
        modules::NodeLinCombThreeVectors<Tscal> node7{block_size};

        // r'0 = r_{k+1} node
        modules::NodeOverwrite<Tscal> node8_1{block_size};
        // p_{k+1} = r_{k+1} node
        modules::NodeOverwrite<Tscal> node8_2{block_size};

        struct Edges {
            const shamrock::solvergraph::Indexes<u32> &sizes;
            const solvergraph::OrientedAMRGraphEdge<Tvec, TgridVec> &cell_neigh_graph;
            const shamrock::solvergraph::Field<Tscal> &spans_block_cell_sizes;
            const shamrock::solvergraph::FieldRefs<Tscal> &spans_rho;
            const shamrock::solvergraph::ScalarEdge<Tscal> &mean_rho;
            shamrock::solvergraph::FieldRefs<Tscal> &spans_phi;
            shamrock::solvergraph::Field<Tscal> &spans_phi_res;
            shamrock::solvergraph::Field<Tscal> &spans_phi_res_bis;
            shamrock::solvergraph::Field<Tscal> &spans_phi_p;
            shamrock::solvergraph::Field<Tscal> &spans_phi_Ap;
            shamrock::solvergraph::Field<Tscal> &spans_phi_s;
            shamrock::solvergraph::Field<Tscal> &spans_phi_As;
            shamrock::solvergraph::Field<Tscal> &spans_phi_hadamard_prod;
            shamrock::solvergraph::ScalarEdge<Tscal> &old_values;
            shamrock::solvergraph::ScalarEdge<Tscal> &new_values;
            shamrock::solvergraph::ScalarEdge<Tscal> &e_norm;
            shamrock::solvergraph::ScalarEdge<Tscal> &alpha;
            shamrock::solvergraph::ScalarEdge<Tscal> &beta;
            shamrock::solvergraph::ScalarEdge<Tscal> &w_stab;
        };

        inline void set_edges(
            std::shared_ptr<shamrock::solvergraph::Indexes<u32>> sizes,
            std::shared_ptr<solvergraph::OrientedAMRGraphEdge<Tvec, TgridVec>> cell_neigh_graph,
            std::shared_ptr<shamrock::solvergraph::Field<Tscal>> spans_block_cell_sizes,
            std::shared_ptr<shamrock::solvergraph::FieldRefs<Tscal>> spans_rho,
            std::shared_ptr<shamrock::solvergraph::ScalarEdge<Tscal>> mean_rho,
            std::shared_ptr<shamrock::solvergraph::FieldRefs<Tscal>> spans_phi,
            std::shared_ptr<shamrock::solvergraph::Field<Tscal>> spans_phi_res,
            std::shared_ptr<shamrock::solvergraph::Field<Tscal>> spans_phi_res_bis,
            std::shared_ptr<shamrock::solvergraph::Field<Tscal>> spans_phi_p,
            std::shared_ptr<shamrock::solvergraph::Field<Tscal>> spans_phi_Ap,
            std::shared_ptr<shamrock::solvergraph::Field<Tscal>> spans_phi_s,
            std::shared_ptr<shamrock::solvergraph::Field<Tscal>> spans_phi_As,
            std::shared_ptr<shamrock::solvergraph::Field<Tscal>> spans_phi_hadamard_prod,
            std::shared_ptr<shamrock::solvergraph::ScalarEdge<Tscal>> old_values,
            std::shared_ptr<shamrock::solvergraph::ScalarEdge<Tscal>> new_values,
            std::shared_ptr<shamrock::solvergraph::ScalarEdge<Tscal>> e_norm,
            std::shared_ptr<shamrock::solvergraph::ScalarEdge<Tscal>> alpha,
            std::shared_ptr<shamrock::solvergraph::ScalarEdge<Tscal>> beta,
            std::shared_ptr<shamrock::solvergraph::ScalarEdge<Tscal>> w_stab

        ) {
            __internal_set_ro_edges(
                {sizes, cell_neigh_graph, spans_block_cell_sizes, spans_rho, mean_rho});
            __internal_set_rw_edges(
                {spans_phi,
                 spans_phi_res,
                 spans_phi_res_bis,
                 spans_phi_p,
                 spans_phi_Ap,
                 spans_phi_s,
                 spans_phi_As,
                 spans_phi_hadamard_prod,
                 old_values,
                 new_values,
                 e_norm,
                 alpha,
                 beta,
                 w_stab

                });

            // set node0_1 edges
            node0_1.set_edges(
                sizes,
                cell_neigh_graph,
                spans_block_cell_sizes,
                spans_phi,
                spans_rho,
                mean_rho,
                spans_phi_res,
                spans_phi_res_bis,
                spans_phi_p);
            // set node0_2
            node0_2.set_edges(sizes, spans_phi_res, spans_phi_res_bis, spans_phi_hadamard_prod);
            // set node0_3
            node0_3.set_edges(sizes, spans_phi_hadamard_prod, old_values);

            // set node1_1
            node1_1.set_edges(
                sizes, cell_neigh_graph, spans_block_cell_sizes, spans_phi_p, spans_phi_Ap);
            // set node1_2
            node1_2.set_edges(sizes, spans_phi_Ap, spans_phi_res_bis, spans_phi_hadamard_prod);
            // set node1_3
            node1_3.set_edges(sizes, spans_phi_hadamard_prod, e_norm);

            // set node2_1
            node2_1.set_edges(sizes, spans_phi_res, spans_phi_Ap, e_norm, beta, alpha, spans_phi_s);
            // set node2_2
            node2_2.set_edges(spans_phi_s, e_norm);
            // set node2_3
            node2_3.set_edges(sizes, spans_phi_p, alpha, spans_phi);

            // set node3_1
            node3_1.set_edges(
                sizes, cell_neigh_graph, spans_block_cell_sizes, spans_phi_s, spans_phi_As);
            // set node3_2
            node3_2.set_edges(sizes, spans_phi_As, spans_phi_s, spans_phi_hadamard_prod);
            // set node3_3
            node3_3.set_edges(sizes, spans_phi_hadamard_prod, e_norm);
            // set node3_4
            node3_4.set_edges(spans_phi_As, new_values);

            // set node4
            node4.set_edges(sizes, spans_phi_p, spans_phi_s, alpha, w_stab, spans_phi);

            // set node5_1
            node5_1.set_edges(
                sizes, spans_phi_s, spans_phi_As, e_norm, new_values, w_stab, spans_phi_res);
            // set node5_2
            node5_2.set_edges(spans_phi_res, e_norm);

            // set node6_1
            node6_1.set_edges(sizes, spans_phi_res, spans_phi_res_bis, spans_phi_hadamard_prod);
            // set node6_2
            node6_2.set_edges(sizes, spans_phi_hadamard_prod, new_values);

            // set node7
            node7.set_edges(sizes, spans_phi_res, spans_phi_Ap, beta, e_norm, alpha, spans_phi_p);

            // set node8_1
            node8_1.set_edges(sizes, spans_phi_res, spans_phi_res_bis);
            node8_2.set_edges(sizes, spans_phi_res, spans_phi_p);
        }

        inline Edges get_edges() {
            return Edges{
                get_ro_edge<shamrock::solvergraph::Indexes<u32>>(0),
                get_ro_edge<solvergraph::OrientedAMRGraphEdge<Tvec, TgridVec>>(1),
                get_ro_edge<shamrock::solvergraph::Field<Tscal>>(2),
                get_ro_edge<shamrock::solvergraph::FieldRefs<Tscal>>(3),
                get_ro_edge<shamrock::solvergraph::ScalarEdge<Tscal>>(4),
                get_rw_edge<shamrock::solvergraph::FieldRefs<Tscal>>(0),
                get_rw_edge<shamrock::solvergraph::Field<Tscal>>(1),
                get_rw_edge<shamrock::solvergraph::Field<Tscal>>(2),
                get_rw_edge<shamrock::solvergraph::Field<Tscal>>(3),
                get_rw_edge<shamrock::solvergraph::Field<Tscal>>(4),
                get_rw_edge<shamrock::solvergraph::Field<Tscal>>(5),
                get_rw_edge<shamrock::solvergraph::Field<Tscal>>(6),
                get_rw_edge<shamrock::solvergraph::Field<Tscal>>(7),
                get_rw_edge<shamrock::solvergraph::ScalarEdge<Tscal>>(8),
                get_rw_edge<shamrock::solvergraph::ScalarEdge<Tscal>>(9),
                get_rw_edge<shamrock::solvergraph::ScalarEdge<Tscal>>(10),
                get_rw_edge<shamrock::solvergraph::ScalarEdge<Tscal>>(11),
                get_rw_edge<shamrock::solvergraph::ScalarEdge<Tscal>>(12),
                get_rw_edge<shamrock::solvergraph::ScalarEdge<Tscal>>(13)
                //
            };
        }

        void _impl_evaluate_internal();

        inline virtual std::string _impl_get_label() { return "BICGSTABMainLoop"; };

        virtual std::string _impl_get_tex();
    };

} // namespace shammodels::basegodunov::modules
