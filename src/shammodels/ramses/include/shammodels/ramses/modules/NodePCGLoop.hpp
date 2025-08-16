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
 * @file NodePCGLoop.hpp
 * @author Léodasce Sewanou (leodasce.sewanou@ens-lyon.fr)
 * @author Timothée David--Cléris (tim.shamrock@proton.me) --no git blame--
 * @brief
 *
 */

#include "shambase/aliases_int.hpp"
#include "shambackends/vec.hpp"
#include "shammodels/ramses/SolverConfig.hpp"
#include "shammodels/ramses/modules/NodeAXPYTwoVectors.hpp"
#include "shammodels/ramses/modules/NodeAYPXTwoVectors.hpp"
#include "shammodels/ramses/modules/NodeHadamardProd.hpp"
#include "shammodels/ramses/modules/NodeJacobiPreconditioner.hpp"
#include "shammodels/ramses/modules/NodeSpMVPoisson3D.hpp"
#include "shammodels/ramses/modules/NodeSumReduction.hpp"
#include "shammodels/ramses/modules/PCGInit.hpp"
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
    class NodePCGLoop : public shamrock::solvergraph::INode {
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
        Tscal tol;

        public:
        NodePCGLoop(
            ShamrockCtx &context,
            Config &solver_config,
            Storage &storage,
            u32 block_size,
            Tscal fourPiG,
            u32 Niter_max,
            Tscal tol)
            : context(context), solver_config(solver_config), storage(storage),
              block_size(block_size), fourPiG(fourPiG), Niter_max(Niter_max), tol(tol) {}

        // init node
        modules::PCGInit<Tvec, TgridVec> node0{block_size, fourPiG};
        // hadamardProd node for dot product <r_0, z_0>
        modules::NodeHadamardProd<Tscal> node1{block_size};
        // dotprod node <r_0, z_0>
        modules::NodeSumReduction<Tscal> node2{};
        // SpMV node
        modules::NodeSpMVPoisson3D<Tvec, TgridVec> node3_1{block_size};
        // hadamardProd node for
        modules::NodeHadamardProd<Tscal> node3_2{block_size};
        // A-norm node
        modules::NodeSumReduction<Tscal> node4{};
        // New-phi node
        modules::NodeAXPYTwoVectors<Tscal> node5{block_size};
        // New-residual node
        modules::NodeAXPYTwoVectors<Tscal> node6{block_size};
        // New-preconditioned residual node
        modules::NodeJacobiPreconditioner<Tscal> node7{block_size};
        // hadamardProd node for dot product <r_k+1, z_k+1>
        modules::NodeHadamardProd<Tscal> node8{block_size};
        // dotprod node <r_k+1, z_k+1>
        modules::NodeSumReduction<Tscal> node9{};
        // New-A-conjugate vector p node
        modules::NodeAYPXTwoVectors<Tscal> node10{block_size};

        struct Edges {
            const shamrock::solvergraph::Indexes<u32> &sizes;
            const solvergraph::OrientedAMRGraphEdge<Tvec, TgridVec> &cell_neigh_graph;
            const shamrock::solvergraph::Field<Tscal> &spans_block_cell_sizes;
            const shamrock::solvergraph::FieldRefs<Tscal> &spans_rho;
            const shamrock::solvergraph::ScalarEdge<Tscal> &mean_rho;
            shamrock::solvergraph::FieldRefs<Tscal> &spans_phi;
            shamrock::solvergraph::Field<Tscal> &spans_phi_res;
            shamrock::solvergraph::Field<Tscal> &spans_phi_pres;
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
            std::shared_ptr<shamrock::solvergraph::FieldRefs<Tscal>> spans_phi,
            std::shared_ptr<shamrock::solvergraph::Field<Tscal>> spans_phi_res,
            std::shared_ptr<shamrock::solvergraph::Field<Tscal>> spans_phi_pres,
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
                {sizes, cell_neigh_graph, spans_block_cell_sizes, spans_rho, mean_rho});
            __internal_set_rw_edges(
                {spans_phi,
                 spans_phi_res,
                 spans_phi_pres,
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
                spans_phi_pres,
                spans_phi_p);

            // set node1 edges
            node1.set_edges(sizes, spans_phi_res, spans_phi_pres, spans_phi_hadamard_prod);

            // set node2 edges
            node2.set_edges(sizes, spans_phi_hadamard_prod, old_values);

            // set node3_1 edges
            node3_1.set_edges(
                sizes, cell_neigh_graph, spans_block_cell_sizes, spans_phi_p, spans_phi_Ap);

            // set node3_2 edges
            node3_2.set_edges(sizes, spans_phi_p, spans_phi_Ap, spans_phi_hadamard_prod);

            // set node4 edges
            node4.set_edges(sizes, spans_phi_hadamard_prod, e_norm);

            // set node5 edges
            node5.set_edges(sizes, spans_phi_p, alpha, spans_phi);

            // set node6 edges
            node6.set_edges(sizes, spans_phi_Ap, alpha, spans_phi_res);

            // set node7 edges
            node7.set_edges(sizes, spans_block_cell_sizes, spans_phi_res, spans_phi_pres);

            // set node8 edges
            node8.set_edges(sizes, spans_phi_res, spans_phi_pres, spans_phi_hadamard_prod);

            // set node9 edges
            node9.set_edges(sizes, spans_phi_hadamard_prod, new_values);

            // set node10 edges
            node10.set_edges(sizes, spans_phi_pres, beta, spans_phi_p);
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
                get_rw_edge<shamrock::solvergraph::ScalarEdge<Tscal>>(6),
                get_rw_edge<shamrock::solvergraph::ScalarEdge<Tscal>>(7),
                get_rw_edge<shamrock::solvergraph::ScalarEdge<Tscal>>(8),
                get_rw_edge<shamrock::solvergraph::ScalarEdge<Tscal>>(9),
                get_rw_edge<shamrock::solvergraph::ScalarEdge<Tscal>>(10)
                //
            };
        }

        void _impl_evaluate_internal();

        inline virtual std::string _impl_get_label() { return "PCGMainLoop"; };

        virtual std::string _impl_get_tex();
    };

} // namespace shammodels::basegodunov::modules
