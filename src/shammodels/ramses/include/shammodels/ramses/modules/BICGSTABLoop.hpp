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
 * @file BICGSTABLoop.hpp
 * @author Léodasce Sewanou (leodasce.sewanou@ens-lyon.fr)
 * @author Timothée David--Cléris (tim.shamrock@proton.me) --no git blame--
 * @brief
 *
 */

#include "shambase/aliases_int.hpp"
#include "shambackends/vec.hpp"
#include "shammodels/ramses/SolverConfig.hpp"
#include "shammodels/ramses/modules/BICGSTABInit.hpp"
#include "shammodels/ramses/modules/NodeAXPY.hpp"
#include "shammodels/ramses/modules/NodeAXPYThreeVectors.hpp"
#include "shammodels/ramses/modules/NodeAYPX.hpp"
#include "shammodels/ramses/modules/NodeHadamardProd.hpp"
#include "shammodels/ramses/modules/NodeLinCombThreeVectors.hpp"
#include "shammodels/ramses/modules/NodeOverwrite.hpp"
#include "shammodels/ramses/modules/NodeSpMVPoisson3D.hpp"
#include "shammodels/ramses/modules/NodeSumReduction.hpp"
#include "shammodels/ramses/modules/ResidualDot.hpp"
#include "shammodels/ramses/modules/SolverStorage.hpp"
#include "shammodels/ramses/solvegraph/OrientedAMRGraphEdge.hpp"
#include "shamrock/scheduler/ShamrockCtx.hpp"
#include "shamrock/solvergraph/ExchangeGhostField.hpp"
#include "shamrock/solvergraph/ExtractGhostField.hpp"
#include "shamrock/solvergraph/Field.hpp"
#include "shamrock/solvergraph/FieldRefs.hpp"
#include "shamrock/solvergraph/INode.hpp"
#include "shamrock/solvergraph/Indexes.hpp"
#include "shamrock/solvergraph/ReplaceGhostField.hpp"
#include "shamrock/solvergraph/ScalarEdge.hpp"
#include <memory>

namespace shammodels::basegodunov::modules {
    template<class Tvec, class TgridVec>
    class NodeBICGSTABLoop : public shamrock::solvergraph::INode {
        using Tscal    = shambase::VecComponent<Tvec>;
        using AMRBlock = typename shammodels::basegodunov::SolverConfig<Tvec, TgridVec>::AMRBlock;

        u32 block_size;
        Tscal fourPiG;
        u32 Niter_max;
        Tscal tol_cvg;
        Tscal tol_happy_bk;

        public:
        NodeBICGSTABLoop(
            u32 block_size, Tscal fourPiG, u32 Niter_max, Tscal tol_cvg, Tscal tol_happy_bk)
            : block_size(block_size), fourPiG(fourPiG), Niter_max(Niter_max), tol_cvg(tol_cvg),
              tol_happy_bk(tol_happy_bk) {}

        // init node
        modules::BICGSTABInit<Tvec, TgridVec> node_init{block_size, fourPiG};
        // hadamardProd node for dot product <r_0, r'_0>
        modules::NodeHadamardProd<Tscal> node_had_prod_rj_rp0{block_size};
        // dotprod node <r_0, z_0>
        modules::NodeSumReduction<Tscal> node_ddot_rj_rp0{block_size};

        // SpMV  Ap node
        modules::NodeSpMVPoisson3D<Tvec, TgridVec> node_Apj{block_size};
        // hadamardProd node Ap x r'0
        modules::NodeHadamardProd<Tscal> node_had_Apj_rp0{block_size};
        // dotprod node <Ap, r'_0>
        modules::NodeSumReduction<Tscal> node_ddot_Apj_rp0{block_size};

        // s_k node
        modules::NodeLinCombThreeVectors<Tscal> node_sj_vec{block_size};
        // ddot <s_k,s_k> node
        modules::ResidualDot<Tscal> node_ddot_sj_sj{block_size};
        // phi'_{k} = phi_{k} + alpha_k p_k
        modules::NodeAXPY<Tscal> node_new_phi_happy_break{block_size};

        // SpMV  As node
        modules::NodeSpMVPoisson3D<Tvec, TgridVec> node_Asj{block_size};
        // hadamardProd node As x s
        modules::NodeHadamardProd<Tscal> node_had_Asj_sj{block_size};
        // dotprod node <As, s>
        modules::NodeSumReduction<Tscal> node_ddot_Asj_sj{block_size};
        // ddot <As_{k},As_{k}> node
        modules::ResidualDot<Tscal> node_ddot_Asj_Asj{block_size};

        // New-phi node
        modules::NodeAXPYThreeVectors<Tscal> node_new_phi{block_size};

        // New-residual node
        modules::NodeLinCombThreeVectors<Tscal> node_new_res{block_size};
        // ddot <r_{k+1},r_{k+1}> node;
        modules::ResidualDot<Tscal> node_ddot_rj_rj{block_size};

        // hadamardProd node r_{k+1} x r'_0
        modules::NodeHadamardProd<Tscal> node_had_rjnew_rp0{block_size};
        // dotprod node < r_{k+1}, r'_0>
        modules::NodeSumReduction<Tscal> node_ddot_rjnew_rp0{block_size};

        // New-A-conjugate vector p node
        modules::NodeLinCombThreeVectors<Tscal> node_new_p_vec{block_size};

        // r'0 = r_{k+1} node
        modules::NodeOverwrite<Tscal> node_overwrite_rp0{block_size};
        // p_{k+1} = r_{k+1} node
        modules::NodeOverwrite<Tscal> node_overwrite_p{block_size};

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

        /***********************************/
        // Extract ghosts for Field
        shamrock::solvergraph::ExtractGhostField<Tscal> node_gz_s{};

        // Exchange ghosts for field
        shamrock::solvergraph::ExchangeGhostField<Tscal> node_exch_gz_s{};

        // Replace ghosts for field
        shamrock::solvergraph::ReplaceGhostField<Tscal> node_replace_gz_s{};

        struct Edges {
            const shamrock::solvergraph::Indexes<u32> &sizes;
            const solvergraph::OrientedAMRGraphEdge<Tvec, TgridVec> &cell_neigh_graph;
            const shamrock::solvergraph::Field<Tscal> &spans_block_cell_sizes;
            const shamrock::solvergraph::IFieldRefs<Tscal> &spans_rho;
            const shamrock::solvergraph::ScalarEdge<Tscal> &mean_rho;
            const shamrock::solvergraph::Indexes<u32> &sizes_no_gz;
            const shamrock::solvergraph::DDSharedBuffers<u32> &idx_in_ghost;
            const shamrock::solvergraph::RankGetter &rank_owner;
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
            std::shared_ptr<shamrock::solvergraph::IFieldRefs<Tscal>> spans_rho,
            std::shared_ptr<shamrock::solvergraph::ScalarEdge<Tscal>> mean_rho,
            std::shared_ptr<shamrock::solvergraph::Indexes<u32>> sizes_no_gz,
            std::shared_ptr<shamrock::solvergraph::DDSharedBuffers<u32>> idx_in_ghost,
            std::shared_ptr<shamrock::solvergraph::RankGetter> rank_owner,
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
                {sizes,
                 cell_neigh_graph,
                 spans_block_cell_sizes,
                 spans_rho,
                 mean_rho,
                 sizes_no_gz,
                 idx_in_ghost,
                 rank_owner});
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

            // set node_init edges
            node_init.set_edges(
                sizes,
                cell_neigh_graph,
                spans_block_cell_sizes,
                spans_phi,
                spans_rho,
                mean_rho,
                spans_phi_res,
                spans_phi_res_bis,
                spans_phi_p);
            // set node_had_prod_rj_rp0
            node_had_prod_rj_rp0.set_edges(
                sizes, spans_phi_res, spans_phi_res_bis, spans_phi_hadamard_prod);
            // set node_ddot_rj_rp0
            node_ddot_rj_rp0.set_edges(sizes_no_gz, spans_phi_hadamard_prod, old_values);

            // set node_Apj
            node_Apj.set_edges(
                sizes, cell_neigh_graph, spans_block_cell_sizes, spans_phi_p, spans_phi_Ap);
            // set node_had_Apj_rp0
            node_had_Apj_rp0.set_edges(
                sizes, spans_phi_Ap, spans_phi_res_bis, spans_phi_hadamard_prod);
            // set node_ddot_Apj_rp0
            node_ddot_Apj_rp0.set_edges(sizes_no_gz, spans_phi_hadamard_prod, e_norm);

            // set node_sj_vec
            node_sj_vec.set_edges(
                sizes, spans_phi_res, spans_phi_Ap, e_norm, beta, alpha, spans_phi_s);
            // set node_ddot_sj_sj
            node_ddot_sj_sj.set_edges(sizes_no_gz, spans_phi_s, e_norm);
            // set node_new_phi_happy_break
            node_new_phi_happy_break.set_edges(sizes, spans_phi_p, alpha, spans_phi);

            // set node_Asj
            node_Asj.set_edges(
                sizes, cell_neigh_graph, spans_block_cell_sizes, spans_phi_s, spans_phi_As);
            // set node_had_Asj_sj
            node_had_Asj_sj.set_edges(sizes, spans_phi_As, spans_phi_s, spans_phi_hadamard_prod);
            // set node_ddot_Asj_sj
            node_ddot_Asj_sj.set_edges(sizes_no_gz, spans_phi_hadamard_prod, e_norm);
            // set node_ddot_Asj_Asj
            node_ddot_Asj_Asj.set_edges(sizes_no_gz, spans_phi_As, new_values);

            // set node_new_phi
            node_new_phi.set_edges(sizes, spans_phi_p, spans_phi_s, alpha, w_stab, spans_phi);

            // set node_new_res
            node_new_res.set_edges(
                sizes, spans_phi_s, spans_phi_As, e_norm, new_values, w_stab, spans_phi_res);
            // set node_ddot_rj_rj
            node_ddot_rj_rj.set_edges(sizes_no_gz, spans_phi_res, e_norm);

            // // set node_had_rjnew_rp0
            node_had_rjnew_rp0.set_edges(
                sizes, spans_phi_res, spans_phi_res_bis, spans_phi_hadamard_prod);
            // set node_ddot_rjnew_rp0
            node_ddot_rjnew_rp0.set_edges(sizes_no_gz, spans_phi_hadamard_prod, new_values);

            // set node_new_p_vec
            node_new_p_vec.set_edges(
                sizes, spans_phi_res, spans_phi_Ap, beta, e_norm, alpha, spans_phi_p);

            // set node_overwrite_rp0
            node_overwrite_rp0.set_edges(sizes, spans_phi_res, spans_phi_res_bis);
            node_overwrite_p.set_edges(sizes, spans_phi_res, spans_phi_p);

            // set node_gz edges  for p-vectors
            node_gz_p.set_edges(spans_phi_p, idx_in_ghost, p_ghosts);

            // set node_exch_gz edges for p-vectors
            node_exch_gz_p.set_edges(rank_owner, p_ghosts);

            // replace ghosts for p-vectors
            node_replace_gz_p.set_edges(p_ghosts, spans_phi_p);

            // set node_gz edges  for s-vectors
            node_gz_s.set_edges(spans_phi_s, idx_in_ghost, p_ghosts);

            // set node_exch_gz edges for s-vectors
            node_exch_gz_s.set_edges(rank_owner, p_ghosts);

            // replace ghosts for s-vectors
            node_replace_gz_s.set_edges(p_ghosts, spans_phi_s);
        }

        inline Edges get_edges() {
            return Edges{
                get_ro_edge<shamrock::solvergraph::Indexes<u32>>(0),
                get_ro_edge<solvergraph::OrientedAMRGraphEdge<Tvec, TgridVec>>(1),
                get_ro_edge<shamrock::solvergraph::Field<Tscal>>(2),
                get_ro_edge<shamrock::solvergraph::IFieldRefs<Tscal>>(3),
                get_ro_edge<shamrock::solvergraph::ScalarEdge<Tscal>>(4),
                get_ro_edge<shamrock::solvergraph::Indexes<u32>>(5),
                get_ro_edge<shamrock::solvergraph::DDSharedBuffers<u32>>(6),
                get_ro_edge<shamrock::solvergraph::RankGetter>(7),
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

        inline virtual std::string _impl_get_label() const { return "BICGSTABMainLoop"; };

        virtual std::string _impl_get_tex() const;
    };

} // namespace shammodels::basegodunov::modules
