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
 * @file NodeBiCGSTABLoop.hpp
 * @author Léodasce Sewanou (leodasce.sewanou@ens-lyon.fr)
 * @author Timothée David--Cléris (tim.shamrock@proton.me)
 * @brief BiCGSTAB solver of the Poisson equation for the self-gravity.
 *
 * Adapted from the PhD work of Léodasce Sewanou.
 */

#include "shambase/aliases_int.hpp"
#include "shambackends/vec.hpp"
#include "shammodels/ramses/modules/BiCGSTABInit.hpp"
#include "shammodels/ramses/modules/NodeSpMVPoisson3D.hpp"
#include "shammodels/ramses/modules/ResidualDot.hpp"
#include "shammodels/ramses/solvegraph/OrientedAMRGraphEdge.hpp"
#include "shamrock/solvergraph/DDSharedBuffers.hpp"
#include "shamrock/solvergraph/ExchangeGhostField.hpp"
#include "shamrock/solvergraph/ExtractGhostField.hpp"
#include "shamrock/solvergraph/Field.hpp"
#include "shamrock/solvergraph/IFieldRefs.hpp"
#include "shamrock/solvergraph/IFieldSpan.hpp"
#include "shamrock/solvergraph/Indexes.hpp"
#include "shamrock/solvergraph/NodeAXPY.hpp"
#include "shamrock/solvergraph/NodeAXPYThreeVectors.hpp"
#include "shamrock/solvergraph/NodeHadamardProd.hpp"
#include "shamrock/solvergraph/NodeLinCombThreeVectors.hpp"
#include "shamrock/solvergraph/NodeOverwrite.hpp"
#include "shamrock/solvergraph/NodeSumReduction.hpp"
#include "shamrock/solvergraph/PatchDataFieldDDShared.hpp"
#include "shamrock/solvergraph/RankGetter.hpp"
#include "shamrock/solvergraph/ReplaceGhostField.hpp"
#include "shamrock/solvergraph/ScalarEdge.hpp"
#include "shamsolvergraph/edge/IDataEdge.hpp"
#include "shamsolvergraph/node/INode.hpp"
#include <memory>

#define NODE_EDGES(X_RO, X_RW)                                                                     \
    /* ------------------- inputs ------------------- */                                           \
    X_RO(shamrock::solvergraph::Indexes<u32>, sizes)                                               \
    X_RO(shamrock::solvergraph::Indexes<u32>, sizes_no_gz)                                         \
    X_RO(CellGraphEdge, cell_neigh_graph)                                                          \
    X_RO(shamrock::solvergraph::IFieldSpan<Tscal>, spans_block_cell_sizes)                         \
    X_RO(shamrock::solvergraph::IFieldSpan<Tscal>, spans_rho)                                      \
    X_RO(shamrock::solvergraph::ScalarEdge<Tscal>, mean_rho)                                       \
    X_RO(shamrock::solvergraph::DDSharedBuffers<u32>, idx_in_ghost)                                \
    X_RO(shamrock::solvergraph::RankGetter, rank_owner)                                            \
                                                                                                   \
    /* ------------------- outputs ------------------- */                                          \
    X_RW(shamrock::solvergraph::IFieldRefs<Tscal>, spans_phi)                                      \
                                                                                                   \
    /* ------------------- work fields ------------------- */                                      \
    X_RW(shamrock::solvergraph::Field<Tscal>, spans_phi_res)                                       \
    X_RW(shamrock::solvergraph::Field<Tscal>, spans_phi_res_bis)                                   \
    X_RW(shamrock::solvergraph::Field<Tscal>, spans_phi_p)                                         \
    X_RW(shamrock::solvergraph::Field<Tscal>, spans_phi_Ap)                                        \
    X_RW(shamrock::solvergraph::Field<Tscal>, spans_phi_s)                                         \
    X_RW(shamrock::solvergraph::Field<Tscal>, spans_phi_As)                                        \
    X_RW(shamrock::solvergraph::Field<Tscal>, spans_phi_hadamard_prod)                             \
    X_RW(shamrock::solvergraph::IDataEdge<Tscal>, old_values)                                      \
    X_RW(shamrock::solvergraph::IDataEdge<Tscal>, new_values)                                      \
    X_RW(shamrock::solvergraph::IDataEdge<Tscal>, e_norm)                                          \
    X_RW(shamrock::solvergraph::IDataEdge<Tscal>, alpha)                                           \
    X_RW(shamrock::solvergraph::IDataEdge<Tscal>, beta)                                            \
    X_RW(shamrock::solvergraph::IDataEdge<Tscal>, w_stab)                                          \
    X_RW(shamrock::solvergraph::IDataEdge<Tscal>, shadow_res_norm)                                 \
    X_RW(shamrock::solvergraph::Field<Tscal>, spans_rhs)                                           \
    X_RW(shamrock::solvergraph::IDataEdge<Tscal>, rhs_norm_value)

namespace shammodels::basegodunov::modules {

    /**
     * @brief Solve \f$ \Delta \phi = 4 \pi G (\rho - \bar{\rho}) \f$ with the BiCGSTAB method,
     * starting from the current content of `spans_phi`.
     *
     * Port of the BiCGSTAB of the `new_amr_plus_grav` branch of Léodasce Sewanou. The matrix, the
     * right-hand side and the handling of the ghost blocks are the same as in NodeCGLoop. The ghost
     * blocks of both \f$ p_k \f$ and \f$ s_k \f$ are exchanged before the matrix-vector products,
     * and the ones of \f$ \phi \f$ at the end.
     *
     * The loop stops when \f$ \langle s_k, s_k \rangle / \langle b, b \rangle < {\rm tol}^2 \f$
     * (then \f$ \phi_{k+1} = \phi_k + \alpha_k p_k \f$) or \f$ \langle r_{k+1}, r_{k+1} \rangle /
     * \langle b, b \rangle < {\rm tol}^2 \f$, on a breakdown (\f$ \langle A p_k, r'_0 \rangle \f$,
     * \f$ \langle A s_k, A s_k \rangle \f$ or \f$ \omega_k \f$ equal to 0), or after `Niter_max`
     * iterations. It restarts (\f$ r'_0 = p_{k+1} = r_{k+1} \f$) when
     * \f$ \langle r_{k+1}, r'_0 \rangle^2 < {\rm tol\_happy\_bk}^2 \langle r_{k+1}, r_{k+1} \rangle
     * \langle r'_0, r'_0 \rangle \f$.
     *
     * The work fields are resized to match `sizes` if needed.
     */
    template<class Tvec, class TgridVec>
    class NodeBiCGSTABLoop : public shamrock::solvergraph::INode {
        using Tscal         = shambase::VecComponent<Tvec>;
        using CellGraphEdge = solvergraph::OrientedAMRGraphEdge<Tvec, TgridVec>;

        u32 block_size;
        Tscal fourPiG;
        u32 Niter_max;
        Tscal tol_cvg;
        Tscal tol_happy_bk;

        u32 last_iteration_count = 0;

        /// r_0 = b - A phi_0, r'_0 = r_0 / 2, p_0 = r_0
        BiCGSTABInit<Tvec, TgridVec> node_init{block_size, fourPiG};
        /// <b, b>
        ResidualDot<Tscal> node_ddot_rhs{block_size};
        /// <r'_0, r'_0>
        ResidualDot<Tscal> node_ddot_rstarj_rstarj{block_size};
        /// <r_k, r'_0>
        shamrock::solvergraph::NodeHadamardProd<Tscal> node_had_prod_rj_rp0{block_size};
        shamrock::solvergraph::NodeSumReduction<Tscal> node_ddot_rj_rp0{block_size};
        /// A p_k
        NodeSpMVPoisson3D<Tvec, TgridVec> node_Apj{block_size};
        /// <A p_k, r'_0>
        shamrock::solvergraph::NodeHadamardProd<Tscal> node_had_Apj_rp0{block_size};
        shamrock::solvergraph::NodeSumReduction<Tscal> node_ddot_Apj_rp0{block_size};
        /// s_k = r_k - alpha_k A p_k
        shamrock::solvergraph::NodeLinCombThreeVectors<Tscal> node_sj_vec{block_size};
        /// <s_k, s_k>
        ResidualDot<Tscal> node_ddot_sj_sj{block_size};
        /// phi_{k+1} = phi_k + alpha_k p_k (convergence on s_k)
        shamrock::solvergraph::NodeAXPY<Tscal> node_new_phi_happy_break{block_size};
        /// A s_k
        NodeSpMVPoisson3D<Tvec, TgridVec> node_Asj{block_size};
        /// <A s_k, s_k>
        shamrock::solvergraph::NodeHadamardProd<Tscal> node_had_Asj_sj{block_size};
        shamrock::solvergraph::NodeSumReduction<Tscal> node_ddot_Asj_sj{block_size};
        /// <A s_k, A s_k>
        ResidualDot<Tscal> node_ddot_Asj_Asj{block_size};
        /// phi_{k+1} = phi_k + alpha_k p_k + w_k s_k
        shamrock::solvergraph::NodeAXPYThreeVectors<Tscal> node_new_phi{block_size};
        /// r_{k+1} = s_k - w_k A s_k
        shamrock::solvergraph::NodeLinCombThreeVectors<Tscal> node_new_res{block_size};
        /// <r_{k+1}, r_{k+1}>
        ResidualDot<Tscal> node_ddot_rj_rj{block_size};
        /// <r_{k+1}, r'_0>
        shamrock::solvergraph::NodeHadamardProd<Tscal> node_had_rjnew_rp0{block_size};
        shamrock::solvergraph::NodeSumReduction<Tscal> node_ddot_rjnew_rp0{block_size};
        /// p_{k+1} = r_{k+1} + beta_k (p_k - w_k A p_k)
        shamrock::solvergraph::NodeLinCombThreeVectors<Tscal> node_new_p_vec{block_size};
        /// restart: r'_0 = r_{k+1}
        shamrock::solvergraph::NodeOverwrite<Tscal> node_overwrite_rp0{block_size};
        /// restart: p_{k+1} = r_{k+1}
        shamrock::solvergraph::NodeOverwrite<Tscal> node_overwrite_p{block_size};

        // ghost exchange of p_k, s_k and phi (they share the same temporary ghost edge)
        std::shared_ptr<shamrock::solvergraph::PatchDataFieldDDShared<Tscal>> p_ghosts
            = std::make_shared<shamrock::solvergraph::PatchDataFieldDDShared<Tscal>>(
                "p_ghosts", "p_{\\rm ghosts}");
        shamrock::solvergraph::ExtractGhostField<Tscal> node_gz_p{};
        shamrock::solvergraph::ExchangeGhostField<Tscal> node_exch_gz_p{};
        shamrock::solvergraph::ReplaceGhostField<Tscal> node_replace_gz_p{};
        shamrock::solvergraph::ExtractGhostField<Tscal> node_gz_s{};
        shamrock::solvergraph::ExchangeGhostField<Tscal> node_exch_gz_s{};
        shamrock::solvergraph::ReplaceGhostField<Tscal> node_replace_gz_s{};
        shamrock::solvergraph::ExtractGhostField<Tscal> node_gz_phi{};
        shamrock::solvergraph::ExchangeGhostField<Tscal> node_exch_gz_phi{};
        shamrock::solvergraph::ReplaceGhostField<Tscal> node_replace_gz_phi{};

        public:
        NodeBiCGSTABLoop(
            u32 block_size, Tscal fourPiG, u32 Niter_max, Tscal tol_cvg, Tscal tol_happy_bk)
            : block_size(block_size), fourPiG(fourPiG), Niter_max(Niter_max), tol_cvg(tol_cvg),
              tol_happy_bk(tol_happy_bk) {}

        // same as EXPAND_NODE_EDGES(NODE_EDGES), with set_edges also wiring the inner nodes
        struct Edges {
            NODE_EDGES(INODE_DECL_RO, INODE_DECL_RW)
        };

        inline void set_edges(
            NODE_EDGES(INODE_PARAM_RO, INODE_PARAM_RW) SourceLocation loc = SourceLocation{}) {
            __shamrock_log_callsite(loc);

            __internal_set_ro_edges({NODE_EDGES(INODE_PUSH_RO1, INODE_PUSH_RW1)});
            __internal_set_rw_edges({NODE_EDGES(INODE_PUSH_RO2, INODE_PUSH_RW2)});

            node_init.set_edges(
                sizes,
                cell_neigh_graph,
                spans_block_cell_sizes,
                spans_phi,
                spans_rho,
                mean_rho,
                spans_phi_res,
                spans_phi_res_bis,
                spans_phi_p,
                spans_rhs);
            node_ddot_rhs.set_edges(sizes_no_gz, spans_rhs, rhs_norm_value);
            node_ddot_rstarj_rstarj.set_edges(sizes_no_gz, spans_phi_res_bis, shadow_res_norm);
            node_had_prod_rj_rp0.set_edges(
                sizes, spans_phi_res, spans_phi_res_bis, spans_phi_hadamard_prod);
            node_ddot_rj_rp0.set_edges(sizes_no_gz, spans_phi_hadamard_prod, old_values);
            node_Apj.set_edges(
                sizes, cell_neigh_graph, spans_block_cell_sizes, spans_phi_p, spans_phi_Ap);
            node_had_Apj_rp0.set_edges(
                sizes, spans_phi_Ap, spans_phi_res_bis, spans_phi_hadamard_prod);
            node_ddot_Apj_rp0.set_edges(sizes_no_gz, spans_phi_hadamard_prod, e_norm);
            // s = e_norm * s + beta * r + alpha * Ap
            node_sj_vec.set_edges(
                sizes, spans_phi_res, spans_phi_Ap, e_norm, beta, alpha, spans_phi_s);
            node_ddot_sj_sj.set_edges(sizes_no_gz, spans_phi_s, e_norm);
            node_new_phi_happy_break.set_edges(sizes, spans_phi_p, alpha, spans_phi);
            node_Asj.set_edges(
                sizes, cell_neigh_graph, spans_block_cell_sizes, spans_phi_s, spans_phi_As);
            node_had_Asj_sj.set_edges(sizes, spans_phi_As, spans_phi_s, spans_phi_hadamard_prod);
            node_ddot_Asj_sj.set_edges(sizes_no_gz, spans_phi_hadamard_prod, e_norm);
            node_ddot_Asj_Asj.set_edges(sizes_no_gz, spans_phi_As, new_values);
            // phi = phi + alpha * p + w_stab * s
            node_new_phi.set_edges(sizes, spans_phi_p, spans_phi_s, alpha, w_stab, spans_phi);
            // r = e_norm * r + new_values * s + w_stab * As
            node_new_res.set_edges(
                sizes, spans_phi_s, spans_phi_As, e_norm, new_values, w_stab, spans_phi_res);
            node_ddot_rj_rj.set_edges(sizes_no_gz, spans_phi_res, e_norm);
            node_had_rjnew_rp0.set_edges(
                sizes, spans_phi_res, spans_phi_res_bis, spans_phi_hadamard_prod);
            node_ddot_rjnew_rp0.set_edges(sizes_no_gz, spans_phi_hadamard_prod, new_values);
            // p = beta * p + e_norm * r + alpha * Ap
            node_new_p_vec.set_edges(
                sizes, spans_phi_res, spans_phi_Ap, beta, e_norm, alpha, spans_phi_p);
            node_overwrite_rp0.set_edges(sizes, spans_phi_res, spans_phi_res_bis);
            node_overwrite_p.set_edges(sizes, spans_phi_res, spans_phi_p);

            node_gz_p.set_edges(spans_phi_p, idx_in_ghost, p_ghosts);
            node_exch_gz_p.set_edges(rank_owner, p_ghosts);
            node_replace_gz_p.set_edges(p_ghosts, spans_phi_p);
            node_gz_s.set_edges(spans_phi_s, idx_in_ghost, p_ghosts);
            node_exch_gz_s.set_edges(rank_owner, p_ghosts);
            node_replace_gz_s.set_edges(p_ghosts, spans_phi_s);
            node_gz_phi.set_edges(spans_phi, idx_in_ghost, p_ghosts);
            node_exch_gz_phi.set_edges(rank_owner, p_ghosts);
            node_replace_gz_phi.set_edges(p_ghosts, spans_phi);
        }

        inline Edges get_edges() {
            int ro = 0;
            int rw = 0;
            return Edges{NODE_EDGES(INODE_GET_RO, INODE_GET_RW)};
        }

        INODE_EXPAND_TEX_SYMBOLS(
            NODE_EDGES(INODE_TEX_DECL, INODE_TEX_DECL),
            NODE_EDGES(INODE_TEX_GET_RO, INODE_TEX_GET_RW),
            NODE_EDGES(INODE_TEX_REPLACE, INODE_TEX_REPLACE))

        /// Number of iterations done by the last evaluation
        inline u32 get_last_iteration_count() const { return last_iteration_count; }

        void _impl_evaluate_internal();

        inline virtual std::string _impl_get_label() const { return "BiCGSTABMainLoop"; };

        virtual std::string _impl_get_tex() const;
    };

} // namespace shammodels::basegodunov::modules

#undef NODE_EDGES
