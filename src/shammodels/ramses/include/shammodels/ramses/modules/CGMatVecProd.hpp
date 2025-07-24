// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright (c) 2021-2024 Timothée David--Cléris <tim.shamrock@proton.me>
// SPDX-License-Identifier: CeCILL Free Software License Agreement v2.1
// Shamrock is licensed under the CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

#pragma once

/**
 * @file CGMatVecProd.hpp
 * @author Léodasce Sewanou (leodasce.sewanou@ens-lyon.fr)
 * @brief Implementation of matrix-vector product [A*p] for the conjugate gradient solver.
 *
 */

#include "shambase/aliases_int.hpp"
#include "shambackends/vec.hpp"
#include "shammodels/ramses/SolverConfig.hpp"
#include "shammodels/ramses/solvegraph/OrientedAMRGraphEdge.hpp"
#include "shamrock/solvergraph/IFieldSpan.hpp"
#include "shamrock/solvergraph/INode.hpp"
#include "shamrock/solvergraph/Indexes.hpp"
#include <memory>

namespace shammodels::basegodunov::modules {

    template<class Tvec, class TgridVec>
    class NodeCGMatVecProd : public shamrock::solvergraph::INode {
        using Tscal = shambase::VecComponent<Tvec>;
        u32 block_size;

        public:
        NodeCGMatVecProd(u32 block_size) : block_size(block_size) {}

        struct Edges {
            const shamrock::solvergraph::Indexes<u32> &sizes;
            const solvergraph::OrientedAMRGraphEdge<Tvec, TgridVec> &cell_neigh_graph;
            const shamrock::solvergraph::IFieldSpan<Tscal> &spans_block_cell_sizes;
            const shamrock::solvergraph::IFieldSpan<Tscal> &spans_phi_p;
            shamrock::solvergraph::IFieldSpan<Tscal> &spans_phi_Ap;
        };

        inline void set_edges(
            std::shared_ptr<shamrock::solvergraph::Indexes<u32>> sizes,
            std::shared_ptr<solvergraph::OrientedAMRGraphEdge<Tvec, TgridVec>> cell_neigh_graph,
            std::shared_ptr<shamrock::solvergraph::IFieldSpan<Tscal>> spans_block_cell_sizes,
            std::shared_ptr<shamrock::solvergraph::IFieldSpan<Tscal>> spans_phi_p,
            std::shared_ptr<shamrock::solvergraph::IFieldSpan<Tscal>> spans_phi_Ap) {
            __internal_set_ro_edges({sizes, cell_neigh_graph, spans_block_cell_sizes, spans_phi_p});
            __internal_set_rw_edges({spans_phi_Ap});
        }

        inline Edges get_edges() {
            return Edges{
                get_ro_edge<shamrock::solvergraph::Indexes<u32>>(0),
                get_ro_edge<solvergraph::OrientedAMRGraphEdge<Tvec, TgridVec>>(1),
                get_ro_edge<shamrock::solvergraph::IFieldSpan<Tscal>>(2),
                get_ro_edge<shamrock::solvergraph::IFieldSpan<Tscal>>(3),
                get_rw_edge<shamrock::solvergraph::IFieldSpan<Tscal>>(0)};
        }

        void _impl_evaluate_internal();
        inline virtual std::string _impl_get_label() { return "CGMatVecProd"; };
        virtual std::string _impl_get_tex();
    };
} // namespace shammodels::basegodunov::modules
