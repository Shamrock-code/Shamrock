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
 * @file BICGSTABInit.hpp
 * @author Léodasce Sewanou (leodasce.sewanou@ens-lyon.fr)
 * @author Timothée David--Cléris (tim.shamrock@proton.me) --no git blame--
 * @brief
 *
 */

#include "shambase/aliases_int.hpp"
#include "shambackends/vec.hpp"
#include "shammodels/ramses/SolverConfig.hpp"
#include "shammodels/ramses/solvegraph/OrientedAMRGraphEdge.hpp"
#include "shamrock/solvergraph/IFieldSpan.hpp"
#include "shamrock/solvergraph/INode.hpp"
#include "shamrock/solvergraph/Indexes.hpp"
#include "shamrock/solvergraph/ScalarEdge.hpp"
#include <memory>

namespace shammodels::basegodunov::modules {

    template<class Tvec, class TgridVec>
    class BICGSTABInit : public shamrock::solvergraph::INode {
        using Tscal = shambase::VecComponent<Tvec>;
        u32 block_size;
        Tscal fourPiG;

        public:
        BICGSTABInit(u32 block_size, Tscal fourPiG) : block_size(block_size), fourPiG(fourPiG) {}

        struct Edges {
            const shamrock::solvergraph::Indexes<u32> &sizes;
            const solvergraph::OrientedAMRGraphEdge<Tvec, TgridVec> &cell_neigh_graph;
            const shamrock::solvergraph::IFieldSpan<Tscal> &spans_block_cell_sizes;
            const shamrock::solvergraph::IFieldSpan<Tscal> &spans_phi;
            const shamrock::solvergraph::IFieldSpan<Tscal> &spans_rho;
            const shamrock::solvergraph::ScalarEdge<Tscal> &mean_rho;
            shamrock::solvergraph::IFieldSpan<Tscal> &spans_phi_res;
            shamrock::solvergraph::IFieldSpan<Tscal> &spans_phi_res_bis;
            shamrock::solvergraph::IFieldSpan<Tscal> &spans_phi_p;
        };

        inline void set_edges(
            std::shared_ptr<shamrock::solvergraph::Indexes<u32>> sizes,
            std::shared_ptr<solvergraph::OrientedAMRGraphEdge<Tvec, TgridVec>> cell_neigh_graph,
            std::shared_ptr<shamrock::solvergraph::IFieldSpan<Tscal>> spans_block_cell_sizes,
            std::shared_ptr<shamrock::solvergraph::IFieldSpan<Tscal>> spans_phi,
            std::shared_ptr<shamrock::solvergraph::IFieldSpan<Tscal>> spans_rho,
            std::shared_ptr<shamrock::solvergraph::ScalarEdge<Tscal>> mean_rho,
            std::shared_ptr<shamrock::solvergraph::IFieldSpan<Tscal>> spans_phi_res,
            std::shared_ptr<shamrock::solvergraph::IFieldSpan<Tscal>> spans_phi_res_bis,
            std::shared_ptr<shamrock::solvergraph::IFieldSpan<Tscal>> spans_phi_p) {
            __internal_set_ro_edges(
                {sizes, cell_neigh_graph, spans_block_cell_sizes, spans_phi, spans_rho, mean_rho});
            __internal_set_rw_edges({spans_phi_res, spans_phi_res_bis, spans_phi_p});
        }

        inline Edges get_edges() {
            return Edges{
                get_ro_edge<shamrock::solvergraph::Indexes<u32>>(0),
                get_ro_edge<solvergraph::OrientedAMRGraphEdge<Tvec, TgridVec>>(1),
                get_ro_edge<shamrock::solvergraph::IFieldSpan<Tscal>>(2),
                get_ro_edge<shamrock::solvergraph::IFieldSpan<Tscal>>(3),
                get_ro_edge<shamrock::solvergraph::IFieldSpan<Tscal>>(4),
                get_ro_edge<shamrock::solvergraph::ScalarEdge<Tscal>>(5),
                get_rw_edge<shamrock::solvergraph::IFieldSpan<Tscal>>(0),
                get_rw_edge<shamrock::solvergraph::IFieldSpan<Tscal>>(1),
                get_rw_edge<shamrock::solvergraph::IFieldSpan<Tscal>>(2)};
        }

        void _impl_evaluate_internal();
        inline virtual std::string _impl_get_label() { return "BICGSTABInit"; };
        virtual std::string _impl_get_tex();
    };
} // namespace shammodels::basegodunov::modules
