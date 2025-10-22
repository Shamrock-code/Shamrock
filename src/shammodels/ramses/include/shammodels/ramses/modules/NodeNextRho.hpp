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
 * @file NodeNextRho.hpp
 * @author Léodasce Sewanou (leodasce.sewanou@ens-lyon.fr)
 * @author Timothée David--Cléris (tim.shamrock@proton.me) --no git blame--
 * @brief
 *
 */

#include "shambase/aliases_int.hpp"
#include "shambackends/vec.hpp"
#include "shammodels/common/amr/NeighGraph.hpp"
#include "shammodels/ramses/SolverConfig.hpp"
#include "shammodels/ramses/solvegraph/NeighGraphLinkFieldEdge.hpp"
#include "shammodels/ramses/solvegraph/OrientedAMRGraphEdge.hpp"
#include "shamrock/solvergraph/IFieldSpan.hpp"
#include "shamrock/solvergraph/INode.hpp"
#include "shamrock/solvergraph/Indexes.hpp"
#include <memory>
#include <string>

namespace shammodels::basegodunov::modules {

    template<class Tvec, class TgridVec>
    class NodeNextRho : public shamrock::solvergraph::INode {

        using Tscal    = shambase::VecComponent<Tvec>;
        using Config   = SolverConfig<Tvec, TgridVec>;
        using AMRBlock = typename Config::AMRBlock;

        u32 block_size;
        Tscal dt;

        public:
        NodeNextRho(u32 block_size, Tscal dt) : block_size(block_size), dt(dt) {}

        struct Edges {
            const shamrock::solvergraph::Indexes<u32> &sizes;
            const solvergraph::OrientedAMRGraphEdge<Tvec, TgridVec> &cell_neigh_graph;
            const shamrock::solvergraph::IFieldSpan<Tscal> &spans_block_cell_sizes;
            const shamrock::solvergraph::IFieldSpan<Tvec> &spans_cell0block_aabb_lower;
            const solvergraph::NeighGraphLinkFieldEdge<Tscal> &flux_rho_face_xp;
            const solvergraph::NeighGraphLinkFieldEdge<Tscal> &flux_rho_face_xm;
            const solvergraph::NeighGraphLinkFieldEdge<Tscal> &flux_rho_face_yp;
            const solvergraph::NeighGraphLinkFieldEdge<Tscal> &flux_rho_face_ym;
            const solvergraph::NeighGraphLinkFieldEdge<Tscal> &flux_rho_face_zp;
            const solvergraph::NeighGraphLinkFieldEdge<Tscal> &flux_rho_face_zm;

            shamrock::solvergraph::IFieldSpan<Tscal> &spans_rho;
        };

        inline void set_edges(
            std::shared_ptr<shamrock::solvergraph::Indexes<u32>> sizes,
            std::shared_ptr<solvergraph::OrientedAMRGraphEdge<Tvec, TgridVec>> cell_neigh_graph,
            std::shared_ptr<shamrock::solvergraph::IFieldSpan<Tscal>> spans_block_cell_sizes,
            std::shared_ptr<shamrock::solvergraph::IFieldSpan<Tvec>> spans_cell0block_aabb_lower,
            std::shared_ptr<solvergraph::NeighGraphLinkFieldEdge<Tscal>> flux_rho_face_xp,
            std::shared_ptr<solvergraph::NeighGraphLinkFieldEdge<Tscal>> flux_rho_face_xm,
            std::shared_ptr<solvergraph::NeighGraphLinkFieldEdge<Tscal>> flux_rho_face_yp,
            std::shared_ptr<solvergraph::NeighGraphLinkFieldEdge<Tscal>> flux_rho_face_ym,
            std::shared_ptr<solvergraph::NeighGraphLinkFieldEdge<Tscal>> flux_rho_face_zp,
            std::shared_ptr<solvergraph::NeighGraphLinkFieldEdge<Tscal>> flux_rho_face_zm,
            std::shared_ptr<shamrock::solvergraph::IFieldSpan<Tscal>> spans_rho) {
            __internal_set_ro_edges(
                {sizes,
                 cell_neigh_graph,
                 spans_block_cell_sizes,
                 spans_cell0block_aabb_lower,
                 flux_rho_face_xp,
                 flux_rho_face_xm,
                 flux_rho_face_yp,
                 flux_rho_face_ym,
                 flux_rho_face_zp,
                 flux_rho_face_zm});
            __internal_set_rw_edges({spans_rho});
        }

        inline Edges get_edges() {
            return Edges{
                get_ro_edge<shamrock::solvergraph::Indexes<u32>>(0),
                get_ro_edge<solvergraph::OrientedAMRGraphEdge<Tvec, TgridVec>>(1),
                get_ro_edge<shamrock::solvergraph::IFieldSpan<Tscal>>(2),
                get_ro_edge<shamrock::solvergraph::IFieldSpan<Tvec>>(3),
                get_ro_edge<solvergraph::NeighGraphLinkFieldEdge<Tscal>>(4),
                get_ro_edge<solvergraph::NeighGraphLinkFieldEdge<Tscal>>(5),
                get_ro_edge<solvergraph::NeighGraphLinkFieldEdge<Tscal>>(6),
                get_ro_edge<solvergraph::NeighGraphLinkFieldEdge<Tscal>>(7),
                get_ro_edge<solvergraph::NeighGraphLinkFieldEdge<Tscal>>(8),
                get_ro_edge<solvergraph::NeighGraphLinkFieldEdge<Tscal>>(9),
                get_rw_edge<shamrock::solvergraph::IFieldSpan<Tscal>>(0)};
        }

        void _impl_evaluate_internal();

        inline virtual std::string _impl_get_label() { return "NodeNextRho"; };

        virtual std::string _impl_get_tex() { return "TODO"; }
    };

} // namespace shammodels::basegodunov::modules
