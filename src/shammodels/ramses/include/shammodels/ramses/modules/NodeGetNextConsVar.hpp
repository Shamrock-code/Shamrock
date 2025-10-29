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
 * @file NodeGetNextConsVar.hpp
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
#include "shamrock/solvergraph/ScalarEdge.hpp"
#include <memory>
#include <string>

namespace shammodels::basegodunov::modules {

    template<class Tvec, class TgridVec>
    class NodeGetNextConsVar : public shamrock::solvergraph::INode {
        using Tscal    = shambase::VecComponent<Tvec>;
        using Config   = SolverConfig<Tvec, TgridVec>;
        using AMRBlock = typename Config::AMRBlock;

        u32 block_size;
        Tscal dt;

        public:
        NodeGetNextConsVar(u32 block_size, Tscal dt) : block_size(block_size), dt(dt) {}

        struct Edges {
            const shamrock::solvergraph::Indexes<u32> &sizes;
            const solvergraph::OrientedAMRGraphEdge<Tvec, TgridVec> &cell_neigh_graph;
            const shamrock::solvergraph::IFieldSpan<Tscal> &spans_block_cell_sizes;
            const shamrock::solvergraph::IFieldSpan<Tvec> &spans_cell0block_aabb_lower;

            const solvergraph::NeighGraphLinkFieldEdge<Tvec> &flux_rhov_face_xp;
            const solvergraph::NeighGraphLinkFieldEdge<Tvec> &flux_rhov_face_xm;
            const solvergraph::NeighGraphLinkFieldEdge<Tvec> &flux_rhov_face_yp;
            const solvergraph::NeighGraphLinkFieldEdge<Tvec> &flux_rhov_face_ym;
            const solvergraph::NeighGraphLinkFieldEdge<Tvec> &flux_rhov_face_zp;
            const solvergraph::NeighGraphLinkFieldEdge<Tvec> &flux_rhov_face_zm;
            const solvergraph::NeighGraphLinkFieldEdge<Tscal> &flux_rhoe_face_xp;
            const solvergraph::NeighGraphLinkFieldEdge<Tscal> &flux_rhoe_face_xm;
            const solvergraph::NeighGraphLinkFieldEdge<Tscal> &flux_rhoe_face_yp;
            const solvergraph::NeighGraphLinkFieldEdge<Tscal> &flux_rhoe_face_ym;
            const solvergraph::NeighGraphLinkFieldEdge<Tscal> &flux_rhoe_face_zp;
            const solvergraph::NeighGraphLinkFieldEdge<Tscal> &flux_rhoe_face_zm;

            const shamrock::solvergraph::IFieldSpan<Tscal> &spans_rho_old;
            const shamrock::solvergraph::IFieldSpan<Tscal> &spans_rho_next;
            const shamrock::solvergraph::IFieldSpan<Tvec> &spans_rhov_old;
            const shamrock::solvergraph::IFieldSpan<Tscal> &spans_rhoe_old;
            const shamrock::solvergraph::IFieldSpan<Tvec> &spans_phi_g_old;
            const shamrock::solvergraph::IFieldSpan<Tvec> &spans_phi_g_next;
            const shamrock::solvergraph::ScalarEdge<Tscal> &dt;

            shamrock::solvergraph::IFieldSpan<Tvec> &spans_rhov_next;
            shamrock::solvergraph::IFieldSpan<Tscal> &spans_rhoe_next;
        };

        inline void set_edges(
            std::shared_ptr<shamrock::solvergraph::Indexes<u32>> sizes,
            std::shared_ptr<solvergraph::OrientedAMRGraphEdge<Tvec, TgridVec>> cell_neigh_graph,
            std::shared_ptr<shamrock::solvergraph::IFieldSpan<Tscal>> spans_block_cell_sizes,
            std::shared_ptr<shamrock::solvergraph::IFieldSpan<Tvec>> spans_cell0block_aabb_lower,

            std::shared_ptr<solvergraph::NeighGraphLinkFieldEdge<Tvec>> flux_rhov_face_xp,
            std::shared_ptr<solvergraph::NeighGraphLinkFieldEdge<Tvec>> flux_rhov_face_xm,
            std::shared_ptr<solvergraph::NeighGraphLinkFieldEdge<Tvec>> flux_rhov_face_yp,
            std::shared_ptr<solvergraph::NeighGraphLinkFieldEdge<Tvec>> flux_rhov_face_ym,
            std::shared_ptr<solvergraph::NeighGraphLinkFieldEdge<Tvec>> flux_rhov_face_zp,
            std::shared_ptr<solvergraph::NeighGraphLinkFieldEdge<Tvec>> flux_rhov_face_zm,
            std::shared_ptr<solvergraph::NeighGraphLinkFieldEdge<Tscal>> flux_rhoe_face_xp,
            std::shared_ptr<solvergraph::NeighGraphLinkFieldEdge<Tscal>> flux_rhoe_face_xm,
            std::shared_ptr<solvergraph::NeighGraphLinkFieldEdge<Tscal>> flux_rhoe_face_yp,
            std::shared_ptr<solvergraph::NeighGraphLinkFieldEdge<Tscal>> flux_rhoe_face_ym,
            std::shared_ptr<solvergraph::NeighGraphLinkFieldEdge<Tscal>> flux_rhoe_face_zp,
            std::shared_ptr<solvergraph::NeighGraphLinkFieldEdge<Tscal>> flux_rhoe_face_zm,

            std::shared_ptr<shamrock::solvergraph::IFieldSpan<Tscal>> spans_rho_old,
            std::shared_ptr<shamrock::solvergraph::IFieldSpan<Tscal>> spans_rho_next,
            std::shared_ptr<shamrock::solvergraph::IFieldSpan<Tvec>> spans_rhov_old,
            std::shared_ptr<shamrock::solvergraph::IFieldSpan<Tscal>> spans_rhoe_old,
            std::shared_ptr<shamrock::solvergraph::IFieldSpan<Tvec>> spans_phi_g_old,
            std::shared_ptr<shamrock::solvergraph::IFieldSpan<Tvec>> spans_phi_g_next,
            std::shared_ptr<shamrock::solvergraph::ScalarEdge<Tscal>> dt,

            std::shared_ptr<shamrock::solvergraph::IFieldSpan<Tvec>> spans_rhov_next,
            std::shared_ptr<shamrock::solvergraph::IFieldSpan<Tscal>> spans_rhoe_next) {

            __internal_set_ro_edges(
                {sizes,
                 cell_neigh_graph,
                 spans_block_cell_sizes,
                 spans_cell0block_aabb_lower,
                 flux_rhov_face_xp,
                 flux_rhov_face_xm,
                 flux_rhov_face_yp,
                 flux_rhov_face_ym,
                 flux_rhov_face_zp,
                 flux_rhov_face_zm,
                 flux_rhoe_face_xp,
                 flux_rhoe_face_xm,
                 flux_rhoe_face_yp,
                 flux_rhoe_face_ym,
                 flux_rhoe_face_zp,
                 flux_rhoe_face_zm,
                 spans_rho_old,
                 spans_rho_next,
                 spans_rhov_old,
                 spans_rhoe_old,
                 spans_phi_g_old,
                 spans_phi_g_next,
                 dt});

            __internal_set_rw_edges({spans_rhov_next, spans_rhoe_next});
        }

        inline Edges get_edges() {
            return Edges{
                get_ro_edge<shamrock::solvergraph::Indexes<u32>>(0),
                get_ro_edge<solvergraph::OrientedAMRGraphEdge<Tvec, TgridVec>>(1),
                get_ro_edge<shamrock::solvergraph::IFieldSpan<Tscal>>(2),
                get_ro_edge<shamrock::solvergraph::IFieldSpan<Tvec>>(3),
                get_ro_edge<solvergraph::NeighGraphLinkFieldEdge<Tvec>>(4),
                get_ro_edge<solvergraph::NeighGraphLinkFieldEdge<Tvec>>(5),
                get_ro_edge<solvergraph::NeighGraphLinkFieldEdge<Tvec>>(6),
                get_ro_edge<solvergraph::NeighGraphLinkFieldEdge<Tvec>>(7),
                get_ro_edge<solvergraph::NeighGraphLinkFieldEdge<Tvec>>(8),
                get_ro_edge<solvergraph::NeighGraphLinkFieldEdge<Tvec>>(9),
                get_ro_edge<solvergraph::NeighGraphLinkFieldEdge<Tscal>>(10),
                get_ro_edge<solvergraph::NeighGraphLinkFieldEdge<Tscal>>(11),
                get_ro_edge<solvergraph::NeighGraphLinkFieldEdge<Tscal>>(12),
                get_ro_edge<solvergraph::NeighGraphLinkFieldEdge<Tscal>>(13),
                get_ro_edge<solvergraph::NeighGraphLinkFieldEdge<Tscal>>(14),
                get_ro_edge<solvergraph::NeighGraphLinkFieldEdge<Tscal>>(15),
                get_ro_edge<shamrock::solvergraph::IFieldSpan<Tscal>>(16),
                get_ro_edge<shamrock::solvergraph::IFieldSpan<Tscal>>(17),
                get_ro_edge<shamrock::solvergraph::IFieldSpan<Tvec>>(18),
                get_ro_edge<shamrock::solvergraph::IFieldSpan<Tscal>>(19),
                get_ro_edge<shamrock::solvergraph::IFieldSpan<Tvec>>(20),
                get_ro_edge<shamrock::solvergraph::IFieldSpan<Tvec>>(21),
                get_ro_edge<shamrock::solvergraph::ScalarEdge<Tscal>>(22),
                get_rw_edge<shamrock::solvergraph::IFieldSpan<Tvec>>(0),
                get_rw_edge<shamrock::solvergraph::IFieldSpan<Tscal>>(1)};
        }

        void _impl_evaluate_internal();

        inline virtual std::string _impl_get_label() { return "NodeGetNextConsVar"; };

        virtual std::string _impl_get_tex() { return "TODO"; }
    };

} // namespace shammodels::basegodunov::modules
