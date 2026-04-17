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
 * @file NodeGetNextConsVar.hpp
 * @author Léodasce Sewanou (leodasce.sewanou@ens-lyon.fr)
 * @author Timothée David--Cléris (tim.shamrock@proton.me) --no git blame--
 * @brief
 *
 */

#include "shambase/aliases_int.hpp"
#include "shambackends/vec.hpp"
#include "shammodels/ramses/SolverConfig.hpp"
#include "shamrock/solvergraph/IFieldSpan.hpp"
#include "shamrock/solvergraph/INode.hpp"
#include "shamrock/solvergraph/Indexes.hpp"
#include "shamrock/solvergraph/ScalarEdge.hpp"
#include <memory>
#include <string>

namespace shammodels::basegodunov::modules {

    template<class Tvec>
    class NodeGetNextConsVar : public shamrock::solvergraph::INode {
        using Tscal = shambase::VecComponent<Tvec>;

        u32 block_size;
        Tscal dt_over2;

        public:
        NodeGetNextConsVar(u32 block_size) : block_size(block_size) {}

        struct Edges {
            const shamrock::solvergraph::Indexes<u32> &sizes;

            const shamrock::solvergraph::IFieldSpan<Tscal> &spans_rho_old;
            const shamrock::solvergraph::IFieldSpan<Tscal> &spans_rho_next;
            const shamrock::solvergraph::IFieldSpan<Tvec> &spans_rhov_old;
            const shamrock::solvergraph::IFieldSpan<Tscal> &spans_rhoe_old;
            const shamrock::solvergraph::IFieldSpan<Tvec> &spans_phi_g_old;
            const shamrock::solvergraph::IFieldSpan<Tvec> &spans_phi_g_next;
            const shamrock::solvergraph::IFieldSpan<Tvec> &spans_dt_rhov_old;
            const shamrock::solvergraph::IFieldSpan<Tscal> &spans_dt_rhoe_old;
            const shamrock::solvergraph::ScalarEdge<Tscal> &dt_over2;

            shamrock::solvergraph::IFieldSpan<Tvec> &spans_rhov_next;
            shamrock::solvergraph::IFieldSpan<Tscal> &spans_rhoe_next;
        };

        inline void set_edges(
            std::shared_ptr<shamrock::solvergraph::Indexes<u32>> sizes,

            std::shared_ptr<shamrock::solvergraph::IFieldSpan<Tscal>> spans_rho_old,
            std::shared_ptr<shamrock::solvergraph::IFieldSpan<Tscal>> spans_rho_next,
            std::shared_ptr<shamrock::solvergraph::IFieldSpan<Tvec>> spans_rhov_old,
            std::shared_ptr<shamrock::solvergraph::IFieldSpan<Tscal>> spans_rhoe_old,
            std::shared_ptr<shamrock::solvergraph::IFieldSpan<Tvec>> spans_phi_g_old,
            std::shared_ptr<shamrock::solvergraph::IFieldSpan<Tvec>> spans_phi_g_next,
            std::shared_ptr<shamrock::solvergraph::IFieldSpan<Tvec>> spans_dt_rhov_old,
            std::shared_ptr<shamrock::solvergraph::IFieldSpan<Tscal>> spans_dt_rhoe_old,
            std::shared_ptr<shamrock::solvergraph::ScalarEdge<Tscal>> dt_over2,
            std::shared_ptr<shamrock::solvergraph::IFieldSpan<Tvec>> spans_rhov_next,
            std::shared_ptr<shamrock::solvergraph::IFieldSpan<Tscal>> spans_rhoe_next) {

            __internal_set_ro_edges(
                {sizes,
                 spans_rho_old,
                 spans_rho_next,
                 spans_rhov_old,
                 spans_rhoe_old,
                 spans_phi_g_old,
                 spans_phi_g_next,
                 spans_dt_rhov_old,
                 spans_dt_rhoe_old,
                 dt_over2});

            __internal_set_rw_edges({spans_rhov_next, spans_rhoe_next});
        }

        inline Edges get_edges() {
            return Edges{
                get_ro_edge<shamrock::solvergraph::Indexes<u32>>(0),
                get_ro_edge<shamrock::solvergraph::IFieldSpan<Tscal>>(1),
                get_ro_edge<shamrock::solvergraph::IFieldSpan<Tscal>>(2),
                get_ro_edge<shamrock::solvergraph::IFieldSpan<Tvec>>(3),
                get_ro_edge<shamrock::solvergraph::IFieldSpan<Tscal>>(4),
                get_ro_edge<shamrock::solvergraph::IFieldSpan<Tvec>>(5),
                get_ro_edge<shamrock::solvergraph::IFieldSpan<Tvec>>(6),
                get_ro_edge<shamrock::solvergraph::IFieldSpan<Tvec>>(7),
                get_ro_edge<shamrock::solvergraph::IFieldSpan<Tscal>>(8),
                get_ro_edge<shamrock::solvergraph::ScalarEdge<Tscal>>(9),

                get_rw_edge<shamrock::solvergraph::IFieldSpan<Tvec>>(0),
                get_rw_edge<shamrock::solvergraph::IFieldSpan<Tscal>>(1)};
        }

        void _impl_evaluate_internal();

        inline virtual std::string _impl_get_label() const { return "NodeGetNextConsVar"; };

        virtual std::string _impl_get_tex() const { return "TODO"; }
    };

} // namespace shammodels::basegodunov::modules
