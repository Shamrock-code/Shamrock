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
 * @file NodeUpdateMomentum.hpp
 * @author Léodasce Sewanou (leodasce.sewanou@ens-lyon.fr)
 * @author Timothée David--Cléris (tim.shamrock@proton.me) --no git blame--
 * @brief
 *
 */

#include "shambackends/vec.hpp"
#include "shammodels/ramses/SolverConfig.hpp"
#include "shamrock/solvergraph/IFieldRefs.hpp"
#include "shamrock/solvergraph/IFieldSpan.hpp"
#include "shamrock/solvergraph/INode.hpp"
#include "shamrock/solvergraph/Indexes.hpp"
#include "shamrock/solvergraph/ScalarEdge.hpp"

namespace shammodels::basegodunov::modules {

    template<class Tvec>
    class NodeUpdateMomentum : public shamrock::solvergraph::INode {
        using Tscal = shambase::VecComponent<Tvec>;
        u32 block_size;

        public:
        NodeUpdateMomentum(u32 block_size) : block_size(block_size) {}

        struct Edges {
            const shamrock::solvergraph::Indexes<u32> &sizes;
            const shamrock::solvergraph::Indexes<u32> &sizes_no_gz;
            const shamrock::solvergraph::IFieldSpan<Tscal> &spans_rho;
            const shamrock::solvergraph::IFieldSpan<Tvec> &spans_g;
            const shamrock::solvergraph::ScalarEdge<Tscal> &dt;
            shamrock::solvergraph::IFieldRefs<Tvec> &spans_rhovel;
            shamrock::solvergraph::IFieldRefs<Tscal> &spans_rhoe;
        };

        inline void set_edges(
            std::shared_ptr<shamrock::solvergraph::Indexes<u32>> sizes,
            std::shared_ptr<shamrock::solvergraph::Indexes<u32>> sizes_no_gz,
            std::shared_ptr<shamrock::solvergraph::IFieldSpan<Tscal>> spans_rho,
            std::shared_ptr<shamrock::solvergraph::IFieldSpan<Tvec>> spans_g,
            std::shared_ptr<shamrock::solvergraph::ScalarEdge<Tscal>> dt,
            std::shared_ptr<shamrock::solvergraph::IFieldSpan<Tvec>> spans_rhovel,
            std::shared_ptr<shamrock::solvergraph::IFieldSpan<Tscal>> spans_rhoe) {

            __internal_set_ro_edges({sizes, sizes_no_gz, spans_rho, spans_g, dt});
            __internal_set_rw_edges({spans_rhovel, spans_rhoe});
        }

        inline Edges get_edges() {
            return Edges{

                get_ro_edge<shamrock::solvergraph::Indexes<u32>>(0),
                get_ro_edge<shamrock::solvergraph::Indexes<u32>>(1),
                get_ro_edge<shamrock::solvergraph::IFieldSpan<Tscal>>(2),
                get_ro_edge<shamrock::solvergraph::IFieldSpan<Tvec>>(3),
                get_ro_edge<shamrock::solvergraph::ScalarEdge<Tscal>>(4),
                get_rw_edge<shamrock::solvergraph::IFieldRefs<Tvec>>(0),
                get_rw_edge<shamrock::solvergraph::IFieldRefs<Tscal>>(1)

            };
        }

        void _impl_evaluate_internal();

        inline virtual std::string _impl_get_label() { return "UnpdateMomemtum"; };

        virtual std::string _impl_get_tex() { return "TODO"; };
    };

} // namespace shammodels::basegodunov::modules
