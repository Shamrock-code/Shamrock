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
 * @file ConsToPrim.hpp
 * @author Yona Lapeyre (yona.lapeyre@ens-lyon.fr)
 * @brief get conserved variables (rho*, momentum, entropy) from primitive variables (rho, vel, internal energy)
 */

#include "shambackends/vec.hpp"
#include "shamrock/solvergraph/IFieldSpan.hpp"
#include "shamrock/solvergraph/INode.hpp"
#include "shamrock/solvergraph/Indexes.hpp"

namespace shammodels::basegodunov::modules {
    template<class Tvec>
    class NodeConsToPrim : public shamrock::solvergraph::INode {
        using Tscal = shambase::VecComponent<Tvec>;
        u32 block_size;
        Tscal gamma;

        public:
        NodeConsToPrim(u32 block_size, Tscal gamma) : block_size(block_size), gamma(gamma) {}

        struct Edges {
            const shamrock::solvergraph::Indexes<u32> &sizes;
            const shamrock::solvergraph::IFieldSpan<Tscal> &spans_rhostar;
            const shamrock::solvergraph::IFieldSpan<Tvec> &spans_momentum;
            const shamrock::solvergraph::IFieldSpan<Tscal> &spans_K;
            shamrock::solvergraph::IFieldSpan<Tvec> &spans_vel;
            shamrock::solvergraph::IFieldSpan<Tscal> &spans_u;
            shamrock::solvergraph::IFieldSpan<Tscal> &spans_P;
        };

        inline void set_edges(
            std::shared_ptr<shamrock::solvergraph::Indexes<u32>> sizes,
            std::shared_ptr<shamrock::solvergraph::IFieldSpan<Tscal>> spans_rhostar,
            std::shared_ptr<shamrock::solvergraph::IFieldSpan<Tvec>> spans_momentum,
            std::shared_ptr<shamrock::solvergraph::IFieldSpan<Tscal>> spans_K,
            std::shared_ptr<shamrock::solvergraph::IFieldSpan<Tvec>> spans_vel,
            std::shared_ptr<shamrock::solvergraph::IFieldSpan<Tscal>> spans_u,
            std::shared_ptr<shamrock::solvergraph::IFieldSpan<Tscal>> spans_P) {
            __internal_set_ro_edges({sizes, spans_rhostar, spans_momentum, spans_K});
            __internal_set_rw_edges({spans_vel, spans_u, spans_P});
        }

        inline Edges get_edges() {
            return Edges{
                get_ro_edge<shamrock::solvergraph::Indexes<u32>>(0),
                get_ro_edge<shamrock::solvergraph::IFieldSpan<Tscal>>(1),
                get_ro_edge<shamrock::solvergraph::IFieldSpan<Tvec>>(2),
                get_ro_edge<shamrock::solvergraph::IFieldSpan<Tscal>>(3),
                get_rw_edge<shamrock::solvergraph::IFieldSpan<Tvec>>(0),
                get_rw_edge<shamrock::solvergraph::IFieldSpan<Tscal>>(1),
                get_rw_edge<shamrock::solvergraph::IFieldSpan<Tscal>>(2)
            };
        }

        void _impl_evaluate_internal();

        inline virtual std::string _impl_get_label() const { return "ConsToPrim"; };

        virtual std::string _impl_get_tex() const;
    };
} // namespace shammodels::basegodunov::modules
