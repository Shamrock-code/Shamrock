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
 * @file ConsToPrimDust.hpp
 * @author Timothée David--Cléris (tim.shamrock@proton.me)
 * @brief Field variant object to instanciate a variant on the patch types
 * @date 2023-07-31
 */

#include "shambackends/vec.hpp"
#include "shamrock/solvergraph/IFieldSpan.hpp"
#include "shamrock/solvergraph/INode.hpp"
#include "shamrock/solvergraph/Indexes.hpp"

namespace shammodels::basegodunov::modules {
    template<class Tvec>
    class NodeConsToPrimDust : public shamrock::solvergraph::INode {
        using Tscal = shambase::VecComponent<Tvec>;
        u32 block_size;
        u32 ndust;

        public:
        NodeConsToPrimDust(u32 block_size, u32 ndust) : block_size(block_size), ndust(ndust) {}

        struct Edges {
            const shamrock::solvergraph::Indexes<u32> &sizes;
            const shamrock::solvergraph::IFieldSpan<Tscal> &spans_rho_dust;
            const shamrock::solvergraph::IFieldSpan<Tvec> &spans_rhov_dust;
            shamrock::solvergraph::IFieldSpan<Tvec> &spans_vel_dust;
        };

        inline void set_edges(
            std::shared_ptr<shamrock::solvergraph::Indexes<u32>> sizes,
            std::shared_ptr<shamrock::solvergraph::IFieldSpan<Tscal>> spans_rho_dust,
            std::shared_ptr<shamrock::solvergraph::IFieldSpan<Tvec>> spans_rhov_dust,
            std::shared_ptr<shamrock::solvergraph::IFieldSpan<Tvec>> spans_vel_dust) {
            __internal_set_ro_edges({sizes, spans_rho_dust, spans_rhov_dust});
            __internal_set_rw_edges({spans_vel_dust});
        }

        inline Edges get_edges() {
            return Edges{
                get_ro_edge<shamrock::solvergraph::Indexes<u32>>(0),
                get_ro_edge<shamrock::solvergraph::IFieldSpan<Tscal>>(1),
                get_ro_edge<shamrock::solvergraph::IFieldSpan<Tvec>>(2),
                get_rw_edge<shamrock::solvergraph::IFieldSpan<Tvec>>(0),
            };
        }

        void _impl_evaluate_internal();

        inline virtual std::string _impl_get_label() { return "ConsToPrimDust"; };

        virtual std::string _impl_get_tex();
    };
} // namespace shammodels::basegodunov::modules
