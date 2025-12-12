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
 * @file ParticleInCell.hpp
 * @author
 * @brief
 */

#include "shambackends/vec.hpp"
#include "shamrock/solvergraph/IFieldSpan.hpp"
#include "shamrock/solvergraph/INode.hpp"
#include "shamrock/solvergraph/Indexes.hpp"

namespace shammodels::basegodunov::modules {
    template<class Tvec>
    class NodePIC : public shamrock::solvergraph::INode {
        using Tscal = shambase::VecComponent<Tvec>;
        u32 block_size;

        public:
        NodePIC(u32 block_size) : block_size(block_size) {}

        struct Edges {
            const shamrock::solvergraph::Indexes<u32> &sizes;
            const shamrock::solvergraph::IFieldSpan<Tscal> &spans_mass_particles;
            shamrock::solvergraph::IFieldSpan<Tscal> &spans_rho_pic;
        };

        inline void set_edges(
            std::shared_ptr<shamrock::solvergraph::Indexes<u32>> sizes,
            std::shared_ptr<shamrock::solvergraph::IFieldSpan<Tscal>> spans_mass_particles,
            std::shared_ptr<shamrock::solvergraph::IFieldSpan<Tscal>> spans_rho_pic) {
            __internal_set_ro_edges({sizes, spans_mass_particles});
            __internal_set_rw_edges({spans_rho_pic});
        }

        inline Edges get_edges() {
            return Edges{
                get_ro_edge<shamrock::solvergraph::Indexes<u32>>(0),
                get_ro_edge<shamrock::solvergraph::IFieldSpan<Tscal>>(1),
                get_rw_edge<shamrock::solvergraph::IFieldSpan<Tscal>>(0),
            };
        }

        void _impl_evaluate_internal();

        inline virtual std::string _impl_get_label() const { return "ParticleInCell"; };

        virtual std::string _impl_get_tex() const;
    };
} // namespace shammodels::basegodunov::modules
