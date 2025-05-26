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
 * @file SlopeLimitedGradient.hpp
 * @author Timothée David--Cléris (timothee.david--cleris@ens-lyon.fr)
 * @brief
 *
 */

#include "shambackends/vec.hpp"
#include "shammodels/ramses/SolverConfig.hpp"
#include "shammodels/ramses/solvegraph/OrientedAMRGraphEdge.hpp"
#include "shamrock/solvergraph/IFieldSpan.hpp"
#include "shamrock/solvergraph/INode.hpp"
#include "shamrock/solvergraph/Indexes.hpp"

namespace shammodels::basegodunov::modules {

    template<class Tvec, class TgridVec>
    class SlopeLimitedScalarGradient : public shamrock::solvergraph::INode {
        using Tscal = shambase::VecComponent<Tvec>;

        using SlopeMode = shammodels::basegodunov::SlopeMode;

        u32 block_size;
        u32 var_per_cell;
        SlopeMode mode;

        public:
        SlopeLimitedScalarGradient(u32 block_size, u32 var_per_cell, SlopeMode mode)
            : block_size(block_size), var_per_cell(var_per_cell), mode(mode) {}

        struct Edges {
            const shamrock::solvergraph::Indexes<u32> &sizes;
            const solvergraph::OrientedAMRGraphEdge<Tvec, TgridVec> &cell_neigh_graph;
            const shamrock::solvergraph::IFieldSpan<Tscal> &spans_block_cell_sizes;
            const shamrock::solvergraph::IFieldSpan<Tscal> &span_field;
            shamrock::solvergraph::IFieldSpan<Tvec> &span_grad_field;
        };

        inline void set_edges(
            std::shared_ptr<shamrock::solvergraph::Indexes<u32>> sizes,
            std::shared_ptr<solvergraph::OrientedAMRGraphEdge<Tvec, TgridVec>> cell_neigh_graph,
            std::shared_ptr<shamrock::solvergraph::IFieldSpan<Tscal>> spans_block_cell_sizes,
            std::shared_ptr<shamrock::solvergraph::IFieldSpan<Tscal>> span_field,
            std::shared_ptr<shamrock::solvergraph::IFieldSpan<Tvec>> span_grad_field) {
            __internal_set_ro_edges({sizes, cell_neigh_graph, spans_block_cell_sizes, span_field});
            __internal_set_rw_edges({span_grad_field});
        }

        inline Edges get_edges() {
            return Edges{
                get_ro_edge<shamrock::solvergraph::Indexes<u32>>(0),
                get_ro_edge<solvergraph::OrientedAMRGraphEdge<Tvec, TgridVec>>(1),
                get_ro_edge<shamrock::solvergraph::IFieldSpan<Tscal>>(2),
                get_ro_edge<shamrock::solvergraph::IFieldSpan<Tscal>>(3),
                get_rw_edge<shamrock::solvergraph::IFieldSpan<Tvec>>(0),
            };
        }

        void _impl_evaluate_internal();

        inline virtual std::string _impl_get_label() { return "SlopeLimitedScalarGradient"; };

        virtual std::string _impl_get_tex();
    };

    template<class Tvec, class TgridVec>
    class SlopeLimitedVectorGradient : public shamrock::solvergraph::INode {
        using Tscal = shambase::VecComponent<Tvec>;

        using SlopeMode = shammodels::basegodunov::SlopeMode;

        u32 block_size;
        u32 var_per_cell;
        SlopeMode mode;

        public:
        SlopeLimitedVectorGradient(u32 block_size, u32 var_per_cell, SlopeMode mode)
            : block_size(block_size), var_per_cell(var_per_cell), mode(mode) {}

        struct Edges {
            const shamrock::solvergraph::Indexes<u32> &sizes;
            const solvergraph::OrientedAMRGraphEdge<Tvec, TgridVec> &cell_neigh_graph;
            const shamrock::solvergraph::IFieldSpan<Tscal> &spans_block_cell_sizes;
            const shamrock::solvergraph::IFieldSpan<Tvec> &span_field;
            shamrock::solvergraph::IFieldSpan<Tvec> &span_dx_field;
            shamrock::solvergraph::IFieldSpan<Tvec> &span_dy_field;
            shamrock::solvergraph::IFieldSpan<Tvec> &span_dz_field;
        };

        inline void set_edges(
            std::shared_ptr<shamrock::solvergraph::Indexes<u32>> sizes,
            std::shared_ptr<solvergraph::OrientedAMRGraphEdge<Tvec, TgridVec>> cell_neigh_graph,
            std::shared_ptr<shamrock::solvergraph::IFieldSpan<Tscal>> spans_block_cell_sizes,
            std::shared_ptr<shamrock::solvergraph::IFieldSpan<Tvec>> span_field,
            std::shared_ptr<shamrock::solvergraph::IFieldSpan<Tvec>> span_dx_field,
            std::shared_ptr<shamrock::solvergraph::IFieldSpan<Tvec>> span_dy_field,
            std::shared_ptr<shamrock::solvergraph::IFieldSpan<Tvec>> span_dz_field) {
            __internal_set_ro_edges({sizes, cell_neigh_graph, spans_block_cell_sizes, span_field});
            __internal_set_rw_edges({span_dx_field, span_dy_field, span_dz_field});
        }

        inline Edges get_edges() {
            return Edges{
                get_ro_edge<shamrock::solvergraph::Indexes<u32>>(0),
                get_ro_edge<solvergraph::OrientedAMRGraphEdge<Tvec, TgridVec>>(1),
                get_ro_edge<shamrock::solvergraph::IFieldSpan<Tscal>>(2),
                get_ro_edge<shamrock::solvergraph::IFieldSpan<Tvec>>(3),
                get_rw_edge<shamrock::solvergraph::IFieldSpan<Tvec>>(0),
                get_rw_edge<shamrock::solvergraph::IFieldSpan<Tvec>>(1),
                get_rw_edge<shamrock::solvergraph::IFieldSpan<Tvec>>(2),
            };
        }

        void _impl_evaluate_internal();

        inline virtual std::string _impl_get_label() { return "SlopeLimitedScalarGradient"; };

        virtual std::string _impl_get_tex();
    };
} // namespace shammodels::basegodunov::modules
