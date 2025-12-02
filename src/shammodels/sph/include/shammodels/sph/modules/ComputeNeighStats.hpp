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
 * @file ComputeNeighStats.hpp
 * @author Timothée David--Cléris (tim.shamrock@proton.me)
 * @brief A module to compute and display statistics on neighbor counts for SPH particles.
 *
 */

#include "shambackends/vec.hpp"
#include "shammodels/sph/solvergraph/NeighCache.hpp"
#include "shamrock/solvergraph/IDataEdge.hpp"
#include "shamrock/solvergraph/IFieldSpan.hpp"
#include "shamrock/solvergraph/INode.hpp"
#include "shamrock/solvergraph/Indexes.hpp"

namespace shammodels::sph::modules {

    template<class Tvec>
    class ComputeNeighStats : public shamrock::solvergraph::INode {

        using Tscal = shambase::VecComponent<Tvec>;

        Tscal kernel_radius;
        std::string filename_dump;

        public:
        ComputeNeighStats(Tscal kernel_radius, std::string filename)
            : kernel_radius(kernel_radius), filename_dump(filename) {}

        struct Edges {
            const shamrock::solvergraph::IDataEdge<Tscal> &sim_time;
            const shamrock::solvergraph::Indexes<u32> &part_counts;
            const shammodels::sph::solvergraph::NeighCache &neigh_cache;
            const shamrock::solvergraph::IFieldSpan<Tvec> &xyz;
            const shamrock::solvergraph::IFieldSpan<Tscal> &hpart;
        };

        inline void set_edges(
            std::shared_ptr<shamrock::solvergraph::IDataEdge<Tscal>> sim_time, // should be optional
            std::shared_ptr<shamrock::solvergraph::Indexes<u32>> part_counts,
            std::shared_ptr<shammodels::sph::solvergraph::NeighCache> neigh_cache,
            std::shared_ptr<shamrock::solvergraph::IFieldSpan<Tvec>> xyz,
            std::shared_ptr<shamrock::solvergraph::IFieldSpan<Tscal>> hpart) {
            __internal_set_ro_edges({sim_time, part_counts, neigh_cache, xyz, hpart});
            __internal_set_rw_edges({});
        }

        inline Edges get_edges() {
            return Edges{
                get_ro_edge<shamrock::solvergraph::IDataEdge<Tscal>>(0),
                get_ro_edge<shamrock::solvergraph::Indexes<u32>>(1),
                get_ro_edge<shammodels::sph::solvergraph::NeighCache>(2),
                get_ro_edge<shamrock::solvergraph::IFieldSpan<Tvec>>(3),
                get_ro_edge<shamrock::solvergraph::IFieldSpan<Tscal>>(4),
            };
        }

        void _impl_evaluate_internal();

        inline virtual std::string _impl_get_label() const { return "ComputeNeighStats"; };

        virtual std::string _impl_get_tex() const;
    };
} // namespace shammodels::sph::modules
