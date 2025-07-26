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
 * @file IterateSmoothingLengthDensity.hpp
 * @author Timothée David--Cléris (tim.shamrock@proton.me)
 * @brief Declares the IterateSmoothingLengthDensity module for removing particles.
 *
 */

#include "shambackends/vec.hpp"
#include "shammodels/sph/solvergraph/NeighCache.hpp"
#include "shamrock/solvergraph/IFieldSpan.hpp"
#include "shamrock/solvergraph/INode.hpp"
#include "shamrock/solvergraph/Indexes.hpp"
#include <memory>

namespace shammodels::sph::modules {

    template<class Tvec, class SPHKernel>
    class IterateSmoothingLengthDensity : public shamrock::solvergraph::INode {

        using Tscal = shambase::VecComponent<Tvec>;

        Tscal gpart_mass;
        Tscal h_evol_max;
        Tscal h_evol_iter_max;

        public:
        IterateSmoothingLengthDensity(Tscal gpart_mass, Tscal h_evol_max, Tscal h_evol_iter_max)
            : gpart_mass(gpart_mass), h_evol_max(h_evol_max), h_evol_iter_max(h_evol_iter_max) {}

        struct Edges {
            const shamrock::solvergraph::Indexes<u32> &sizes;
            const shammodels::sph::solvergraph::NeighCache &neigh_cache;
            const shamrock::solvergraph::IFieldSpan<Tvec> &positions;
            const shamrock::solvergraph::IFieldSpan<Tscal> &old_h;
            shamrock::solvergraph::IFieldSpan<Tscal> &new_h;
            shamrock::solvergraph::IFieldSpan<Tscal> &eps_h;
        };

        inline void set_edges(
            std::shared_ptr<shamrock::solvergraph::Indexes<u32>> sizes,
            std::shared_ptr<shammodels::sph::solvergraph::NeighCache> neigh_cache,
            std::shared_ptr<shamrock::solvergraph::IFieldSpan<Tvec>> positions,
            std::shared_ptr<shamrock::solvergraph::IFieldSpan<Tscal>> old_h,
            std::shared_ptr<shamrock::solvergraph::IFieldSpan<Tscal>> new_h,
            std::shared_ptr<shamrock::solvergraph::IFieldSpan<Tscal>> eps_h) {
            __internal_set_ro_edges({sizes, neigh_cache, positions, old_h});
            __internal_set_rw_edges({new_h, eps_h});
        }

        inline Edges get_edges() {
            return Edges{
                get_ro_edge<shamrock::solvergraph::Indexes<u32>>(0),
                get_ro_edge<shammodels::sph::solvergraph::NeighCache>(1),
                get_ro_edge<shamrock::solvergraph::IFieldSpan<Tvec>>(2),
                get_ro_edge<shamrock::solvergraph::IFieldSpan<Tscal>>(3),
                get_rw_edge<shamrock::solvergraph::IFieldSpan<Tscal>>(0),
                get_rw_edge<shamrock::solvergraph::IFieldSpan<Tscal>>(1)};
        }

        void _impl_evaluate_internal();

        inline virtual std::string _impl_get_label() { return "IterateSmoothingLengthDensity"; };

        virtual std::string _impl_get_tex();
    };
} // namespace shammodels::sph::modules
