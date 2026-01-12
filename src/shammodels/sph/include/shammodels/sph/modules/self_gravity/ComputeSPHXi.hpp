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
 * @file ComputeSPHXi.hpp
 * @author Timothée David--Cléris (tim.shamrock@proton.me)
 * @brief
 *
 */

#include "shambase/exception.hpp"
#include "shambase/integer.hpp"
#include "shambase/stacktrace.hpp"
#include "shambackends/kernel_call.hpp"
#include "shamcomm/logs.hpp"
#include "shammodels/sph/solvergraph/NeighCache.hpp"
#include "shamrock/patch/PatchDataField.hpp"
#include "shamrock/solvergraph/FieldRefs.hpp"
#include "shamrock/solvergraph/IDataEdge.hpp"
#include "shamrock/solvergraph/INode.hpp"
#include "shamrock/solvergraph/Indexes.hpp"
#include <sycl/sycl.hpp>

namespace shammodels::sph::modules {

    /**
     * @brief Compute the varying softening length correction for the self-gravity contribution.
     *
     * \f$ \xi_a = (\partial h_a / \partial \rho_a)  \sum_b m_b \partial \phi_ab(h_a) / \partial
     * h_a\f$
     */
    template<class Tvec, template<class> class SPHKernel>
    class ComputeSPHXi : public shamrock::solvergraph::INode {

        using Tscal = shambase::VecComponent<Tvec>;

        static constexpr Tscal kernel_radius = SPHKernel<Tscal>::Rkern;
        Tscal part_mass;

        public:
        ComputeSPHXi(Tscal part_mass) : part_mass(part_mass) {}

        struct Edges {
            const shamrock::solvergraph::Indexes<u32> &part_counts;
            const shammodels::sph::solvergraph::NeighCache &neigh_cache;
            const shamrock::solvergraph::IFieldSpan<Tvec> &xyz;
            const shamrock::solvergraph::IFieldSpan<Tscal> &hpart;
            shamrock::solvergraph::IFieldSpan<Tscal> &xi;
        };

        inline void set_edges(
            std::shared_ptr<shamrock::solvergraph::Indexes<u32>> part_counts,
            std::shared_ptr<shammodels::sph::solvergraph::NeighCache> neigh_cache,
            std::shared_ptr<shamrock::solvergraph::IFieldSpan<Tvec>> xyz,
            std::shared_ptr<shamrock::solvergraph::IFieldSpan<Tscal>> hpart,
            std::shared_ptr<shamrock::solvergraph::IFieldSpan<Tscal>> xi) {
            __internal_set_ro_edges({part_counts, neigh_cache, xyz, hpart});
            __internal_set_rw_edges({xi});
        }

        inline Edges get_edges() {
            return Edges{
                get_ro_edge<shamrock::solvergraph::Indexes<u32>>(0),
                get_ro_edge<shammodels::sph::solvergraph::NeighCache>(1),
                get_ro_edge<shamrock::solvergraph::IFieldSpan<Tvec>>(2),
                get_ro_edge<shamrock::solvergraph::IFieldSpan<Tscal>>(3),
                get_rw_edge<shamrock::solvergraph::IFieldSpan<Tscal>>(0),
            };
        }

        void _impl_evaluate_internal();

        inline virtual std::string _impl_get_label() const { return "ComputeSPHXi"; };

        virtual std::string _impl_get_tex() const;
    };

} // namespace shammodels::sph::modules
