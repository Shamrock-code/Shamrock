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
 * @file SGSPHCorrection.hpp
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
     * \f$ - \frac{G}{2} \sum_b m_b \left[ \frac{\xi_a}{\Omega_a} \nabla_a W_{ab}\(h_a) +
     * \frac{\xi_b}{\Omega_b} \nabla_a W_{ab}\(h_b)right] \f$
     */
    template<class Tvec, template<class> class SPHKernel>
    class SGSPHCorrection : public shamrock::solvergraph::INode {

        using Tscal  = shambase::VecComponent<Tvec>;
        using Kernel = SPHKernel<Tscal>;

        public:
        explicit SGSPHCorrection() {}

        struct Edges {
            const shamrock::solvergraph::Indexes<u32> &part_counts;
            const shamrock::solvergraph::IDataEdge<Tscal> &gpart_mass;
            const shamrock::solvergraph::IDataEdge<Tscal> &constant_G;
            const shammodels::sph::solvergraph::NeighCache &neigh_cache;
            const shamrock::solvergraph::IFieldSpan<Tvec> &xyz;
            const shamrock::solvergraph::IFieldSpan<Tscal> &hpart;
            const shamrock::solvergraph::IFieldSpan<Tscal> &omega;
            const shamrock::solvergraph::IFieldSpan<Tscal> &xi;
            shamrock::solvergraph::IFieldSpan<Tvec> &add_to_force;
        };

        inline void set_edges(
            std::shared_ptr<shamrock::solvergraph::Indexes<u32>> part_counts,
            std::shared_ptr<shamrock::solvergraph::IDataEdge<Tscal>> gpart_mass,
            std::shared_ptr<shamrock::solvergraph::IDataEdge<Tscal>> constant_G,
            std::shared_ptr<shammodels::sph::solvergraph::NeighCache> neigh_cache,
            std::shared_ptr<shamrock::solvergraph::IFieldSpan<Tvec>> xyz,
            std::shared_ptr<shamrock::solvergraph::IFieldSpan<Tscal>> hpart,
            std::shared_ptr<shamrock::solvergraph::IFieldSpan<Tscal>> omega,
            std::shared_ptr<shamrock::solvergraph::IFieldSpan<Tscal>> xi,
            std::shared_ptr<shamrock::solvergraph::IFieldSpan<Tvec>> add_to_force) {
            __internal_set_ro_edges({part_counts, gpart_mass, neigh_cache, xyz, hpart, omega, xi});
            __internal_set_rw_edges({add_to_force});
        }

        inline Edges get_edges() {
            return Edges{
                get_ro_edge<shamrock::solvergraph::Indexes<u32>>(0),
                get_ro_edge<shamrock::solvergraph::IDataEdge<Tscal>>(1),
                get_ro_edge<shamrock::solvergraph::IDataEdge<Tscal>>(2),
                get_ro_edge<shammodels::sph::solvergraph::NeighCache>(3),
                get_ro_edge<shamrock::solvergraph::IFieldSpan<Tvec>>(4),
                get_ro_edge<shamrock::solvergraph::IFieldSpan<Tscal>>(5),
                get_ro_edge<shamrock::solvergraph::IFieldSpan<Tscal>>(6),
                get_ro_edge<shamrock::solvergraph::IFieldSpan<Tscal>>(7),
                get_rw_edge<shamrock::solvergraph::IFieldSpan<Tvec>>(0)};
        }

        inline std::string _impl_get_label() const override { return "SGSPHCorrection"; }
        std::string _impl_get_tex() const override { return "TODO"; }

        protected:
        void _impl_evaluate_internal() override;
    };

} // namespace shammodels::sph::modules
