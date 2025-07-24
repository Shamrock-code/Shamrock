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
 * @file GetParticlesOutsideSphere.hpp
 * @author Timothée David--Cléris (tim.shamrock@proton.me)
 * @brief Declares the GetParticlesOutsideSphere module for removing particles.
 *
 */

#include "shamrock/solvergraph/DistributedBuffers.hpp"
#include "shamrock/solvergraph/IFieldRefs.hpp"
#include "shamrock/solvergraph/INode.hpp"
#include "shamrock/solvergraph/PatchDataLayerRefs.hpp"

namespace shammodels::sph::modules {

    template<typename Tvec>
    class GetParticlesOutsideSphere : public shamrock::solvergraph::INode {

        using Tscal = shambase::VecComponent<Tvec>;

        Tvec sphere_center;
        Tscal sphere_radius;

        public:
        GetParticlesOutsideSphere(const Tvec& sphere_center, Tscal sphere_radius)
            : sphere_center(sphere_center), sphere_radius(sphere_radius) {}

        struct Edges {
            const shamrock::solvergraph::IFieldRefs<Tvec> &pos;
            shamrock::solvergraph::DistributedBuffers<u32> &part_ids_outside_sphere;
        };

        inline void set_edges(
            std::shared_ptr<shamrock::solvergraph::IFieldRefs<Tvec>> pos,
            std::shared_ptr<shamrock::solvergraph::DistributedBuffers<u32>>
                part_ids_outside_sphere) {
            __internal_set_ro_edges({pos});
            __internal_set_rw_edges({part_ids_outside_sphere});
        }

        inline Edges get_edges() {
            return Edges{
                get_ro_edge<shamrock::solvergraph::IFieldRefs<Tvec>>(0),
                get_rw_edge<shamrock::solvergraph::DistributedBuffers<u32>>(0)};
        }

        void _impl_evaluate_internal();

        inline virtual std::string _impl_get_label() { return "GetParticlesOutsideSphere"; };

        virtual std::string _impl_get_tex();
    };
} // namespace shammodels::sph::modules
