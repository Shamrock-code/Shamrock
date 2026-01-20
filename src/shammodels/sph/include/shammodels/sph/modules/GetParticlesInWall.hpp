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
 * @file GetParticlesInWall.hpp
 * @author Yona Lapeyre (yona.lapeyre@ens-lyon.fr)
 * @brief Declares the GetParticlesInWall module for disabling particles update.
 *
 */

#include "shamrock/solvergraph/DistributedBuffers.hpp"
#include "shamrock/solvergraph/IFieldRefs.hpp"
#include "shamrock/solvergraph/INode.hpp"
#include "shamrock/solvergraph/PatchDataLayerRefs.hpp"

namespace shammodels::sph::modules {

    template<typename Tvec>
    class GetParticlesInWall : public shamrock::solvergraph::INode {

        using Tscal = shambase::VecComponent<Tvec>;

        Tvec wall_pos;
        Tscal wall_thickness;

        public:
        GetParticlesInWall(const Tvec &wall_pos, Tscal wall_thickness)
            : wall_pos(wall_pos), wall_thickness(wall_thickness) {}

        struct Edges {
            const shamrock::solvergraph::IFieldRefs<Tvec> &pos;
            shamrock::solvergraph::IFieldSpan<u32> &part_ids_in_wall;
        };

        inline void set_edges(
            std::shared_ptr<shamrock::solvergraph::IFieldRefs<Tvec>> pos,
            std::shared_ptr<shamrock::solvergraph::IFieldSpan<u32>> part_ids_in_wall) {
            __internal_set_ro_edges({pos});
            __internal_set_rw_edges({part_ids_in_wall});
        }

        inline Edges get_edges() {
            return Edges{
                get_ro_edge<shamrock::solvergraph::IFieldRefs<Tvec>>(0),
                get_rw_edge<shamrock::solvergraph::IFieldSpan<u32>>(1)};
        }

        void _impl_evaluate_internal();

        inline virtual std::string _impl_get_label() const { return "GetParticlesInWall"; };

        virtual std::string _impl_get_tex() const;
    };
} // namespace shammodels::sph::modules
