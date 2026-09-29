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
 * @file FindGhostInterfaces.hpp
 * @author Timothée David--Cléris (tim.shamrock@proton.me)
 * @brief Solvergraph nodes finding the SPH ghost interfaces, one node per boundary condition
 *
 * Each node lists, for every local (sender) patch, the patches (receivers) whose interaction
 * volume overlaps the sender, possibly through a periodic / shearing periodic image, along with
 * the metadata required to build the corresponding ghost interface.
 */

#include "shambackends/vec.hpp"
#include "shammath/AABB.hpp"
#include "shammath/CoordRange.hpp"
#include "shammodels/sph/BasicSPHGhosts.hpp"
#include "shamrock/solvergraph/DDSharedScalar.hpp"
#include "shamrock/solvergraph/PatchtreeFieldEdge.hpp"
#include "shamrock/solvergraph/ScalarEdge.hpp"
#include "shamrock/solvergraph/ScalarsEdge.hpp"
#include "shamrock/solvergraph/SerialPatchTreeEdge.hpp"
#include "shamsolvergraph/edge/IDataEdge.hpp"
#include "shamsolvergraph/node/INode.hpp"

/// Edges common to every ghost interface finder
#define FIND_GHOST_INTERFACES_EDGES_IN(X_RO)                                                       \
    X_RO(shamrock::solvergraph::ScalarEdge<shammath::AABB<Tvec>>, sim_box)                         \
    X_RO(shamrock::solvergraph::SerialPatchTreeRefEdge<Tvec>, patch_tree)                          \
    X_RO(shamrock::solvergraph::PatchtreeFieldEdge<Tscal>, interact_radius_tree)                   \
    X_RO(shamrock::solvergraph::ScalarsEdge<Tscal>, interact_radius)                               \
    X_RO(shamrock::solvergraph::ScalarsEdge<shammath::CoordRange<Tvec>>, local_patch_boxes)

#define FIND_GHOST_INTERFACES_EDGES_OUT(X_RW)                                                      \
    X_RW(shamrock::solvergraph::DDSharedScalar<InterfaceBuildInfos>, interface_infos)

namespace shammodels::sph::modules {

    /**
     * @brief Types shared by the ghost interface finders
     *
     * Edges description:
     *  - `sim_box` : bounding box of the simulation domain
     *  - `patch_tree` : the serial patch tree
     *  - `interact_radius_tree` : max of `interact_radius` over each patch tree node
     *  - `interact_radius` : interaction radius of every (global) patch
     *  - `local_patch_boxes` : bounding box of every local patch, i.e. the interface senders
     *  - `interface_infos` (output) : metadata of every interface (sender -> receiver)
     *
     * @tparam Tvec position vector type
     */
    template<class Tvec>
    struct FindGhostInterfacesTypes {
        using Tscal               = shambase::VecComponent<Tvec>;
        using InterfaceBuildInfos = typename BasicSPHGhostHandler<Tvec>::InterfaceBuildInfos;
    };

#define NODE_EDGES(X_RO, X_RW)                                                                     \
    FIND_GHOST_INTERFACES_EDGES_IN(X_RO)                                                           \
    FIND_GHOST_INTERFACES_EDGES_OUT(X_RW)

    /**
     * @brief Ghost interface finder for free boundaries: only direct neighbouring patches
     */
    template<class Tvec>
    class FindGhostInterfacesFree : public shamrock::solvergraph::INode {
        using Tscal               = typename FindGhostInterfacesTypes<Tvec>::Tscal;
        using InterfaceBuildInfos = typename FindGhostInterfacesTypes<Tvec>::InterfaceBuildInfos;

        public:
        FindGhostInterfacesFree() = default;

        EXPAND_NODE_EDGES(NODE_EDGES)

        void _impl_evaluate_internal();

        inline virtual std::string _impl_get_label() const { return "FindGhostInterfacesFree"; };

        virtual std::string _impl_get_tex() const;
    };

    /**
     * @brief Ghost interface finder for periodic boundaries: neighbouring patches, including
     * through the 26 periodic images of the domain
     */
    template<class Tvec>
    class FindGhostInterfacesPeriodic : public shamrock::solvergraph::INode {
        using Tscal               = typename FindGhostInterfacesTypes<Tvec>::Tscal;
        using InterfaceBuildInfos = typename FindGhostInterfacesTypes<Tvec>::InterfaceBuildInfos;

        public:
        FindGhostInterfacesPeriodic() = default;

        EXPAND_NODE_EDGES(NODE_EDGES)

        void _impl_evaluate_internal();

        inline virtual std::string _impl_get_label() const {
            return "FindGhostInterfacesPeriodic";
        };

        virtual std::string _impl_get_tex() const;
    };

#undef NODE_EDGES

#define NODE_EDGES(X_RO, X_RW)                                                                     \
    FIND_GHOST_INTERFACES_EDGES_IN(X_RO)                                                           \
    X_RO(shamrock::solvergraph::IDataEdge<Tscal>, time)                                            \
    FIND_GHOST_INTERFACES_EDGES_OUT(X_RW)

    /**
     * @brief Ghost interface finder for shearing periodic boundaries
     *
     * The periodic images along `shear_base` are shifted by `shear_speed * time` along
     * `shear_dir`, and move at `shear_speed`.
     */
    template<class Tvec>
    class FindGhostInterfacesShearingPeriodic : public shamrock::solvergraph::INode {
        using Tscal               = typename FindGhostInterfacesTypes<Tvec>::Tscal;
        using InterfaceBuildInfos = typename FindGhostInterfacesTypes<Tvec>::InterfaceBuildInfos;

        i32_3 shear_base;
        i32_3 shear_dir;
        Tscal shear_speed;

        public:
        FindGhostInterfacesShearingPeriodic(i32_3 shear_base, i32_3 shear_dir, Tscal shear_speed)
            : shear_base(shear_base), shear_dir(shear_dir), shear_speed(shear_speed) {}

        EXPAND_NODE_EDGES(NODE_EDGES)

        void _impl_evaluate_internal();

        inline virtual std::string _impl_get_label() const {
            return "FindGhostInterfacesShearingPeriodic";
        };

        virtual std::string _impl_get_tex() const;
    };

#undef NODE_EDGES

} // namespace shammodels::sph::modules

#undef FIND_GHOST_INTERFACES_EDGES_IN
#undef FIND_GHOST_INTERFACES_EDGES_OUT
