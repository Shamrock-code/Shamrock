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
 * @file PatchDataLayerDDShared.hpp
 * @author Timothée David--Cléris (tim.shamrock@proton.me)
 * @brief Shared distributed data layer for patch data management in solver graphs
 *
 * This header defines the PatchDataLayerDDShared class, which provides a shared
 * distributed data interface for patch data layers within solver graph structures.
 * It extends IDataEdgeNamed to provide named data edge functionality while
 * managing shared patch data layers across distributed systems.
 *
 * The class is designed to work with the solver graph architecture, allowing
 * efficient sharing and management of patch data across multiple computational
 * nodes or processes.
 */

#include "shambase/DistributedDataShared.hpp"
#include "shamrock/patch/PatchDataLayer.hpp"
#include "shamrock/solvergraph/IEdgeNamed.hpp"

namespace shamrock::solvergraph {

    /**
     * @brief Shared distributed data layer for patch data management
     *
     * PatchDataLayerDDShared provides a shared distributed data interface for
     * managing patch data layers within solver graph structures. It extends
     * IDataEdgeNamed to provide named data edge functionality while maintaining
     * shared patch data layers across distributed systems.
     *
     * This class is particularly useful in scenarios where multiple solver
     * components need access to the same patch data layer, ensuring efficient
     * memory usage and data consistency across distributed computations.
     *
     * @code{.cpp}
     * // Create a shared distributed data layer
     * auto sharedLayer = std::make_unique<PatchDataLayerDDShared>("label","tex symbol");
     *
     * // Access the shared patch data
     * auto& patchData = sharedLayer->patchdatas;
     *
     * // Free allocated resources when done
     * sharedLayer->free_alloc();
     * @endcode
     */
    class PatchDataLayerDDShared : public IEdgeNamed {

        public:
        /**
         * @brief Shared distributed data containing patch data layers
         *
         * This member provides access to shared distributed data that contains
         * patch data layers. The data is shared across multiple processes or
         * computational nodes, allowing efficient access to patch information
         * without unnecessary data duplication.
         */
        shambase::DistributedDataShared<patch::PatchDataLayer> patchdatas;

        /// Inherit constructors from IDataEdgeNamed
        using IEdgeNamed::IEdgeNamed;

        /// Free allocated resources
        inline virtual void free_alloc() { patchdatas = {}; }
    };

} // namespace shamrock::solvergraph
