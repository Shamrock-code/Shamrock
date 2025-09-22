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
 * @file DDSharedBuffers.hpp
 * @author Timothée David--Cléris (tim.shamrock@proton.me)
 * @brief Defines the DDSharedBuffers class for managing buffers contained in a distributed data
 * shared.
 *
 */

#include "shambase/DistributedDataShared.hpp"
#include "shambackends/DeviceBuffer.hpp"
#include "shamrock/solvergraph/IEdgeNamed.hpp"
#include "shamsys/NodeInstance.hpp"

namespace shamrock::solvergraph {

    /// Alias for a DistributedDataShared of DeviceBuffer
    template<class T>
    using DDSharedDeviceBuffer = shambase::DistributedDataShared<sham::DeviceBuffer<T>>;

    /**
     * @brief Interface for a solver graph edge representing a buffer contained in a distributed
     * data shared.
     *
     * @tparam T The primitive type of the buffer
     */
    template<class T>
    class DDSharedBuffers : public IEdgeNamed {
        public:
        using IEdgeNamed::IEdgeNamed;

        DDSharedDeviceBuffer<T> buffers;

        inline virtual void free_alloc() { buffers = {}; }
    };

} // namespace shamrock::solvergraph
