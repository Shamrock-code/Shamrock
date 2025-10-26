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
 * @file DistributedBuffers.hpp
 * @author Timothée David--Cléris (tim.shamrock@proton.me)
 * @brief Defines the DistributedBuffers class for managing distributed device buffers in a solver
 * graph.
 *
 */

#include "shambase/DistributedData.hpp"
#include "shambackends/DeviceBuffer.hpp"
#include "shamrock/solvergraph/IEdgeNamed.hpp"
#include "shamsys/NodeInstance.hpp"

namespace shamrock::solvergraph {

    /// Alias for a DistributedData of PatchDataFieldSpans
    template<class T>
    using DDDeviceBuffer = shambase::DistributedData<sham::DeviceBuffer<T>>;

    /**
     * @brief Interface for a solver graph edge representing a field as spans.
     *
     * Here a field refer to a field that is distributed over several patches.
     *
     * @tparam T The primitive type of the field
     */
    template<class T>
    class DistributedBuffers : public IEdgeNamed {
        public:
        using IEdgeNamed::IEdgeNamed;

        DDDeviceBuffer<T> buffers;

        inline virtual void free_alloc() { buffers = {}; }

        inline virtual void check_allocated(const std::vector<u64> &ids) const {
            on_distributeddata_ids_diff(
                buffers,
                ids,
                [](u64 id) {
                    shambase::throw_with_loc<std::runtime_error>(
                        "Missing buffer in distributed data at id " + std::to_string(id));
                },
                [](u64 id) {},
                [](u64 id) {
                    shambase::throw_with_loc<std::runtime_error>(
                        "Extra buffer in distributed data at id " + std::to_string(id));
                });
        }

        // overload only the non const case
        inline virtual void ensure_allocated(const std::vector<u64> &ids) {

            auto new_buf = [&]() {
                auto ret = sham::DeviceBuffer<T>(0, shamsys::instance::get_compute_scheduler_ptr());
                return ret;
            };

            on_distributeddata_ids_diff(
                buffers,
                ids,
                [&](u64 id) {
                    buffers.add_obj(id, new_buf());
                },
                [](u64 id) {
                    // Nothing for now
                },
                [&](u64 id) {
                    buffers.erase(id);
                });
        }
    };

} // namespace shamrock::solvergraph
