// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright(C) 2021-2023 Timothée David--Cléris <timothee.david--cleris@ens-lyon.fr>
// Licensed under CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

#pragma once

/**
 * @file DeviceScheduler.hpp
 * @author Timothée David--Cléris (timothee.david--cleris@ens-lyon.fr)
 * @brief
 */

#include "shambackends/DeviceQueue.hpp"

namespace sham {

    /**
     * @brief Class to manage the scheduling of kernels on a device
     *
     * The DeviceScheduler is responsible for scheduling kernels on the device.
     * It contains a set of DeviceQueues.
     */
    class DeviceScheduler {
        public:
        /**
         * @brief Reference to the device context associated with this
         * DeviceScheduler
         */
        std::shared_ptr<DeviceContext> ctx;

        /**
         * @brief Vector of unique pointers to the DeviceQueues associated with
         * this DeviceScheduler
         */
        std::vector<std::unique_ptr<DeviceQueue>> queues;

        /**
         * @brief Constructor
         * @param ctx Reference to the device context associated with this
         * DeviceScheduler
         */
        explicit DeviceScheduler(std::shared_ptr<DeviceContext> ctx);

        /**
         * @brief Get a reference to a DeviceQueue
         * @param id Id of the DeviceQueue to retrieve
         * @return Reference to the requested DeviceQueue
         */
        DeviceQueue &get_queue(u32 id = 0);

        /**
         * @brief Print information about the DeviceScheduler
         */
        void print_info();

        /**
         * @brief Function to test that all the queues are working properly
         */
        void test();

        /**
         * @brief Check if the context corresponding to the device scheduler should use direct
         * communication
         *
         * This method returns true if the context should use direct
         * communication, false otherwise.
         *
         * @return true if direct communication should be used
         */
        bool use_direct_comm();
    };

    using DeviceScheduler_ptr = std::shared_ptr<DeviceScheduler>;

} // namespace sham
