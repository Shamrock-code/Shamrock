// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright(C) 2021-2023 Timothée David--Cléris <timothee.david--cleris@ens-lyon.fr>
// Licensed under CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

#pragma once

/**
 * @file DeviceQueue.hpp
 * @author Timothée David--Cléris (timothee.david--cleris@ens-lyon.fr)
 * @brief
 */

#include "shambackends/DeviceContext.hpp"

namespace sham {

    /**
     * @brief A SYCL queue associated with a device and a context
     *
     * This class represents a SYCL queue. The queue is used to schedule
     * kernels on the device and to transfer data between the host and the
     * device.
     */
    class DeviceQueue {
        public:

        /**
         * @brief The device context of this queue
         */
        std::shared_ptr<DeviceContext> ctx;

        /**
         * @brief The SYCL queue associated with this context
         */
        sycl::queue q;

        /**
         * @brief The name of this queue
         *
         * This is the name of this queue. It is used to identify the queue in logs
         * and debug messages.
         */
        std::string queue_name;

        /**
         * @brief Whether the queue is in order
         *
         * This is true if the queue is in order, false otherwise. If the queue is
         * in order, the kernels are executed in the order they are enqueued,
         * otherwise the kernels are executed using a DAG.
         */
        bool in_order;

        /**
         * @brief Test if the queueus working properly
         *
         * Enqueue a simple kernel to test that the queue can execute something.
         */
        void test();

        /**
         * @brief Create a device queue
         *
         * @param queue_name The name of the queue
         * @param ctx The device context of the queue
         * @param in_order Whether the queue is in order
         *
         * Create a device queue with the given name, device context and order
         * property.
         */
        DeviceQueue(std::string queue_name, std::shared_ptr<DeviceContext> ctx, bool in_order);

    };

} // namespace sham