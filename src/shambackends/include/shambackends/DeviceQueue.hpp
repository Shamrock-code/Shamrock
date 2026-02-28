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
 * @file DeviceQueue.hpp
 * @author Timothée David--Cléris (tim.shamrock@proton.me)
 * @brief
 */

#include "shambase/memory.hpp"
#include "shambackends/Device.hpp"
#include "shambackends/DeviceContext.hpp"
#include "shambackends/EventList.hpp"

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
         * @brief Whether to wait for the kernel to finish after submitting it
         *
         * This is true if the queue should wait for the kernel to finish after
         * submitting it, false otherwise.
         *
         * This is never enabled by default but is set on based on the env flag
         * `SHAMROCK_WAIT_AFTER_SUBMIT`.
         */
        bool wait_after_submit = false;

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

        /**
         * @brief Submits a kernel to the SYCL queue
         *
         * This function submits a kernel, encapsulated within the provided
         * functor, to the SYCL queue for execution. The functor is expected to
         * take a sycl::handler as its argument and use it to define the kernel
         * operations.
         *
         * @tparam Fct The type of the functor
         * @param fct The functor that defines the kernel to be executed
         * @return sycl::event The event associated with the kernel execution
         *
         * If the `wait_after_submit` flag is true, this function will wait for
         * the kernel to finish and will throw any exceptions that occur during
         * execution.
         */
        template<class Fct>
        sycl::event submit(Fct &&fct) {

            __shamrock_stack_entry();

            wait_and_throw();

            auto e = q.submit([&](sycl::handler &h) {
                fct(h);
            });

            if (wait_after_submit) {
                __shamrock_stack_entry();
                e.wait_and_throw();
            }

            return e;
        }

        /**
         * @brief Wait for the queue to finish and throw an exception if one has occurred
         *
         * This function waits for the queue to finish and throws an exception if one has occurred.
         */
        void wait_and_throw() {
            __shamrock_stack_entry();
            q.wait_and_throw();
        }

        /**
         * @brief Submits a kernel to the SYCL queue, adding the events in the
         * provided EventList as dependencies
         *
         * This function submits a kernel, encapsulated within the provided
         * functor, to the SYCL queue for execution. The functor is expected to
         * take a sycl::handler as its argument and use it to define the kernel
         * operations.
         *
         * The events in the EventList are added as dependencies to the
         * submitted kernel. This allows to create a dependency graph between
         * kernels and events, which can be used to coordinate the execution of
         * kernels.
         *
         * @tparam Fct The type of the functor
         * @param elist The EventList containing the events to be added as
         * dependencies
         * @param fct The functor that defines the kernel to be executed
         * @return sycl::event The event associated with the kernel execution
         *
         * If the `wait_after_submit` flag is true, this function will wait for
         * the kernel to finish and will throw any exceptions that occur during
         * execution.
         */
        template<class Fct>
        sycl::event submit(EventList &elist, Fct &&fct) {

            __shamrock_stack_entry();

            wait_and_throw();

            elist.consumed = true;
            auto e         = q.submit([&](sycl::handler &h) {
                elist.apply_dependancy(h);
                fct(h);
            });

            if (wait_after_submit) {
                __shamrock_stack_entry();
                e.wait_and_throw();
            }

            return e;
        }

        /**
         * @brief Retrieves the properties of the associated device
         *
         * This function returns the properties of the device associated with
         * the current device context. It fetches the device properties from
         * the context reference and provides details such as vendor, backend,
         * memory capacity, and other device-specific characteristics.
         *
         * @return DeviceProperties The properties of the associated device
         */
        inline DeviceProperties &get_device_prop() {
            return shambase::get_check_ref(ctx).device->prop;
        }
    };

} // namespace sham
