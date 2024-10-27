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

            auto e = q.submit([&](sycl::handler &h) {
                fct(h);
            });

            if (wait_after_submit) {
                e.wait_and_throw();
            }

            return e;
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

            elist.consumed = true;

            auto e = q.submit([&](sycl::handler &h) {
                elist.apply_dependancy(h);
                fct(h);
            });

            if (wait_after_submit) {
                e.wait_and_throw();
            }

            return e;
        }
    };

} // namespace sham
