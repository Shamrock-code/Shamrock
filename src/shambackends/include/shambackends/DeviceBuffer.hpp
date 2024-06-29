// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright(C) 2021-2023 Timothée David--Cléris <timothee.david--cleris@ens-lyon.fr>
// Licensed under CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

#pragma once

/**
 * @file DeviceBuffer.hpp
 * @author Timothée David--Cléris (timothee.david--cleris@ens-lyon.fr)
 * @brief
 *
 */
 
#include "shambackends/DeviceScheduler.hpp"
#include "shambackends/USMPtrHolder.hpp"
#include "shambackends/details/BufferEventHandler.hpp"
#include "shambackends/details/memoryHandle.hpp"
#include <memory>

namespace sham {

    /**
     * @brief A buffer allocated in USM (Unified Shared Memory)
     *
     * @tparam T The type of the buffer's elements
     * @tparam target The USM target where the buffer is allocated (host, device, shared)
     */
    template<class T, USMKindTarget target = device>
    class DeviceBuffer{
        
        public:

        /**
         * @brief Construct a new Device Buffer object
         *
         * @param sz The size of the buffer in number of elements
         * @param dev_sched A shared pointer to the Device Scheduler
         *
         * This constructor creates a new Device Buffer object with the given size.
         * It allocates the buffer as USM memory and stores the USM pointer and the
         * size in the respective member variables.
         */
        DeviceBuffer(size_t sz, std::shared_ptr<DeviceScheduler> dev_sched)
            : hold(details::create_usm_ptr<target>(sz*sizeof(T), dev_sched))     // Create USM pointer
            , size(sz)                                                           // Store size
        {}

        /**
         * @brief Deleted copy constructor
         */
        DeviceBuffer(const DeviceBuffer& other) = delete;

        /**
         * @brief Deleted copy assignment operator
         */
        DeviceBuffer& operator=(const DeviceBuffer& other) = delete;

        /**
         * @brief Destructor for DeviceBuffer
         *
         * This destructor releases the USM pointer and event handler
         * by transfering them back to the memory handler
         */
        ~DeviceBuffer(){
            //give the ptr holder and event handler to the memory handler
            details::release_usm_ptr(std::move(hold), std::move(events_hndl));
        }

        /**
         * @brief Get a read-only pointer to the buffer's data.
         *
         * This function returns a const pointer to the buffer's data. The
         * pointer is locked for reading and the event handler is updated to
         * reflect the read access.
         *
         * @param depends_list A vector of SYCL events to wait for before
         *        accessing the buffer.
         * @return A const pointer to the buffer's data.
         */
        [[nodiscard]] inline const T * get_read_access(std::vector<sycl::event> &depends_list) {
            events_hndl.read_access(depends_list);
            return hold.template ptr_cast<T>();
        }

        /**
         * @brief Get a read-write pointer to the buffer's data
         *
         * This function returns a pointer to the buffer's data. The event handler is updated to
         * reflect the write access.
         *
         * @param depends_list A vector of SYCL events to wait for before
         *        accessing the buffer.
         * @return A pointer to the buffer's data.
         */
        [[nodiscard]] inline T * get_write_access(std::vector<sycl::event> &depends_list) {
            events_hndl.write_access(depends_list);
            return hold.template ptr_cast<T>();
        }

        /**
         * @brief Complete the event state of the buffer.
         *
         * This function complete the event state of the buffer by registering the 
         * event resulting of the last queried access 
         *
         * @param e The SYCL event resulting of the queried access.
         */
        void complete_event_state(sycl::event e){
            events_hndl.complete_state(e);
        }

        /**
         * @brief Gets the Device scheduler corresponding to the held allocation
         *
         * @return The Device scheduler
         */
        [[nodiscard]] inline DeviceScheduler& get_dev_scheduler() const {
            return hold.get_dev_scheduler();
        }

        /**
         * @brief Gets the number of elements in the buffer
         *
         * @return The number of elements in the buffer
         */
        [[nodiscard]] inline size_t get_size() const {
            return size;
        }

        /**
         * @brief Gets the size of the buffer in bytes
         *
         * @return The size of the buffer in bytes
         */
        [[nodiscard]] inline size_t get_bytesize() const {
            return hold.get_bytesize();
        }

        private:

        /**
         * @brief The USM pointer holder
         */
        USMPtrHolder<target> hold;

        /**
         * @brief The number of elements in the buffer
         */
        size_t size = 0;

        /**
         * @brief Event handler for the buffer
         *
         * This event handler keeps track of the events associated with read and write
         * accesses to the buffer. It is used to ensure that the buffer is not accessed
         * before the data is in a complete state.
         */
        details::BufferEventHandler events_hndl;

    };

} // namespace sham