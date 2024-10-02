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
#include "shambackends/sycl_utils.hpp"
#include <memory>

namespace sham {

    /**
     * @brief A buffer allocated in USM (Unified Shared Memory)
     *
     * @tparam T The type of the buffer's elements
     * @tparam target The USM target where the buffer is allocated (host, device, shared)
     */
    template<class T, USMKindTarget target = device>
    class DeviceBuffer {

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
            : hold(details::create_usm_ptr<target>(sz * sizeof(T), dev_sched)), size(sz) {}

        /**
         * @brief Deleted copy constructor
         */
        DeviceBuffer(const DeviceBuffer &other) = delete;

        /**
         * @brief Deleted copy assignment operator
         */
        DeviceBuffer &operator=(const DeviceBuffer &other) = delete;

        /**
         * @brief Move constructor for DeviceBuffer
         *
         * This move constructor moves the USM pointer and the event handler
         * from the other object to this object.
         */
        DeviceBuffer(DeviceBuffer &&other) noexcept
            : hold(std::move(other.hold)), size(other.size),
              events_hndl(std::move(other.events_hndl)) {}

        /**
         * @brief Move assignment operator for DeviceBuffer
         *
         * This move assignment operator moves the USM pointer and the event handler
         * from the other object to this object.
         */
        DeviceBuffer &operator=(DeviceBuffer &&other) noexcept {
            hold        = std::move(other.hold);
            size        = other.size;
            events_hndl = std::move(other.events_hndl);
            return *this;
        }

        /**
         * @brief Destructor for DeviceBuffer
         *
         * This destructor releases the USM pointer and event handler
         * by transfering them back to the memory handler
         */
        ~DeviceBuffer() {
            // give the ptr holder and event handler to the memory handler
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
        [[nodiscard]] inline const T *get_read_access(std::vector<sycl::event> &depends_list) {
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
        [[nodiscard]] inline T *get_write_access(std::vector<sycl::event> &depends_list) {
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
        void complete_event_state(sycl::event e) { events_hndl.complete_state(e); }

        /**
         * @brief Gets the Device scheduler corresponding to the held allocation
         *
         * @return The Device scheduler
         */
        [[nodiscard]] inline DeviceScheduler &get_dev_scheduler() const {
            return hold.get_dev_scheduler();
        }

        /**
         * @brief Gets the Device scheduler pointer corresponding to the held allocation
         *
         * @return The Device scheduler
         */
        [[nodiscard]] inline std::shared_ptr<DeviceScheduler> &get_dev_scheduler_ptr() {
            return hold.get_dev_scheduler_ptr();
        }

        /**
         * @brief Gets the number of elements in the buffer
         *
         * @return The number of elements in the buffer
         */
        [[nodiscard]] inline size_t get_size() const { return size; }

        /**
         * @brief Gets the size of the buffer in bytes
         *
         * @return The size of the buffer in bytes
         */
        [[nodiscard]] inline size_t get_bytesize() const { return hold.get_bytesize(); }

        /**
         * @brief Copy the content of the buffer to a std::vector
         *
         * This function creates a new std::vector with the same size and content than the current
         * one and returns it.
         *
         * @return The new std::vector
         */
        [[nodiscard]] inline std::vector<T> copy_to_stdvec() {
            std::vector<T> ret(size);

            std::vector<sycl::event> depends_list;
            const T *ptr = get_read_access(depends_list);

            sycl::event e = hold.get_dev_scheduler().get_queue().q.submit([&](sycl::handler &cgh) {
                cgh.depends_on(depends_list);
                cgh.copy(ptr, ret.data(), size);
            });

            complete_event_state(e);

            return ret;
        }

        /**
         * @brief Copy the content of the buffer to a new buffer with a different USM target
         *
         * This function creates a new buffer with the same size and content than the current one
         * but with a different USM target. The new buffer is returned.
         *
         * @return The new buffer
         */
        template<USMKindTarget new_target>
        [[nodiscard]] inline DeviceBuffer<T, new_target> copy_to() {
            DeviceBuffer<T, new_target> ret(size, hold.get_dev_scheduler_ptr());

            std::vector<sycl::event> depends_list;
            const T *ptr_src = get_read_access(depends_list);
            T *ptr_dest      = ret.get_write_access(depends_list);

            sycl::event e = hold.get_dev_scheduler().get_queue().q.submit([&](sycl::handler &cgh) {
                cgh.depends_on(depends_list);
                cgh.copy(ptr_src, ptr_dest, size);
            });

            complete_event_state(e);
            ret.complete_event_state(e);

            return ret;
        }

        /**
         * @brief Copies the data from another buffer to this one
         *
         * This function copies the data from another buffer to this one. The
         * two buffers must have the same size.
         *
         * @param other The buffer from which to copy the data
         */
        template<USMKindTarget new_target>
        inline void copy_from(DeviceBuffer<T, new_target> &other) {

            if (other.get_size() != get_size()) {
                shambase::throw_with_loc<std::invalid_argument>(
                    "The two fields must have the same size");
            }

            std::vector<sycl::event> depends_list;
            T *ptr_src        = get_write_access(depends_list);
            const T *ptr_dest = other.get_read_access(depends_list);

            sycl::event e = hold.get_dev_scheduler().get_queue().q.submit([&](sycl::handler &cgh) {
                cgh.depends_on(depends_list);
                cgh.copy(ptr_dest, ptr_src, size);
            });

            complete_event_state(e);
            other.complete_event_state(e);
        }

        /**
         * @brief Fill the buffer with a given value.
         *
         * This function fills the buffer with the given value. The function
         * returns immediately, and the filling operation is executed
         * asynchronously.
         *
         * @param value The value to fill the buffer with.
         */
        inline void fill(T value) {
            std::vector<sycl::event> depends_list;
            T *ptr = get_write_access(depends_list);

            sycl::event e1 = hold.get_dev_scheduler().get_queue().q.submit(
                [&, ptr, value](sycl::handler &cgh) {
                    cgh.depends_on(depends_list);
                    shambase::parralel_for(cgh, size, "fill field", [=](u32 gid) {
                        ptr[gid] = value;
                    });
                });
            complete_event_state(e1);
        }

        /**
         * @brief Copy the current buffer
         *
         * This function creates a new buffer of the same type and size as the current one,
         * and copies the content of the current buffer to the new one.
         *
         * @return The new buffer.
         */
        inline DeviceBuffer<T, target> copy() { return copy_to<target>(); }

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
