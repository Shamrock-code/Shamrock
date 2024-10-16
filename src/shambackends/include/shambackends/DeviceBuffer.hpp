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
         * @brief Get the memory alignment of the type T in bytes
         *
         * @return The memory alignment of the type T in bytes
         */
        static std::optional<size_t> get_alignment() { return alignof(T); }

        /**
         * @brief Convert a size in number of elements to a size in bytes
         *
         * @param sz The size in number of elements
         * @return The size in bytes
         */
        static size_t to_bytesize(size_t sz) { return sz * sizeof(T); }

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
            : hold(details::create_usm_ptr<target>(to_bytesize(sz), dev_sched, get_alignment())),
              size(sz) {}

        DeviceBuffer(sycl::buffer<T> &syclbuf, std::shared_ptr<DeviceScheduler> dev_sched)
            : DeviceBuffer(syclbuf.size(), dev_sched) {
            sham::EventList depends_list;
            T *ptr_dest = get_write_access(depends_list);

            sycl::event e = get_queue().submit(depends_list, [&](sycl::handler &cgh) {
                sycl::accessor acc{syclbuf, cgh, sycl::read_only};

                cgh.copy(acc, ptr_dest);
            });

            complete_event_state(e);
        }

        DeviceBuffer(sycl::buffer<T> &&syclbuf, std::shared_ptr<DeviceScheduler> dev_sched)
            : DeviceBuffer(syclbuf, dev_sched) {}

        /**
         * @brief Construct a new Device Buffer object with a given USM pointer
         *
         * @param sz The size of the buffer in number of elements
         * @param _hold A USMPtrHolder holding the USM pointer
         *
         * This constructor is used to create a Device Buffer object with a
         * pre-allocated USM pointer. The size of the buffer is given by the
         * `sz` parameter, and the USM pointer is given by the `_hold` parameter.
         * The constructor forwards the `_hold` parameter to the USMPtrHolder
         * constructor.
         */
        DeviceBuffer(size_t sz, USMPtrHolder<target> &&_hold)
            : hold(std::forward<USMPtrHolder<target>>(_hold)), size(sz) {}

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
            std::swap(hold, other.hold);
            std::swap(events_hndl, other.events_hndl);
            size = other.size;
            return *this;
        }

        /**
         * @brief Destructor for DeviceBuffer
         *
         * This destructor releases the USM pointer and event handler
         * by transfering them back to the memory handler
         */
        ~DeviceBuffer() {
            // This object is empty, it was probably moved
            if (hold.get_raw_ptr() == nullptr && events_hndl.is_empty()) {
                return;
            }
            // give the ptr holder and event handler to the memory handler
            details::release_usm_ptr(std::move(hold), std::move(events_hndl));
        }

        ///////////////////////////////////////////////////////////////////////
        // Event handling
        ///////////////////////////////////////////////////////////////////////

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
        [[nodiscard]] inline const T *get_read_access(sham::EventList &depends_list) {
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
        [[nodiscard]] inline T *get_write_access(sham::EventList &depends_list) {
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

        ///////////////////////////////////////////////////////////////////////
        // Event handling (End)
        ///////////////////////////////////////////////////////////////////////

        ///////////////////////////////////////////////////////////////////////

        ///////////////////////////////////////////////////////////////////////
        // Queue / Scheduler getters
        ///////////////////////////////////////////////////////////////////////

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
         * @brief Gets the DeviceQueue associated with the held allocation
         *
         * @return The DeviceQueue associated with the held allocation
         */
        [[nodiscard]] inline DeviceQueue &get_queue() {
            return hold.get_dev_scheduler().get_queue();
        }

        ///////////////////////////////////////////////////////////////////////
        // Queue / Scheduler getters (END)
        ///////////////////////////////////////////////////////////////////////

        ///////////////////////////////////////////////////////////////////////

        ///////////////////////////////////////////////////////////////////////
        // Size getters
        ///////////////////////////////////////////////////////////////////////

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
        [[nodiscard]] inline size_t get_bytesize() const { return to_bytesize(get_size()); }

        /**
         * @brief Gets the amount of memory used by the buffer
         *
         * @return The amount of memory used by the buffer
         */
        [[nodiscard]] inline size_t get_mem_usage() const { return hold.get_bytesize(); }

        ///////////////////////////////////////////////////////////////////////
        // Size getters (END)
        ///////////////////////////////////////////////////////////////////////

        ///////////////////////////////////////////////////////////////////////

        ///////////////////////////////////////////////////////////////////////
        // Copy fcts
        ///////////////////////////////////////////////////////////////////////

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

            sham::EventList depends_list;
            const T *ptr = get_read_access(depends_list);

            sycl::event e = get_queue().submit(depends_list, [&](sycl::handler &cgh) {
                cgh.copy(ptr, ret.data(), size);
            });

            e.wait_and_throw();
            complete_event_state({});

            return ret;
        }

        /**
         * @brief Copy the content of the buffer to a new SYCL buffer
         *
         * This function creates a new SYCL buffer with the same size and content than the current
         * one and returns it.
         *
         * @return The new SYCL buffer
         */
        [[nodiscard]] inline sycl::buffer<T> copy_to_sycl_buffer() {
            sycl::buffer<T> ret(size);

            sham::EventList depends_list;
            const T *ptr = get_read_access(depends_list);

            sycl::event e = get_queue().submit(depends_list, [&](sycl::handler &cgh) {
                sycl::accessor acc(ret, cgh, sycl::write_only, sycl::no_init);
                cgh.copy(ptr, acc);
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
            DeviceBuffer<T, new_target> ret(size, get_dev_scheduler_ptr());

            sham::EventList depends_list;
            const T *ptr_src = get_read_access(depends_list);
            T *ptr_dest      = ret.get_write_access(depends_list);

            sycl::event e = get_queue().submit(depends_list, [&](sycl::handler &cgh) {
                cgh.copy(ptr_src, ptr_dest, size);
            });

            complete_event_state(e);
            ret.complete_event_state(e);

            return ret;
        }

        /**
         * @brief Copies the content of another buffer to this one
         *
         * This function copies the content of another buffer to this one. The two buffers must have
         * the same size, and the size of the copy must be smaller than the size of the buffer
         * involved.
         *
         * @param other The buffer from which to copy the data
         * @param copy_size The size of the copy
         */
        template<USMKindTarget new_target>
        inline void copy_from(DeviceBuffer<T, new_target> &other, size_t copy_size) {

            if (!(copy_size <= get_size() && copy_size <= other.get_size())) {
                shambase::throw_with_loc<std::invalid_argument>(shambase::format(
                    "The size of the copy must be smaller than the size of the buffer involved\n  "
                    "copy_size: {}\n  get_size(): {}\n  other.get_size(): {}",
                    copy_size,
                    get_size(),
                    other.get_size()));
            }

            sham::EventList depends_list;
            T *ptr_dest      = get_write_access(depends_list);
            const T *ptr_src = other.get_read_access(depends_list);

            sycl::event e = get_queue().submit(depends_list, [&](sycl::handler &cgh) {
                cgh.copy(ptr_src, ptr_dest, copy_size);
            });

            complete_event_state(e);
            other.complete_event_state(e);
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

            if (get_size() != other.get_size()) {
                shambase::throw_with_loc<std::invalid_argument>(shambase::format(
                    "The other field must be of the same size\n  get_size = {},\n  other.get_size "
                    "= {}",
                    get_size(),
                    other.get_size()));
            }

            copy_from(other, get_size());
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

        ///////////////////////////////////////////////////////////////////////
        // Copy fcts (END)
        ///////////////////////////////////////////////////////////////////////

        ///////////////////////////////////////////////////////////////////////

        ///////////////////////////////////////////////////////////////////////
        // Filler fcts
        ///////////////////////////////////////////////////////////////////////

        /**
         * @brief Fill a subpart of the buffer with a given value
         *
         * This function fills a subpart of the buffer with a given value. The subpart is
         * defined by a range of indices, given as a pair `[start_index,idx_count]`. The
         * start index is the first index of the range, and the count is the number of
         * elements to fill.
         *
         * The function checks that the range of indices is valid, i.e. that
         * `start_index + idx_count <= get_size()`.
         *
         * @param value The value to fill the buffer with
         * @param idx_range The range of indices to fill, given as a pair
         * `[start_index,idx_count]`.
         */
        inline void fill(T value, std::array<size_t, 2> idx_range) {

            size_t start_index = idx_range[0];
            size_t idx_count   = idx_range[1] - start_index;

            if (!(start_index + idx_count <= get_size())) {
                shambase::throw_with_loc<std::invalid_argument>(shambase::format(
                    "!(start_index + idx_count <= get_size())\n  start_index = {},\n  idx_count = "
                    "{},\n  get_size() = {}",
                    start_index,
                    idx_count,
                    get_size()));
            }

            sham::EventList depends_list;
            T *ptr = get_write_access(depends_list);

            sycl::event e1 = get_queue().submit(
                depends_list, [&, ptr, value, start_index, idx_count](sycl::handler &cgh) {
                    shambase::parralel_for(cgh, idx_count, "fill field", [=](u32 gid) {
                        ptr[start_index + gid] = value;
                    });
                });

            complete_event_state(e1);
        }

        /**
         * @brief Fill the first `idx_count` elements of the buffer with a given value
         *
         * This function fills the first `idx_count` elements of the buffer with the given
         * value. The function returns immediately, and the filling operation is executed
         * asynchronously.
         *
         * @param value The value to fill the buffer with
         * @param idx_count The number of elements to fill
         */
        inline void fill(T value, size_t idx_count) { fill(value, {0, idx_count}); }

        /**
         * @brief Fill the buffer with a given value.
         *
         * This function fills the buffer with the given value. The function
         * returns immediately, and the filling operation is executed
         * asynchronously.
         *
         * @param value The value to fill the buffer with.
         */
        inline void fill(T value) { fill(value, get_size()); }

        ///////////////////////////////////////////////////////////////////////
        // Filler fcts (END)
        ///////////////////////////////////////////////////////////////////////

        ///////////////////////////////////////////////////////////////////////

        ///////////////////////////////////////////////////////////////////////
        // Size manipulation
        ///////////////////////////////////////////////////////////////////////

        /**
         * @brief Resizes the buffer to a given size.
         *
         * @param new_size The new size of the buffer.
         */
        inline void resize(u32 new_size) {
            if (to_bytesize(new_size) > hold.get_bytesize()) {
                // expand storage

                size_t new_storage_size = to_bytesize(new_size) * 1.5;

                DeviceBuffer new_buf(
                    new_size,
                    details::create_usm_ptr<target>(
                        new_storage_size, get_dev_scheduler_ptr(), get_alignment()));

                // copy data
                new_buf.copy_from(*this, get_size());

                // override old buffer
                std::swap(new_buf, *this);

            } else if (to_bytesize(new_size) < hold.get_bytesize() * 0.5) {
                // shrink storage

                size_t new_storage_size = to_bytesize(new_size);

                DeviceBuffer new_buf(
                    new_size,
                    details::create_usm_ptr<target>(
                        new_storage_size, get_dev_scheduler_ptr(), get_alignment()));

                // copy data
                new_buf.copy_from(*this, new_size);

                // override old buffer
                std::swap(new_buf, *this);
                // *this = std::move(new_buf);
            } else {
                size = new_size;
                // no need to resize
            }
        }

        ///////////////////////////////////////////////////////////////////////
        // Size manipulation (END)
        ///////////////////////////////////////////////////////////////////////

#if false
        // I'm not sure if enabling this one is a good idea
        /**
         * @brief Reserves space in the buffer for `add_sz` elements, but doesn't change the
         * buffer's size.
         *
         * This function is useful when you know you'll need to add `add_sz` elements to the buffer,
         * but you don't want to resize the buffer just yet. After calling this function, you can
         * add `add_sz` elements to the buffer without triggering a resize.
         *
         * @param add_sz The number of elements to reserve space for.
         */
        inline void reserve(size_t add_sz) {
            size_t old_sz = get_size();
            resize(old_sz + add_sz);
            size = old_sz;
        }
#endif

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
