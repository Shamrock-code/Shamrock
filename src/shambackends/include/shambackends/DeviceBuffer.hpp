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
 * @file DeviceBuffer.hpp
 * @author Timothée David--Cléris (tim.shamrock@proton.me)
 * @brief
 *
 */

#include "shambase/assert.hpp"
#include "shambase/memory.hpp"
#include "shambackends/DeviceScheduler.hpp"
#include "shambackends/USMPtrHolder.hpp"
#include "shambackends/details/BufferEventHandler.hpp"
#include "shambackends/details/memoryHandle.hpp"
#include "shambackends/sycl_utils.hpp"
#include <memory>

namespace sham {

    template<class T, USMKindTarget target = host, USMKindTarget orgin_target = device>
    class BufferMirror;

    /**
     * @brief A buffer allocated in USM (Unified Shared Memory)
     *
     * @tparam T The type of the buffer's elements
     * @tparam target The USM target where the buffer is allocated (host, device, shared)
     */
    template<class T, USMKindTarget target = device>
    class DeviceBuffer {

        /**
         * @brief Upgrade the size to the next multiple of mult
         *
         * This function takes a size in bytes and a multiple, and returns the
         * next multiple of the given multiple.
         *
         * @param sz The size in bytes
         * @param mult The multiple
         * @return The upgraded size
         */
        static size_t upgrade_multiple(size_t sz, size_t mult) {
            if (sz % mult)
                return sz + (mult - sz % mult);
            return sz;
        };

        public:
        /**
         * @brief Get the memory alignment of the type T in bytes
         *
         * @return The memory alignment of the type T in bytes
         */
        static std::optional<size_t> get_alignment(DeviceScheduler_ptr dev_sched) {
            return upgrade_multiple(
                alignof(T),
                shambase::get_check_ref(dev_sched)
                    .get_queue()
                    .get_device_prop()
                    .mem_base_addr_align);
        }

        /**
         * @brief Convert a size in number of elements to a size in bytes
         *
         * @param sz The size in number of elements
         * @return The size in bytes
         */
        static size_t alloc_request_size_fct(size_t sz, DeviceScheduler_ptr dev_sched) {
            size_t ret = sz * sizeof(T);

            auto align = get_alignment(dev_sched);
            if (align) {
                ret = upgrade_multiple(ret, *align);
                SHAM_ASSERT(ret % *align == 0);
            }

            SHAM_ASSERT((sz == 0) ? (ret == 0) : (ret >= sz * sizeof(T)));

            return ret;
        }

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
            : hold(std::forward<USMPtrHolder<target>>(_hold)), size(sz),
              events_hndl(std::make_unique<details::BufferEventHandler>()) {}

        /**
         * @brief Construct a new Device Buffer object
         *
         * @param sz The size of the buffer in number of elements
         * @param dev_sched A shared pointer to the Device Scheduler
         *
         * This constructor creates a new Device Buffer object with the given size.
         * It allocates the buffer as USM memory and stores the USM pointer and the
         * size in the respective member variables. The constructor also creates a
         * BufferEventHandler object and stores it in the `events_hndl` member
         * variable.
         */
        DeviceBuffer(size_t sz, DeviceScheduler_ptr dev_sched)
            : DeviceBuffer(
                  sz,
                  details::create_usm_ptr<target>(
                      alloc_request_size_fct(sz, dev_sched), dev_sched, get_alignment(dev_sched))) {
        }

        /**
         * @brief Construct a new Device Buffer object from a SYCL buffer
         *
         * @param syclbuf The SYCL buffer to copy from
         * @param dev_sched A shared pointer to the Device Scheduler
         *
         * This constructor creates a new Device Buffer object with the same size
         * as the given SYCL buffer. It allocates the buffer as USM memory and stores
         * the USM pointer and the size in the respective member variables. The
         * constructor also copies the content of the SYCL buffer into the newly
         * allocated USM buffer.
         */
        DeviceBuffer(sycl::buffer<T> &syclbuf, std::shared_ptr<DeviceScheduler> dev_sched)
            : DeviceBuffer(syclbuf.size(), dev_sched) {
            copy_from_sycl_buffer(syclbuf);
        }

        /**
         * @brief Construct a new Device Buffer object from a SYCL buffer with a given size
         *
         * @param syclbuf The SYCL buffer to copy from
         * @param sz The size of the buffer in number of elements
         * @param dev_sched A shared pointer to the Device Scheduler
         *
         * This constructor creates a new Device Buffer object with the given size.
         * It allocates the buffer as USM memory and stores the USM pointer and the
         * size in the respective member variables. The constructor also copies the
         * first `sz` elements of the SYCL buffer into the newly allocated USM
         * buffer.
         */
        DeviceBuffer(
            sycl::buffer<T> &syclbuf, size_t sz, std::shared_ptr<DeviceScheduler> dev_sched)
            : DeviceBuffer(sz, dev_sched) {
            copy_from_sycl_buffer(syclbuf, sz);
        }

        /**
         * @brief Construct a new Device Buffer object by moving from a SYCL buffer
         *
         * @param syclbuf The SYCL buffer to move from
         * @param dev_sched A shared pointer to the Device Scheduler
         *
         * This constructor moves a SYCL buffer into a new Device Buffer object.
         * It forwards the SYCL buffer and the device scheduler to another constructor.
         */
        DeviceBuffer(sycl::buffer<T> &&syclbuf, std::shared_ptr<DeviceScheduler> dev_sched)
            : DeviceBuffer(syclbuf, dev_sched) {}

        /**
         * @brief Construct a new Device Buffer object by moving from a SYCL buffer
         * with a given size
         *
         * @param syclbuf The SYCL buffer to move from
         * @param sz The size of the buffer in number of elements
         * @param dev_sched A shared pointer to the Device Scheduler
         *
         * This constructor moves a SYCL buffer into a new Device Buffer object.
         * It forwards the SYCL buffer and the device scheduler to another
         * constructor. The size of the buffer is also given as a parameter.
         */
        DeviceBuffer(
            sycl::buffer<T> &&syclbuf, size_t sz, std::shared_ptr<DeviceScheduler> dev_sched)
            : DeviceBuffer(syclbuf, sz, dev_sched) {}

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
            if (!bool(events_hndl)) {
                // If this is not allocated it must be a moved object
                if (hold.get_raw_ptr() != nullptr) {
                    shambase::throw_with_loc<std::runtime_error>(
                        "you have an event handler but not pointer, like how ???");
                }
                return;
            }
            // This object is empty, it was probably moved
            if (hold.get_raw_ptr() == nullptr && events_hndl->is_empty()) {
                return;
            }
            // give the ptr holder and event handler to the memory handler
            details::release_usm_ptr(std::move(hold), shambase::extract_pointer(events_hndl));
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
        [[nodiscard]] inline const T *get_read_access(
            sham::EventList &depends_list, SourceLocation src_loc = SourceLocation{}) const {
            shambase::get_check_ref(events_hndl).read_access(depends_list, src_loc);
            return hold.template ptr_cast<T>();
        }

        /**
         * @brief Get a read-write pointer to the buffer's data
         *
         * This function returns a pointer to the buffer's data. The event handler is updated to
         * reflect the write access.
         *
         * @todo should be made const also ???
         *
         * @param depends_list A vector of SYCL events to wait for before
         *        accessing the buffer.
         * @return A pointer to the buffer's data.
         */
        [[nodiscard]] inline T *get_write_access(
            sham::EventList &depends_list, SourceLocation src_loc = SourceLocation{}) {
            shambase::get_check_ref(events_hndl).write_access(depends_list, src_loc);
            return hold.template ptr_cast<T>();
        }

        /**
         * @brief Complete the event state of the buffer.
         *
         * This function complete the event state of the buffer by registering the
         * event resulting of the last queried access
         *
         * @note This function is const compatible.
         *
         * @warning This function must be called on ALL accessed buffers after a kernel
         * to ensure that the state of the event handlers are up to date.
         *
         * @param e The SYCL event resulting of the queried access.
         */
        void complete_event_state(sycl::event e) const {
            shambase::get_check_ref(events_hndl).complete_state(e);
        }

        /**
         * @brief Complete the event state of the buffer.
         *
         * This function complete the event state of the buffer by registering the
         * event resulting of the last queried access
         *
         * @note This function is const compatible.
         *
         * @warning This function must be called on ALL accessed buffers after a kernel
         * to ensure that the state of the event handlers are up to date.
         *
         * @param e The SYCL event resulting of the queried access.
         */
        void complete_event_state(const std::vector<sycl::event> &e) const {
            shambase::get_check_ref(events_hndl).complete_state(e);
        }

        /**
         * @brief Complete the event state of the buffer.
         *
         * This function complete the event state of the buffer by registering the
         * event resulting of the last queried access
         *
         * @note This function is const compatible.
         *
         * @warning This function must be called on ALL accessed buffers after a kernel
         * to ensure that the state of the event handlers are up to date.
         *
         * @param e The SYCL event resulting of the queried access.
         */
        void complete_event_state(sham::EventList &e) const {
            shambase::get_check_ref(events_hndl).complete_state(e);
        }

        /**
         * @brief Wait for all the events associated with the buffer to be completed
         *
         * This function waits for all the events associated with the buffer to be
         * completed. The events are associated with the buffer by calling
         * complete_event_state.
         */
        void synchronize() const { shambase::get_check_ref(events_hndl).wait_all(); }

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
         * @return The Device scheduler pointer corresponding to the held allocation
         */
        [[nodiscard]] inline std::shared_ptr<DeviceScheduler> &get_dev_scheduler_ptr() {
            return hold.get_dev_scheduler_ptr();
        }

        /**
         * @brief Gets the Device scheduler pointer corresponding to the held allocation
         *
         * @return The Device scheduler pointer corresponding to the held allocation
         */
        [[nodiscard]] inline const std::shared_ptr<DeviceScheduler> &get_dev_scheduler_ptr() const {
            return hold.get_dev_scheduler_ptr();
        }

        /**
         * @brief Gets the DeviceQueue associated with the held allocation
         *
         * @return The DeviceQueue associated with the held allocation
         */
        [[nodiscard]] inline DeviceQueue &get_queue() const {
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
         * @brief Gets the amount of memory used by the buffer
         *
         * @return The amount of memory used by the buffer
         */
        [[nodiscard]] inline size_t get_mem_usage() const { return hold.get_bytesize(); }

        /**
         * @brief Check if the buffer is empty
         *
         * @return `true` if the buffer is empty, `false` otherwise
         */
        [[nodiscard]] inline bool is_empty() const { return size == 0; }

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
        [[nodiscard]] inline std::vector<T> copy_to_stdvec() const {
            std::vector<T> ret(size);

            if (size > 0) {
                sham::EventList depends_list;
                const T *ptr = get_read_access(depends_list);

                sycl::event e = get_queue().submit(depends_list, [&](sycl::handler &cgh) {
                    cgh.copy(ptr, ret.data(), size);
                });

                e.wait_and_throw();
                complete_event_state(sycl::event{});
            }

            return ret;
        }

        /**
         * @brief Copies a specified range of elements from the buffer to a std::vector.
         *
         * This function creates a new std::vector containing elements from the buffer
         * within the specified index range [begin, end). The function ensures that the
         * indices are valid and throws an exception if they are not.
         *
         * @param begin The starting index of the range to copy, inclusive.
         * @param end The ending index of the range to copy, exclusive.
         * @return A std::vector containing the elements from the specified range.
         * @throws std::invalid_argument if the end index is greater than the buffer size
         *         or if the begin index is greater than or equal to the end index.
         */
        [[nodiscard]] inline std::vector<T> copy_to_stdvec_idx_range(
            size_t begin, size_t end) const {

            if (end > size) {
                shambase::throw_with_loc<std::invalid_argument>(shambase::format(
                    "copy_to_stdvec_idx_range: end > size\n  end = {},\n  size = {}", end, size));
            }

            if (begin > end) {
                shambase::throw_with_loc<std::invalid_argument>(shambase::format(
                    "copy_to_stdvec_idx_range: begin >= end\n  begin = {},\n  end = {}",
                    begin,
                    end));
            }

            u32 size_cp = end - begin;
            std::vector<T> ret(size_cp);

            if (size_cp > 0) {
                sham::EventList depends_list;
                const T *ptr = get_read_access(depends_list);

                sycl::event e = get_queue().submit(depends_list, [&](sycl::handler &cgh) {
                    cgh.copy(ptr + begin, ret.data(), size_cp);
                });

                e.wait_and_throw();
                complete_event_state(sycl::event{});
            }

            return ret;
        }

        /**
         * @brief Copy a range of elements from the buffer to another buffer
         *
         * This function copies a range of elements from the buffer to another buffer.
         * The range is specified by the begin and end indices.
         *
         * @param begin The starting index of the range to copy, inclusive.
         * @param end The ending index of the range to copy, exclusive.
         * @param dest The destination buffer to copy to.
         */
        template<USMKindTarget dest_target>
        inline void copy_range(
            size_t begin, size_t end, sham::DeviceBuffer<T, dest_target> &dest) const {

            if (begin > end) {
                shambase::throw_with_loc<std::invalid_argument>(shambase::format(
                    "copy_range: begin > end\n  begin = {},\n  end = {}", begin, end));
            }

            if (end - begin > dest.get_size()) {
                shambase::throw_with_loc<std::invalid_argument>(shambase::format(
                    "copy_range: end - begin > dest.get_size()\n  end - begin = {},\n  "
                    "dest.get_size() = {}",
                    end - begin,
                    dest.get_size()));
            }

            if (begin == end) {
                return;
            }

            size_t len = end - begin;

            sham::EventList depends_list;
            const T *ptr_src = get_read_access(depends_list) + begin;
            T *ptr_dest      = dest.get_write_access(depends_list);

            sycl::event e = get_queue().submit(depends_list, [&](sycl::handler &cgh) {
                cgh.copy(ptr_src, ptr_dest, len);
            });

            complete_event_state(e);
            dest.complete_event_state(e);
        }

        /**
         * @brief Copy the content of a std::vector into the buffer
         *
         * This function copies the content of a given std::vector into the buffer.
         * The size of the vector must be equal to the size of the buffer.
         *
         * @param vec The std::vector to copy from
         */
        inline void copy_from_stdvec(const std::vector<T> &vec) {

            if (size != vec.size()) {
                shambase::throw_with_loc<std::invalid_argument>(shambase::format(
                    "copy_from_stdvec: size mismatch\n  size = {},\n  vec.size() = {}",
                    size,
                    vec.size()));
            }

            if (size > 0) {
                sham::EventList depends_list;
                T *ptr = get_write_access(depends_list);

                sycl::event e = get_queue().submit(depends_list, [&](sycl::handler &cgh) {
                    cgh.copy(vec.data(), ptr, size);
                });

                e.wait_and_throw();
                complete_event_state(sycl::event{});
            }
        }

        /**
         * @brief Copy the content of a std::vector into the buffer
         *
         * This function copies the content of a given std::vector into the buffer.
         * The size of the vector must be equal to the size of the buffer.
         *
         * @param vec The std::vector to copy from
         * @param sz The number of elements to copy
         */
        inline void copy_from_stdvec(const std::vector<T> &vec, size_t sz) {

            if (sz > vec.size() || sz > size) {
                shambase::throw_with_loc<std::invalid_argument>(shambase::format(
                    "copy_from_stdvec: size mismatch (sz > vec.size() || sz > size)\n  size = "
                    "{},\n  vec.size() = {},\n  sz = {}",
                    size,
                    vec.size(),
                    sz));
            }

            if (sz > 0) {
                sham::EventList depends_list;
                T *ptr = get_write_access(depends_list);

                sycl::event e = get_queue().submit(depends_list, [&](sycl::handler &cgh) {
                    cgh.copy(vec.data(), ptr, sz);
                });

                e.wait_and_throw();
                complete_event_state(sycl::event{});
            }
        }

        /**
         * @brief Copy the content of the buffer to a new SYCL buffer
         *
         * This function creates a new SYCL buffer with the same size and content than the current
         * one and returns it.
         *
         * @return The new SYCL buffer
         */
        [[nodiscard]] inline sycl::buffer<T> copy_to_sycl_buffer() const {
            sycl::buffer<T> ret(size);

            if (size > 0) {
                sham::EventList depends_list;
                const T *ptr = get_read_access(depends_list);

                sycl::event e = get_queue().submit(depends_list, [&](sycl::handler &cgh) {
                    sycl::accessor acc(ret, cgh, sycl::write_only, sycl::no_init);
                    cgh.copy(ptr, acc);
                });

                complete_event_state(e);
            }

            return ret;
        }

        /**
         * @brief Copy the content of a SYCL buffer into the buffer
         *
         * This function copies the content of a given SYCL buffer into the buffer.
         * The size of the SYCL buffer must be equal to the size of the buffer.
         *
         * @param buf The SYCL buffer to copy from
         */
        inline void copy_from_sycl_buffer(sycl::buffer<T> &buf) {

            if (size != buf.size()) {
                shambase::throw_with_loc<std::invalid_argument>(shambase::format(
                    "copy_from_sycl_buffer: size mismatch\n  size = {},\n  buf.size() = {}",
                    size,
                    buf.size()));
            }

            if (size > 0) {
                sham::EventList depends_list;
                T *ptr = get_write_access(depends_list);

                sycl::event e = get_queue().submit(depends_list, [&](sycl::handler &cgh) {
                    sycl::accessor acc(buf, cgh, sycl::read_only);
                    cgh.copy(acc, ptr);
                });

                complete_event_state(e);
            }
        }

        /**
         * @brief Copy the content of a SYCL buffer into the buffer
         *
         * This function copies the content of a given SYCL buffer into the buffer.
         * The size of the SYCL buffer must be equal to the size of the buffer.
         *
         * @param buf The SYCL buffer to copy from
         * @param sz The number of elements to copy
         */
        inline void copy_from_sycl_buffer(sycl::buffer<T> &buf, size_t sz) {

            if (sz > buf.size() || sz > size) {
                shambase::throw_with_loc<std::invalid_argument>(shambase::format(
                    "copy_from_sycl_buffer: size mismatch (sz > buf.size() || sz > size)\n  size = "
                    "{},\n  buf.size() = {},\n  sz = {}",
                    size,
                    buf.size(),
                    sz));
            }

            if (sz > u32_max) {
                shambase::throw_with_loc<std::invalid_argument>(shambase::format(
                    "copy_from_sycl_buffer: size mismatch (sz > u32_max)\n  sz = {}", sz));
            }

            if (size > 0) {
                sham::EventList depends_list;
                T *ptr = get_write_access(depends_list);

                sycl::event e = get_queue().submit(depends_list, [&](sycl::handler &cgh) {
                    sycl::accessor acc(buf, cgh, sycl::read_only);

                    shambase::parallel_for(cgh, sz, "copy field", [=](u32 gid) {
                        ptr[gid] = acc[gid];
                    });
                });

                complete_event_state(e);
            }
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
        [[nodiscard]] inline DeviceBuffer<T, new_target> copy_to() const {
            DeviceBuffer<T, new_target> ret(size, get_dev_scheduler_ptr());

            if (size > 0) {
                sham::EventList depends_list;
                const T *ptr_src = get_read_access(depends_list);
                T *ptr_dest      = ret.get_write_access(depends_list);

                sycl::event e = get_queue().submit(depends_list, [&](sycl::handler &cgh) {
                    cgh.copy(ptr_src, ptr_dest, size);
                });

                complete_event_state(e);
                ret.complete_event_state(e);
            }

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
        inline void copy_from(const DeviceBuffer<T, new_target> &other, size_t copy_size) {

            if (!(copy_size <= get_size() && copy_size <= other.get_size())) {
                shambase::throw_with_loc<std::invalid_argument>(shambase::format(
                    "The size of the copy must be smaller than the size of the buffer involved\n  "
                    "copy_size: {}\n  get_size(): {}\n  other.get_size(): {}",
                    copy_size,
                    get_size(),
                    other.get_size()));
            }

            if (copy_size > 0) {
                sham::EventList depends_list;
                T *ptr_dest      = get_write_access(depends_list);
                const T *ptr_src = other.get_read_access(depends_list);

                sycl::event e = get_queue().submit(depends_list, [&](sycl::handler &cgh) {
                    cgh.copy(ptr_src, ptr_dest, copy_size);
                });

                complete_event_state(e);
                other.complete_event_state(e);
            }
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
        inline void copy_from(const DeviceBuffer<T, new_target> &other) {

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
        inline DeviceBuffer<T, target> copy() const { return copy_to<target>(); }

        /**
         * @brief Creates a new buffer that is a mirror of the current one.
         * Upon destruction of the mirror the changes will be propagated to the original buffer
         *
         * @return The mirror buffer
         */
        template<USMKindTarget mirror_target>
        inline BufferMirror<T, mirror_target, target> mirror_to() {
            return BufferMirror<T, mirror_target, target>(*this);
        }

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
                    shambase::parallel_for(cgh, idx_count, "fill field", [=](u32 gid) {
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
        // Getter fcts
        ///////////////////////////////////////////////////////////////////////

        /**
         * @brief Get the value at a given index in the buffer.
         *
         * This function returns the value at the given index in the buffer.
         *
         * @param idx The index of the value to retrieve
         * @return The value at the given index
         */
        T get_val_at_idx(size_t idx) const {
            T ret;

            if (idx >= size) {
                shambase::throw_with_loc<std::invalid_argument>(shambase::format(
                    "get_val_at_idx: idx >= size\n  idx = {},\n  size = {}", idx, size));
            }

            sham::EventList depends_list;
            const T *ptr = get_read_access(depends_list);

            sycl::event e = get_queue().submit(depends_list, [&](sycl::handler &cgh) {
                cgh.copy(ptr + idx, &ret, 1);
            });

            e.wait_and_throw();
            complete_event_state(sycl::event{});

            return ret;
        }

        void set_val_at_idx(size_t idx, T val) {

            if (idx >= size) {
                shambase::throw_with_loc<std::invalid_argument>(shambase::format(
                    "set_val_at_idx: idx >= size\n  idx = {},\n  size = {}", idx, size));
            }

            sham::EventList depends_list;
            T *ptr = get_write_access(depends_list);

            sycl::event e = get_queue().submit(depends_list, [&](sycl::handler &cgh) {
                cgh.copy(&val, ptr + idx, 1);
            });

            e.wait_and_throw();
            complete_event_state(sycl::event{});
        }
        ///////////////////////////////////////////////////////////////////////
        // Getter fcts (END)
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
        inline void resize(size_t new_size, bool keep_data = true) {

            auto dev_sched = hold.get_dev_scheduler_ptr();

            StackEntry __st{};

            if (alloc_request_size_fct(new_size, dev_sched) > hold.get_bytesize()) {
                // expand storage

                size_t new_storage_size = alloc_request_size_fct(new_size * 1.5, dev_sched);

                DeviceBuffer new_buf(
                    new_size,
                    details::create_usm_ptr<target>(
                        new_storage_size, get_dev_scheduler_ptr(), get_alignment(dev_sched)));

                // copy data
                if (keep_data) {
                    new_buf.copy_from(*this, get_size());
                }

                // override old buffer
                std::swap(new_buf, *this);

            } else if (alloc_request_size_fct(new_size, dev_sched) < hold.get_bytesize() * 0.5) {
                // shrink storage

                size_t new_storage_size = alloc_request_size_fct(new_size, dev_sched);

                DeviceBuffer new_buf(
                    new_size,
                    details::create_usm_ptr<target>(
                        new_storage_size, get_dev_scheduler_ptr(), get_alignment(dev_sched)));

                // copy data
                if (keep_data) {
                    new_buf.copy_from(*this, new_size);
                }

                // override old buffer
                std::swap(new_buf, *this);

            } else {
                size = new_size;
                // no need to resize
            }
        }

        /// same as resize but data will not be copied if reallocation is needed
        inline void resize_discard_data(size_t new_size) { resize(new_size, false); }

        /**
         * @brief Alias for resize_discard_data(0).
         */
        inline void free_alloc() { resize_discard_data(0); }

        /**
         * @brief Expand the buffer by `add_sz` elements.
         *
         * This functions reserves space in the buffer for `add_sz` elements, but doesn't change the
         * buffer's size.
         *
         * @param add_sz The number of elements to add to the buffer.
         */
        inline void expand(u32 add_sz) { resize(get_size() + add_sz); }

        /**
         * @brief Shrink the buffer by `sub_sz` elements.
         *
         * If `sub_sz` is greater than the current size of the buffer, this function will throw an
         * std::invalid_argument.
         *
         * @param sub_sz The number of elements to remove from the buffer.
         */
        inline void shrink(u32 sub_sz) {
            if (sub_sz > get_size()) {
                shambase::throw_with_loc<std::invalid_argument>(shambase::format(
                    "shrink called with sub_sz > get_size()\n  sub_sz: {}\n  get_size(): {}",
                    sub_sz,
                    get_size()));
            }
            resize(get_size() - sub_sz);
        }

        /**
         * @brief Append the content of another buffer to this one.
         *
         * This function appends the content of another buffer to this one. The content of the other
         * buffer is copied into this buffer, and the size of this buffer is increased by the size
         * of the other buffer.
         *
         * @param other The buffer from which to copy the data.
         */
        inline void append(const DeviceBuffer &other) {
            if (this == &other) {
                shambase::throw_with_loc<std::invalid_argument>("cannot append a buffer to itself");
            }

            u32 other_size = other.get_size();
            if (other_size == 0) {
                return; // early exit if the other buffer is empty
            }
            u32 old_size = get_size();

            // allocate space
            expand(other_size);

            sham::EventList depends_list;
            T *ptr             = get_write_access(depends_list);
            const T *other_ptr = other.get_read_access(depends_list);

            sycl::event e = get_queue().submit(depends_list, [&](sycl::handler &cgh) {
                cgh.copy(other_ptr, ptr + old_size, other_size);
            });

            complete_event_state(e);
            other.complete_event_state(e);
        }

        ///////////////////////////////////////////////////////////////////////
        // Size manipulation (END)
        ///////////////////////////////////////////////////////////////////////

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
#if false
            size_t old_sz = get_size();
            resize(old_sz + add_sz);
            size = old_sz;
#endif
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
         *
         * This is wrapped in a unique_ptr to allow DeviceBuffer to be const while registering
         * events
         */
        std::unique_ptr<details::BufferEventHandler> events_hndl;
    };

} // namespace sham
