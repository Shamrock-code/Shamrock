// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright (c) 2021-2024 Timothée David--Cléris <tim.shamrock@proton.me>
// SPDX-License-Identifier: CeCILL Free Software License Agreement v2.1
// Shamrock is licensed under the CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

#pragma once

/**
 * @file USMPtrHolder.hpp
 * @author Timothée David--Cléris (tim.shamrock@proton.me)
 * @brief This file contains the declaration of the USMPtrHolder class.
 *
 * The USMPtrHolder class is a smart pointer that manages the memory allocated
 * using SYCL unified shared memory (USM). It provides a way to safely allocate, use,
 * and deallocate memory in USM.
 */

#include "shambase/ptr.hpp"
#include "shambackends/DeviceScheduler.hpp"
#include <memory>
#include <utility>

namespace sham {

    /**
     * @brief Enum listing the different types of USM pointers allocations
     *
     * - Device USM pointers are allocated on the device's memory, and can only be accessed by the
     *   device.
     *
     * - Shared USM pointers are allocated on the host's memory, and can be accessed by both the
     *   host and the device. (May induce implicit communications between the host and the device)
     *
     * - Host USM pointers are allocated on the host's memory, and can only be accessed by the host.
     */
    enum USMKindTarget {
        device, ///< Device memory
        shared, ///< Shared memory
        host    ///< Host memory
    };

    /**
     * @brief Class for holding a USM pointer
     *
     * This class is a simple RAII wrapper around a USM (Unified Shared Memory) pointer.
     * It is a move-only class that manages the lifetime of the USM buffer.
     *
     * The USM buffer can be either a device, shared or host buffer, depending on the
     * template parameter `target`.
     *
     * The move constructor and move assignment operator are deleted to prevent
     * accidental copies of the class.
     */
    template<USMKindTarget target>
    class USMPtrHolder {

        void *usm_ptr = nullptr; ///< The USM buffer pointer
        size_t size   = 0;       ///< The size of the USM buffer
        std::shared_ptr<DeviceScheduler>
            dev_sched; ///< The SYCL queue used to allocate/free the USM buffer

        USMPtrHolder(void *usm_ptr, size_t size, std::shared_ptr<DeviceScheduler> dev_sched)
            : usm_ptr(usm_ptr), size(size), dev_sched(std::move(dev_sched)) {}

        public:
        void free_ptr(); ///< Free the held pointer

        /**
         * @brief Create a USM pointer holder
         *
         * Allocate a USM buffer of the given size using the provided SYCL queue.
         * The USM buffer can be either a device, shared or host buffer,
         * depending on the template parameter `target`.
         *
         * @param sz The size of the USM buffer to be allocated
         * @param dev_sched The Device Scheduler used to allocate/free the USM buffer
         * @param alignment The alignment of the USM buffer (optional)
         *
         * @return A USMPtrHolder instance wrapping the allocated USM buffer
         */
        static USMPtrHolder create(
            size_t sz,
            std::shared_ptr<DeviceScheduler> dev_sched,
            std::optional<size_t> alignment = std::nullopt);

        static USMPtrHolder create_nullptr(std::shared_ptr<DeviceScheduler> dev_sched);

        /**
         * @brief USM pointer holder destructor
         *
         * Frees the USM pointer if not equall to nullptr
         */
        ~USMPtrHolder();

        /**
         * @brief Deleted copy constructor
         */
        USMPtrHolder(const USMPtrHolder &other) = delete;

        /**
         * @brief Move constructor
         *
         * Moves the contents of the other USMPtrHolder into this one, leaving the other
         * one with a nullptr USM pointer, which disable the destructor.
         *
         * @param other The USMPtrHolder to be moved from
         */
        USMPtrHolder(USMPtrHolder &&other) noexcept
            : usm_ptr(std::exchange(other.usm_ptr, nullptr)), size(other.size),
              dev_sched(other.dev_sched) {}

        /**
         * @brief Deleted copy assignment operator
         */
        USMPtrHolder &operator=(const USMPtrHolder &other) = delete;

        /**
         * @brief Move assignment operator
         *
         * Moves the contents of the other USMPtrHolder into this one, leaving the other
         * one in a valid but unspecified state. The other USMPtrHolder will not free the
         * USM buffer on destruction.
         *
         * @param other The USMPtrHolder to be moved from
         */
        USMPtrHolder &operator=(USMPtrHolder &&other) noexcept {
            dev_sched = other.dev_sched;
            size      = other.size;
            std::swap(usm_ptr, other.usm_ptr);
            return *this;
        }

        /**
         * @brief Cast the USM pointer to the given type
         *
         * @tparam T The type to cast the USM buffer pointer to
         * @return The casted USM pointer
         */
        template<class T>
        inline T *ptr_cast() const {
            if (!shambase::is_aligned<T>(usm_ptr)) {
                shambase::throw_with_loc<std::runtime_error>(
                    "The USM pointer is not aligned with the given type");
            }
            return reinterpret_cast<T *>(usm_ptr);
        }

        /**
         * @brief Get the raw pointer of the USM allocation
         *
         * This method returns the raw pointer to the USM allocation. The caller must
         * be careful with the type and the usage of the returned pointer.
         *
         * @return The raw pointer of the USM allocation
         */
        [[nodiscard]] inline void *get_raw_ptr() const { return usm_ptr; }

        /**
         * @brief Get the size of the USM allocation (in byte)
         *
         * @return The size of the USM allocation (in byte)
         */
        [[nodiscard]] inline size_t get_bytesize() const { return size; }

        /**
         * @brief Get the SYCL context used for allocation/freeing the USM buffer
         *
         * @return The SYCL context used for allocation/freeing the USM buffer
         */
        [[nodiscard]] inline DeviceScheduler &get_dev_scheduler() const { return *dev_sched; }

        /**
         * @brief Get the SYCL context used for allocation/freeing the USM buffer
         *
         * @return The SYCL context used for allocation/freeing the USM buffer
         */
        [[nodiscard]] inline std::shared_ptr<DeviceScheduler> &get_dev_scheduler_ptr() {
            return dev_sched;
        }

        /**
         * @brief Get the SYCL context used for allocation/freeing the USM buffer
         *
         * @return The SYCL context used for allocation/freeing the USM buffer
         */
        [[nodiscard]] inline const std::shared_ptr<DeviceScheduler> &get_dev_scheduler_ptr() const {
            return dev_sched;
        }
    };

} // namespace sham
