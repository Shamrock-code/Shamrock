// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright(C) 2021-2023 Timothée David--Cléris <timothee.david--cleris@ens-lyon.fr>
// Licensed under CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

#pragma once

/**
 * @file usmbuffer.hpp
 * @author Timothée David--Cléris (timothee.david--cleris@ens-lyon.fr)
 * @brief
 *
 */
 
#include "shambase/exception.hpp"
#include "shambackends/DeviceContext.hpp"
#include "shambackends/DeviceScheduler.hpp"
#include "shambackends/sycl.hpp"
#include <memory>

namespace sham {

    /**
     * @brief Enum listing the different types of USM buffers
     *
     * There are three types of USM buffers:
     *
     * - Device buffers are allocated on the device's memory, and can only be accessed by the
     *   device.
     *
     * - Shared buffers are allocated on the host's memory, and can be accessed by both the host
     *   and the device. (May induce implicit communications between the host and the device)
     *
     * - Host buffers are allocated on the host's memory, and can only be accessed by the host.
     */
    enum USMKindTarget {
        device, ///< Device buffer
        shared, ///< Shared buffer
        host    ///< Host buffer
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
    class usmptr_holder{
        void* usm_ptr = nullptr; ///< The USM buffer pointer
        size_t size = 0;         ///< The size of the USM buffer
        std::shared_ptr<DeviceScheduler> dev_sched;    ///< The SYCL queue used to allocate/free the USM buffer

        public:

        /**
         * @brief Constructor
         *
         * @param sz The size of the USM buffer to be allocated
         * @param q The SYCL queue used to allocate/free the USM buffer
         */
        usmptr_holder(size_t sz, std::shared_ptr<DeviceScheduler> dev_sched);

        /**
         * @brief Default destructor
         *
         * Frees the USM buffer using the SYCL queue used for allocation
         */
        ~usmptr_holder();

        /**
         * @brief Deleted copy constructor
         */
        usmptr_holder(const usmptr_holder& other) = delete;

        /**
         * @brief Move constructor
         *
         * Moves the contents of the other usmptr_holder into this one, leaving the other
         * one in a valid but unspecified state. The other usmptr_holder will not free the
         * USM buffer on destruction.
         *
         * @param other The usmptr_holder to be moved from
         */
        usmptr_holder(usmptr_holder&& other) noexcept
            : usm_ptr(std::exchange(other.usm_ptr, nullptr)), 
            size(other.size),
            dev_sched(other.dev_sched) {}

        /**
         * @brief Deleted copy assignment operator
         */
        usmptr_holder& operator=(const usmptr_holder& other) = delete;

        /**
         * @brief Move assignment operator
         *
         * Moves the contents of the other usmptr_holder into this one, leaving the other
         * one in a valid but unspecified state. The other usmptr_holder will not free the
         * USM buffer on destruction.
         *
         * @param other The usmptr_holder to be moved from
         */
        usmptr_holder& operator=(usmptr_holder&& other) noexcept
        {
            dev_sched = other.dev_sched;
            size = other.size;
            std::swap(usm_ptr, other.usm_ptr);
            return *this;
        }

        /**
         * @brief Cast the USM buffer pointer to the given type
         *
         * @tparam T The type to cast the USM buffer pointer to
         * @return The casted USM buffer pointer
         */
        template<class T>
        inline T* ptr_cast() const {
            return reinterpret_cast<T*>(usm_ptr);
        }

        /**
         * @brief Get the size of the USM buffer
         *
         * @return The size of the USM buffer
         */
        inline size_t get_size() const{
            return size;
        }

        /**
         * @brief Get the SYCL context used for allocation/freeing the USM buffer
         *
         * @return The SYCL context used for allocation/freeing the USM buffer
         */
        inline DeviceScheduler& get_dev_scheduler() const {
            return *dev_sched;
        }
    };
    
    /**
     * @brief A buffer allocated in USM (Unified Shared Memory)
     *
     * @tparam T The type of the buffer's elements
     * @tparam target The USM target where the buffer is allocated (host, device, shared)
     */
    template<class T, USMKindTarget target>
    class usmbuffer{
        usmptr_holder<target> hold; ///< The USM pointer holder
        size_t size = 0; ///< The number of elements in the buffer

        public:

        /**
         * @brief Constructs the USM buffer from its size and a SYCL queue
         *
         * @param sz The number of elements in the buffer
         * @param q The SYCL queue to use for allocation/deallocation
         */
        usmbuffer(size_t sz, std::shared_ptr<DeviceScheduler> dev_sched)
            : hold(sz*sizeof(T), dev_sched), size(sz) {}

        /**
         * @brief Deleted copy constructor
         */
        usmbuffer(const usmbuffer& other) = delete;

        /**
         * @brief Deleted copy assignment operator
         */
        usmbuffer& operator=(const usmbuffer& other) = delete;

        /**
         * @brief Gets a read-only pointer to the buffer's data
         *
         * @return A const pointer to the buffer's data
         */
        [[nodiscard]] inline const T * get_read_only_ptr() const {
            return hold.template ptr_cast<T>();
        }

        /**
         * @brief Gets a read-write pointer to the buffer's data
         *
         * @return A pointer to the buffer's data
         */
        [[nodiscard]] inline T * get_ptr() const {
            return hold.template ptr_cast<T>();
        }

        /**
         * @brief Gets the SYCL context used related to the buffer
         *
         * @return The SYCL context used related to the buffer
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
            return hold.get_size();
        }

    };

    

} // namespace sham
