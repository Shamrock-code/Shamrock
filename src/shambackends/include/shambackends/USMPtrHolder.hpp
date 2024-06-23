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
    class USMPtrHolder{
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
        USMPtrHolder(size_t sz, std::shared_ptr<DeviceScheduler> dev_sched);

        /**
         * @brief Default destructor
         *
         * Frees the USM buffer using the SYCL queue used for allocation
         */
        ~USMPtrHolder();

        /**
         * @brief Deleted copy constructor
         */
        USMPtrHolder(const USMPtrHolder& other) = delete;

        /**
         * @brief Move constructor
         *
         * Moves the contents of the other USMPtrHolder into this one, leaving the other
         * one in a valid but unspecified state. The other USMPtrHolder will not free the
         * USM buffer on destruction.
         *
         * @param other The USMPtrHolder to be moved from
         */
        USMPtrHolder(USMPtrHolder&& other) noexcept
            : usm_ptr(std::exchange(other.usm_ptr, nullptr)), 
            size(other.size),
            dev_sched(other.dev_sched) {}

        /**
         * @brief Deleted copy assignment operator
         */
        USMPtrHolder& operator=(const USMPtrHolder& other) = delete;

        /**
         * @brief Move assignment operator
         *
         * Moves the contents of the other USMPtrHolder into this one, leaving the other
         * one in a valid but unspecified state. The other USMPtrHolder will not free the
         * USM buffer on destruction.
         *
         * @param other The USMPtrHolder to be moved from
         */
        USMPtrHolder& operator=(USMPtrHolder&& other) noexcept
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

}