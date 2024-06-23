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
#include "shambackends/USMPtrHolder.hpp"
#include "shambackends/sycl.hpp"
#include <memory>

namespace sham {

    
    /**
     * @brief A buffer allocated in USM (Unified Shared Memory)
     *
     * @tparam T The type of the buffer's elements
     * @tparam target The USM target where the buffer is allocated (host, device, shared)
     */
    template<class T, USMKindTarget target>
    class DeviceBuffer{
        
        USMPtrHolder<target> hold; ///< The USM pointer holder
        size_t size = 0; ///< The number of elements in the buffer

        public:

        /**
         * @brief Constructs the USM buffer from its size and a SYCL queue
         *
         * @param sz The number of elements in the buffer
         * @param q The SYCL queue to use for allocation/deallocation
         */
        DeviceBuffer(size_t sz, std::shared_ptr<DeviceScheduler> dev_sched)
            : hold(sz*sizeof(T), dev_sched), size(sz) {}

        /**
         * @brief Deleted copy constructor
         */
        DeviceBuffer(const DeviceBuffer& other) = delete;

        /**
         * @brief Deleted copy assignment operator
         */
        DeviceBuffer& operator=(const DeviceBuffer& other) = delete;

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
