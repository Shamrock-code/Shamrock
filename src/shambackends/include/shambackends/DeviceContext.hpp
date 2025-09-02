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
 * @file DeviceContext.hpp
 * @author Timothée David--Cléris (tim.shamrock@proton.me)
 * @brief
 */

#include "shambackends/Device.hpp"

namespace sham {

    /**
     * @brief A class that represents a SYCL context
     *
     * This class is responsible for creating and holding the SYCL context
     * object, as well as providing methods for accessing it.
     */
    class DeviceContext {
        public:
        /**
         * The device(s) associated with this context
         */
        std::shared_ptr<Device> device;

        /**
         * The SYCL context object
         */
        sycl::context ctx;

        /**
         * @brief Print information about this context
         */
        void print_info();

        /**
         * @brief Construct a new Device Context object
         *
         * @param device The device(s) to use for this context
         */
        explicit DeviceContext(std::shared_ptr<Device> device);

        /**
         * @brief Check if the context should use direct communication
         *
         * This method returns true if the context should use direct
         * communication, false otherwise.
         *
         * @return true if direct communication should be used
         */
        bool use_direct_comm();
    };

} // namespace sham
