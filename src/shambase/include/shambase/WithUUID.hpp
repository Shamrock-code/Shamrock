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
 * @file WithUUID.hpp
 * @author Timothée David--Cléris (tim.shamrock@proton.me)
 * @brief
 */

#include <atomic>

namespace shambase {

    /**
     * @brief A class that provides unique identifiers (UUID) to instances
     *
     * @tparam T The class type to which the UUID will be assigned
     * @tparam Tint The data type for the UUID
     * @tparam thread_safe Whether the UUID constructor should be thread-safe
     *
     * Example usage:
     * @code {.cpp}
     *    class A1 : public WithUUID<A1, u64> {};
     *    ...
     *    std::cout << "Instance1 UUID: " << A1{}.get_uuid() << std::endl;
     * @endcode
     */
    template<typename T, class Tint, bool thread_safe = true>
    class WithUUID {

        protected:
        /**
         * @brief The unique identifier of the class
         */
        Tint uuid;

        public:
        /**
         * @brief Get the uuid of the class
         *
         * @return The uuid of the class
         */
        inline Tint get_uuid() { return uuid; }

        /**
         * @brief Constructor of the class
         *
         * Assigns a unique identifier to the class
         */
        inline WithUUID() {
            if constexpr (thread_safe) {
                // local atomic static storage for the UUID
                static std::atomic<Tint> _uuid = 0;
                // increment and store the UUID (atomic)
                uuid = _uuid.fetch_add(1, std::memory_order_relaxed);
            } else {
                // we need to redo the static storage in this case otherwise
                // some lock xadd would be emitted as std::atomic is thread safe.
                static Tint _uuid = 0;
                uuid              = _uuid++;
            }
        }
    };

} // namespace shambase
