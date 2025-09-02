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
 * @file StorageComponent.hpp
 * @author Timothée David--Cléris (tim.shamrock@proton.me)
 * @brief
 *
 */

#include "shambase/SourceLocation.hpp"
#include "shambase/memory.hpp"
#include "shambase/stacktrace.hpp"

namespace shambase {

    /**
     * @brief Helper class for Storage Module of any solver
     *
     * This class provides a simple way to store an object in the storage module of
     * any solver. it will delete the stored object when
     * it goes out of scope.
     *
     * @tparam T Type of the stored object
     */
    template<class T>
    class StorageComponent {
        private:
        std::unique_ptr<T> hndl;

        public:
        /**
         * @brief Replace the held object by moving the given argument
         *
         * This function replaces the held object by moving the given argument into
         * the StorageComponent. If the StorageComponent already holds an object,
         * this function throws a std::runtime_error.
         *
         * @param arg The object to be moved into the StorageComponent.
         *
         * @throws std::runtime_error If the StorageComponent already holds an
         * object.
         */
        void set(T &&arg, SourceLocation loc = SourceLocation()) {
            StackEntry stack_loc{};
            if (bool(hndl)) {
                throw shambase::make_except_with_loc<std::runtime_error>(
                    "please reset the storage component before", loc);
            }
            hndl = std::make_unique<T>(std::forward<T>(arg));
        }

        /**
         * @brief Get the reference to the held object if it was allocated
         *
         * This function returns a reference to the object held by this
         * StorageComponent. If the StorageComponent does not hold anything,
         * it throws a std::runtime_error.
         *
         * @return T& the reference held
         *
         * @throws std::runtime_error If the StorageComponent does not hold
         * anything.
         */
        T &get(SourceLocation loc = SourceLocation()) {
            StackEntry stack_loc{};
            return shambase::get_check_ref(hndl, loc);
        }

        /**
         * @brief Reset the storage component
         *
         * Delete the content of the StorageComponent by resetting the unique_ptr
         * holding the stored object.
         */
        void reset() {
            StackEntry stack_loc{};
            hndl.reset();
        }
        /**
         * @brief Check if the storage component is empty
         *
         * This function returns whether the storage component is empty or not.
         *
         * @return true if the storage component does not hold anything, false
         * otherwise.
         */
        bool is_empty() { return !bool(hndl); }
    };

} // namespace shambase
