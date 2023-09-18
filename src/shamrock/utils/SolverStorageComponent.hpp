// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright(C) 2021-2023 Timothée David--Cléris <timothee.david--cleris@ens-lyon.fr>
// Licensed under CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

#pragma once

#include "shambase/SourceLocation.hpp"
#include "shambase/memory.hpp"
#include "shambase/stacktrace.hpp"

namespace shamrock {

    /**
     * @brief Helper class for Storage Module of any solver
     *
     * @tparam T
     */
    template<class T>
    class StorageComponent {
        private:
        std::unique_ptr<T> hndl;

        public:
        /**
         * @brief replace (by move) the held object
         *
         * @param arg
         */
        void set(T &&arg, SourceLocation loc = SourceLocation()) {
            StackEntry stack_loc{};
            if (hndl) {
                throw shambase::throw_with_loc<std::runtime_error>(
                    "please reset the storage component before",loc);
            }
            hndl = std::make_unique<T>(std::forward<T>(arg));
        }

        /**
         * @brief Get the reference to the held object if it was allocated
         *
         * @return T& the reference held
         */
        T &get(SourceLocation loc = SourceLocation()) {
            StackEntry stack_loc{};
            return shambase::get_check_ref(hndl,loc);
        }

        /**
         * @brief delete the content of the Storage
         */
        void reset() {
            StackEntry stack_loc{};
            hndl.reset();
        }

        /**
         * @brief return whether the storage hold an object or not
         * 
         * @return true the storage is empty
         * @return false the storage hold an object
         */
        bool is_empty(){
            return ! hndl;
        }
    };

} // namespace shamrock