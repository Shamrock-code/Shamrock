// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright(C) 2021-2023 Timothée David--Cléris <timothee.david--cleris@ens-lyon.fr>
// Licensed under CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

#pragma once

#include "shambase/memory.hpp"
#include "shambase/stacktrace.hpp"

namespace shamrock {

    template<class T>
    class StorageComponent {
        private:
        std::unique_ptr<T> hndl;

        public:
        void set(T &&arg) {
            StackEntry stack_loc{};
            if (hndl) {
                throw shambase::throw_with_loc<std::runtime_error>(
                    "please reset the serial patch tree before");
            }
            hndl = std::make_unique<T>(std::forward<T>(arg));
        }

        T &get() {
            StackEntry stack_loc{};
            return shambase::get_check_ref(hndl);
        }
        void reset() {
            StackEntry stack_loc{};
            hndl.reset();
        }
    };

} // namespace shamrock