// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright(C) 2021-2023 Timothée David--Cléris <timothee.david--cleris@ens-lyon.fr>
// Licensed under CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

#pragma once

#include "shambase/stacktrace.hpp"
#include "shambase/sycl_utils/vectorProperties.hpp"
#include "shammodels/sph/SPHModelSolverConfig.hpp"
#include "shamrock/scheduler/SerialPatchTree.hpp"
#include "shamrock/scheduler/ShamrockCtx.hpp"
#include "shamsys/legacy/log.hpp"

namespace shammodels {

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

    template<class Tvec>
    class SPHSolverStorage {
        public:
        using Tscal              = shambase::VecComponent<Tvec>;
        static constexpr u32 dim = shambase::VectorProperties<Tvec>::dimension;

        StorageComponent<SerialPatchTree<Tvec>> serial_patch_tree;
    };

} // namespace shammodels