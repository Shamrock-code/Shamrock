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
#include "shammodels/sph/BasicSPHGhosts.hpp"
#include "shammodels/sph/SolverConfig.hpp"
#include "shamrock/scheduler/SerialPatchTree.hpp"
#include "shamrock/scheduler/ShamrockCtx.hpp"
#include "shamrock/tree/RadixTree.hpp"
#include "shamrock/tree/TreeTaversalCache.hpp"
#include "shamsys/legacy/log.hpp"

namespace shammodels::sph {

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

    template<class Tvec, class Tmorton>
    class SolverStorage {
        public:
        using Tscal              = shambase::VecComponent<Tvec>;
        static constexpr u32 dim = shambase::VectorProperties<Tvec>::dimension;


        using GhostHandle        = BasicSPHGhostHandler<Tvec>;
        using GhostHandleCache   = typename GhostHandle::CacheMap;
        using PreStepMergedField = typename GhostHandle::PreStepMergedField;

        using RTree = RadixTree<Tmorton, Tvec>;

        StorageComponent<SerialPatchTree<Tvec>> serial_patch_tree;

        StorageComponent<GhostHandle> ghost_handler;

        StorageComponent<GhostHandleCache> ghost_patch_cache;

        StorageComponent<shambase::DistributedData<PreStepMergedField>> merged_xyzh;

        StorageComponent<shambase::DistributedData<RTree>> merged_pos_trees;


        StorageComponent<shambase::DistributedData<RadixTreeField<Tscal>>> rtree_rint_field;

        StorageComponent<shamrock::tree::ObjectCacheHandler> neighbors_cache;


        StorageComponent<shamrock::ComputeField<Tscal>> omega;


        StorageComponent<shamrock::patch::PatchDataLayout> ghost_layout;

        StorageComponent<shambase::DistributedData<shamrock::MergedPatchData>> merged_patchdata_ghost;

        StorageComponent<shamrock::ComputeField<Tscal>> pressure;

        StorageComponent<shamrock::ComputeField<Tvec>> old_axyz;
        StorageComponent<shamrock::ComputeField<Tscal>> old_duint;
    };

} // namespace shammodels