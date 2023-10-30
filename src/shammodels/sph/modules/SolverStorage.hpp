// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright(C) 2021-2023 Timothée David--Cléris <timothee.david--cleris@ens-lyon.fr>
// Licensed under CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

#pragma once

/**
 * @file SolverStorage.hpp
 * @author Timothée David--Cléris (timothee.david--cleris@ens-lyon.fr)
 * @brief 
 * 
 */

#include "shambase/stacktrace.hpp"
#include "shambase/sycl_utils/vectorProperties.hpp"
#include "shammodels/sph/BasicSPHGhosts.hpp"
#include "shammodels/sph/SinkPartStruct.hpp"
#include "shammodels/sph/SolverConfig.hpp"
#include "shamrock/scheduler/SerialPatchTree.hpp"
#include "shamrock/scheduler/ShamrockCtx.hpp"
#include "shamrock/tree/RadixTree.hpp"
#include "shamrock/tree/TreeTraversalCache.hpp"
#include "shambase/StorageComponent.hpp"
#include "shamsys/legacy/log.hpp"

namespace shammodels::sph {

    template<class T>
    using Component = shambase::StorageComponent<T>;

    template<class Tvec, class Tmorton>
    class SolverStorage {
        public:
        using Tscal              = shambase::VecComponent<Tvec>;
        static constexpr u32 dim = shambase::VectorProperties<Tvec>::dimension;

        using GhostHandle        = BasicSPHGhostHandler<Tvec>;
        using GhostHandleCache   = typename GhostHandle::CacheMap;
        using PreStepMergedField = typename GhostHandle::PreStepMergedField;

        using RTree = RadixTree<Tmorton, Tvec>;

        Component<SerialPatchTree<Tvec>> serial_patch_tree;

        Component<GhostHandle> ghost_handler;

        Component<GhostHandleCache> ghost_patch_cache;

        Component<shambase::DistributedData<PreStepMergedField>> merged_xyzh;

        Component<shambase::DistributedData<RTree>> merged_pos_trees;

        Component<shambase::DistributedData<RadixTreeField<Tscal>>> rtree_rint_field;

        Component<shamrock::tree::ObjectCacheHandler> neighbors_cache;

        Component<shamrock::ComputeField<Tscal>> omega;

        Component<shamrock::patch::PatchDataLayout> ghost_layout;

        Component<shambase::DistributedData<shamrock::MergedPatchData>> merged_patchdata_ghost;

        Component<shamrock::ComputeField<Tscal>> pressure;

        Component<shamrock::ComputeField<Tvec>> old_axyz;
        Component<shamrock::ComputeField<Tscal>> old_duint;

        Component<std::vector<SinkParticle<Tvec>>> sinks;

        struct {
            f64 interface = 0;
            f64 neighbors = 0;
            f64 io = 0;

            void reset(){
                *this = {};
            }
        } timings_details;
    };

} // namespace shammodels::sph