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
 */

#include "shambase/stacktrace.hpp"
#include "shambase/sycl_utils/vectorProperties.hpp"
#include "shammodels/amr/AMRStencilCache.hpp"
#include "shammodels/amr/basegodunov/GhostZoneData.hpp"
#include "shammodels/amr/NeighGraph.hpp"
#include "shammodels/amr/AMRCellInfos.hpp"
#include "shamrock/scheduler/ComputeField.hpp"
#include "shamrock/scheduler/SerialPatchTree.hpp"
#include "shamrock/scheduler/ShamrockCtx.hpp"
#include "shamrock/tree/RadixTree.hpp"
#include "shamrock/tree/TreeTraversalCache.hpp"
#include "shamsys/legacy/log.hpp"
#include "shambase/StorageComponent.hpp"
#include "shamrock/scheduler/InterfacesUtility.hpp"

namespace shammodels::basegodunov {

    template<class T>
    using Component = shambase::StorageComponent<T>;

    template<class Tvec, class TgridVec, class Tmorton>
    class SolverStorage {
        public:
        using Tscal              = shambase::VecComponent<Tvec>;
        using Tgridscal          = shambase::VecComponent<TgridVec>;
        static constexpr u32 dim = shambase::VectorProperties<Tvec>::dimension;

        using RTree = RadixTree<Tmorton, TgridVec>;

        Component<SerialPatchTree<TgridVec>> serial_patch_tree;

        Component<GhostZonesData<Tvec, TgridVec>> ghost_zone_infos;

        Component<shamrock::patch::PatchDataLayout> ghost_layout;

        Component<shambase::DistributedData<shamrock::MergedPatchData>> merged_patchdata_ghost;

        Component<shammodels::basegodunov::modules::CellInfos<Tvec, TgridVec>> cell_infos; 

        Component<shambase::DistributedData<shammath::AABB<TgridVec>>> merge_patch_bounds;
        Component<shambase::DistributedData<RTree>> trees;

        Component<shambase::DistributedData<shammodels::basegodunov::modules::OrientedAMRGraph<Tvec, TgridVec>>> cell_link_graph;

        Component<shamrock::ComputeField<Tvec>> grad_rho;
        Component<shamrock::ComputeField<Tvec>> dx_rhov;
        Component<shamrock::ComputeField<Tvec>> dy_rhov;
        Component<shamrock::ComputeField<Tvec>> dz_rhov;
        Component<shamrock::ComputeField<Tvec>> grad_rhoe;

        struct {
            f64 interface = 0;
            f64 neighbors = 0;
            f64 io = 0;

            void reset(){
                *this = {};
            }
        } timings_details;
    };

} // namespace shammodels::basegodunov