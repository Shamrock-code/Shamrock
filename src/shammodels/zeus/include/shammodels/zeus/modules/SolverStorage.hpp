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
 * @file SolverStorage.hpp
 * @author Timothée David--Cléris (tim.shamrock@proton.me)
 * @brief
 *
 */

#include "shambase/StorageComponent.hpp"
#include "shambase/stacktrace.hpp"
#include "shambackends/vec.hpp"
#include "shammodels/zeus/GhostZoneData.hpp"
#include "shammodels/zeus/NeighFaceList.hpp"
#include "shamrock/scheduler/ComputeField.hpp"
#include "shamrock/scheduler/InterfacesUtility.hpp"
#include "shamrock/scheduler/SerialPatchTree.hpp"
#include "shamrock/scheduler/ShamrockCtx.hpp"
#include "shamsys/legacy/log.hpp"
#include "shamtree/RadixTree.hpp"
#include "shamtree/TreeTraversalCache.hpp"
namespace shammodels::zeus {

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
        Component<shamrock::patch::PatchDataLayout> ghost_layout_Q;

        Component<shambase::DistributedData<shamrock::MergedPatchData>> merged_patchdata_ghost;
        Component<shambase::DistributedData<shamrock::MergedPatchData>> merged_patchdata_ghost_Q;
        Component<shambase::DistributedData<shammath::AABB<TgridVec>>> merge_patch_bounds;
        Component<shambase::DistributedData<RTree>> trees;

        Component<shamrock::tree::ObjectCacheHandler> neighbors_cache;

        Component<shamrock::ComputeField<Tscal>> pressure;

        /**
         * @brief for each face give a lookup table for the normal orientation
         * 0 = x-
         * 1 = x+
         * 2 = y-
         * 3 = y+
         * 4 = z-
         * 5 = z+
         *
         */
        Component<shambase::DistributedData<sycl::buffer<u8>>> face_normals_lookup;

        Component<shambase::DistributedData<NeighFaceList<Tvec>>> face_lists;

        Component<shamrock::ComputeField<Tvec>> vel_n;
        Component<shamrock::ComputeField<Tvec>> vel_n_xp;
        Component<shamrock::ComputeField<Tvec>> vel_n_yp;
        Component<shamrock::ComputeField<Tvec>> vel_n_zp;

        Component<shamrock::ComputeField<Tscal>> rho_n_xm;
        Component<shamrock::ComputeField<Tscal>> rho_n_ym;
        Component<shamrock::ComputeField<Tscal>> rho_n_zm;

        Component<shamrock::ComputeField<Tscal>> pres_n_xm;
        Component<shamrock::ComputeField<Tscal>> pres_n_ym;
        Component<shamrock::ComputeField<Tscal>> pres_n_zm;

        Component<shamrock::ComputeField<Tvec>> q_AV_n_xm;
        Component<shamrock::ComputeField<Tvec>> q_AV_n_ym;
        Component<shamrock::ComputeField<Tvec>> q_AV_n_zm;

        Component<shamrock::ComputeField<Tvec>> forces;

        Component<shamrock::ComputeField<Tvec>> q_AV;
        Component<shamrock::ComputeField<Tscal>> div_v_n;

        Component<shamrock::ComputeField<sycl::vec<Tscal, 8>>> Q;
        Component<shamrock::ComputeField<sycl::vec<Tscal, 8>>> a_x;
        Component<shamrock::ComputeField<sycl::vec<Tscal, 8>>> a_y;
        Component<shamrock::ComputeField<sycl::vec<Tscal, 8>>> a_z;

        Component<shamrock::ComputeField<sycl::vec<Tscal, 8>>> Q_xm;
        Component<shamrock::ComputeField<sycl::vec<Tscal, 8>>> Q_ym;
        Component<shamrock::ComputeField<sycl::vec<Tscal, 8>>> Q_zm;

        Component<shamrock::ComputeField<sycl::vec<Tscal, 8>>> Qstar_x;
        Component<shamrock::ComputeField<sycl::vec<Tscal, 8>>> Qstar_y;
        Component<shamrock::ComputeField<sycl::vec<Tscal, 8>>> Qstar_z;

        Component<shamrock::ComputeField<sycl::vec<Tscal, 8>>> Flux_x;
        Component<shamrock::ComputeField<sycl::vec<Tscal, 8>>> Flux_y;
        Component<shamrock::ComputeField<sycl::vec<Tscal, 8>>> Flux_z;
        Component<shamrock::ComputeField<sycl::vec<Tscal, 8>>> Flux_xp;
        Component<shamrock::ComputeField<sycl::vec<Tscal, 8>>> Flux_yp;
        Component<shamrock::ComputeField<sycl::vec<Tscal, 8>>> Flux_zp;

        /**
         * @brief derivatives of the velocity
         * layout : \f$ [\partial_i u_x, \partial_i u_y, \partial_i u_z] \f$
         */
        Component<shamrock::ComputeField<Tvec>> gradu;

        struct Timings {
            f64 interface = 0;
            f64 neighbors = 0;
            f64 io        = 0;

            /// Reset the timings logged in the storage
            void reset() { *this = {}; }
        } timings_details;
    };

} // namespace shammodels::zeus
