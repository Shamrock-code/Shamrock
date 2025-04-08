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
 * @author Timothée David--Cléris (timothee.david--cleris@ens-lyon.fr)
 * @brief
 */

#include "shambase/StorageComponent.hpp"
#include "shambase/stacktrace.hpp"
#include "shambackends/vec.hpp"
#include "shammodels/common/amr/AMRCellInfos.hpp"
#include "shammodels/common/amr/AMRStencilCache.hpp"
#include "shammodels/common/amr/NeighGraph.hpp"
#include "shammodels/common/amr/NeighGraphLinkField.hpp"
#include "shammodels/ramses/GhostZoneData.hpp"
#include "shamrock/scheduler/ComputeField.hpp"
#include "shamrock/scheduler/InterfacesUtility.hpp"
#include "shamrock/scheduler/SerialPatchTree.hpp"
#include "shamrock/scheduler/ShamrockCtx.hpp"
#include "shamsys/legacy/log.hpp"
#include "shamtree/RadixTree.hpp"
#include "shamtree/TreeTraversalCache.hpp"

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

        Component<shambase::DistributedData<
            shammodels::basegodunov::modules::OrientedAMRGraph<Tvec, TgridVec>>>
            cell_link_graph;

        Component<shamrock::ComputeField<Tscal>> rho_old;
        Component<shamrock::ComputeField<Tvec>> rhovel_old;
        Component<shamrock::ComputeField<Tscal>> rhoetot_old;
        Component<shamrock::ComputeField<Tscal>> rho_dust_old;
        Component<shamrock::ComputeField<Tvec>> rhovel_dust_old;

        Component<shamrock::ComputeField<Tvec>> vel;
        Component<shamrock::ComputeField<Tscal>> press;

        Component<shamrock::ComputeField<Tvec>> grad_rho;
        Component<shamrock::ComputeField<Tvec>> dx_v;
        Component<shamrock::ComputeField<Tvec>> dy_v;
        Component<shamrock::ComputeField<Tvec>> dz_v;
        Component<shamrock::ComputeField<Tvec>> grad_P;

        Component<shambase::DistributedData<
            shammodels::basegodunov::modules::NeighGraphLinkField<std::array<Tscal, 2>>>>
            rho_face_xp;
        Component<shambase::DistributedData<
            shammodels::basegodunov::modules::NeighGraphLinkField<std::array<Tscal, 2>>>>
            rho_face_xm;
        Component<shambase::DistributedData<
            shammodels::basegodunov::modules::NeighGraphLinkField<std::array<Tscal, 2>>>>
            rho_face_yp;
        Component<shambase::DistributedData<
            shammodels::basegodunov::modules::NeighGraphLinkField<std::array<Tscal, 2>>>>
            rho_face_ym;
        Component<shambase::DistributedData<
            shammodels::basegodunov::modules::NeighGraphLinkField<std::array<Tscal, 2>>>>
            rho_face_zp;
        Component<shambase::DistributedData<
            shammodels::basegodunov::modules::NeighGraphLinkField<std::array<Tscal, 2>>>>
            rho_face_zm;

        /**/

        Component<shambase::DistributedData<
            shammodels::basegodunov::modules::NeighGraphLinkField<std::array<Tscal, 2>>>>
            rho_tilde_xp;
        Component<shambase::DistributedData<
            shammodels::basegodunov::modules::NeighGraphLinkField<std::array<Tscal, 2>>>>
            rho_tilde_xm;
        Component<shambase::DistributedData<
            shammodels::basegodunov::modules::NeighGraphLinkField<std::array<Tscal, 2>>>>
            rho_tilde_yp;
        Component<shambase::DistributedData<
            shammodels::basegodunov::modules::NeighGraphLinkField<std::array<Tscal, 2>>>>
            rho_tilde_ym;
        Component<shambase::DistributedData<
            shammodels::basegodunov::modules::NeighGraphLinkField<std::array<Tscal, 2>>>>
            rho_tilde_zp;
        Component<shambase::DistributedData<
            shammodels::basegodunov::modules::NeighGraphLinkField<std::array<Tscal, 2>>>>
            rho_tilde_zm;
        /**/

        Component<shambase::DistributedData<
            shammodels::basegodunov::modules::NeighGraphLinkField<std::array<Tvec, 2>>>>
            vel_face_xp;
        Component<shambase::DistributedData<
            shammodels::basegodunov::modules::NeighGraphLinkField<std::array<Tvec, 2>>>>
            vel_face_xm;
        Component<shambase::DistributedData<
            shammodels::basegodunov::modules::NeighGraphLinkField<std::array<Tvec, 2>>>>
            vel_face_yp;
        Component<shambase::DistributedData<
            shammodels::basegodunov::modules::NeighGraphLinkField<std::array<Tvec, 2>>>>
            vel_face_ym;
        Component<shambase::DistributedData<
            shammodels::basegodunov::modules::NeighGraphLinkField<std::array<Tvec, 2>>>>
            vel_face_zp;
        Component<shambase::DistributedData<
            shammodels::basegodunov::modules::NeighGraphLinkField<std::array<Tvec, 2>>>>
            vel_face_zm;

        /**/
        Component<shambase::DistributedData<
            shammodels::basegodunov::modules::NeighGraphLinkField<std::array<Tvec, 2>>>>
            vel_tilde_xp;
        Component<shambase::DistributedData<
            shammodels::basegodunov::modules::NeighGraphLinkField<std::array<Tvec, 2>>>>
            vel_tilde_xm;
        Component<shambase::DistributedData<
            shammodels::basegodunov::modules::NeighGraphLinkField<std::array<Tvec, 2>>>>
            vel_tilde_yp;
        Component<shambase::DistributedData<
            shammodels::basegodunov::modules::NeighGraphLinkField<std::array<Tvec, 2>>>>
            vel_tilde_ym;
        Component<shambase::DistributedData<
            shammodels::basegodunov::modules::NeighGraphLinkField<std::array<Tvec, 2>>>>
            vel_tilde_zp;
        Component<shambase::DistributedData<
            shammodels::basegodunov::modules::NeighGraphLinkField<std::array<Tvec, 2>>>>
            vel_tilde_zm;
        /**/

        Component<shambase::DistributedData<
            shammodels::basegodunov::modules::NeighGraphLinkField<std::array<Tscal, 2>>>>
            press_face_xp;
        Component<shambase::DistributedData<
            shammodels::basegodunov::modules::NeighGraphLinkField<std::array<Tscal, 2>>>>
            press_face_xm;
        Component<shambase::DistributedData<
            shammodels::basegodunov::modules::NeighGraphLinkField<std::array<Tscal, 2>>>>
            press_face_yp;
        Component<shambase::DistributedData<
            shammodels::basegodunov::modules::NeighGraphLinkField<std::array<Tscal, 2>>>>
            press_face_ym;
        Component<shambase::DistributedData<
            shammodels::basegodunov::modules::NeighGraphLinkField<std::array<Tscal, 2>>>>
            press_face_zp;
        Component<shambase::DistributedData<
            shammodels::basegodunov::modules::NeighGraphLinkField<std::array<Tscal, 2>>>>
            press_face_zm;

        /**/
        Component<shambase::DistributedData<
            shammodels::basegodunov::modules::NeighGraphLinkField<std::array<Tscal, 2>>>>
            press_tilde_xp;
        Component<shambase::DistributedData<
            shammodels::basegodunov::modules::NeighGraphLinkField<std::array<Tscal, 2>>>>
            press_tilde_xm;
        Component<shambase::DistributedData<
            shammodels::basegodunov::modules::NeighGraphLinkField<std::array<Tscal, 2>>>>
            press_tilde_yp;
        Component<shambase::DistributedData<
            shammodels::basegodunov::modules::NeighGraphLinkField<std::array<Tscal, 2>>>>
            press_tilde_ym;
        Component<shambase::DistributedData<
            shammodels::basegodunov::modules::NeighGraphLinkField<std::array<Tscal, 2>>>>
            press_tilde_zp;
        Component<shambase::DistributedData<
            shammodels::basegodunov::modules::NeighGraphLinkField<std::array<Tscal, 2>>>>
            press_tilde_zm;
        /**/

        Component<
            shambase::DistributedData<shammodels::basegodunov::modules::NeighGraphLinkField<Tscal>>>
            flux_rho_face_xp;
        Component<
            shambase::DistributedData<shammodels::basegodunov::modules::NeighGraphLinkField<Tscal>>>
            flux_rho_face_xm;
        Component<
            shambase::DistributedData<shammodels::basegodunov::modules::NeighGraphLinkField<Tscal>>>
            flux_rho_face_yp;
        Component<
            shambase::DistributedData<shammodels::basegodunov::modules::NeighGraphLinkField<Tscal>>>
            flux_rho_face_ym;
        Component<
            shambase::DistributedData<shammodels::basegodunov::modules::NeighGraphLinkField<Tscal>>>
            flux_rho_face_zp;
        Component<
            shambase::DistributedData<shammodels::basegodunov::modules::NeighGraphLinkField<Tscal>>>
            flux_rho_face_zm;

        Component<
            shambase::DistributedData<shammodels::basegodunov::modules::NeighGraphLinkField<Tvec>>>
            flux_rhov_face_xp;
        Component<
            shambase::DistributedData<shammodels::basegodunov::modules::NeighGraphLinkField<Tvec>>>
            flux_rhov_face_xm;
        Component<
            shambase::DistributedData<shammodels::basegodunov::modules::NeighGraphLinkField<Tvec>>>
            flux_rhov_face_yp;
        Component<
            shambase::DistributedData<shammodels::basegodunov::modules::NeighGraphLinkField<Tvec>>>
            flux_rhov_face_ym;
        Component<
            shambase::DistributedData<shammodels::basegodunov::modules::NeighGraphLinkField<Tvec>>>
            flux_rhov_face_zp;
        Component<
            shambase::DistributedData<shammodels::basegodunov::modules::NeighGraphLinkField<Tvec>>>
            flux_rhov_face_zm;

        Component<
            shambase::DistributedData<shammodels::basegodunov::modules::NeighGraphLinkField<Tscal>>>
            flux_rhoe_face_xp;
        Component<
            shambase::DistributedData<shammodels::basegodunov::modules::NeighGraphLinkField<Tscal>>>
            flux_rhoe_face_xm;
        Component<
            shambase::DistributedData<shammodels::basegodunov::modules::NeighGraphLinkField<Tscal>>>
            flux_rhoe_face_yp;
        Component<
            shambase::DistributedData<shammodels::basegodunov::modules::NeighGraphLinkField<Tscal>>>
            flux_rhoe_face_ym;
        Component<
            shambase::DistributedData<shammodels::basegodunov::modules::NeighGraphLinkField<Tscal>>>
            flux_rhoe_face_zp;
        Component<
            shambase::DistributedData<shammodels::basegodunov::modules::NeighGraphLinkField<Tscal>>>
            flux_rhoe_face_zm;

        Component<shamrock::ComputeField<Tscal>> dtrho;
        Component<shamrock::ComputeField<Tvec>> dtrhov;
        Component<shamrock::ComputeField<Tscal>> dtrhoe;

        Component<shamrock::ComputeField<Tscal>> rho_next_no_drag;
        Component<shamrock::ComputeField<Tvec>> rhov_next_no_drag;
        Component<shamrock::ComputeField<Tscal>> rhoe_next_no_drag;

        /**
         * @brief Dust velocity : primitives variables get from conservative rhovel_dust variable
         */
        Component<shamrock::ComputeField<Tvec>> vel_dust;
        /// dust fields gradients (grad rho_dust)
        Component<shamrock::ComputeField<Tvec>> grad_rho_dust;
        /// dust fields gradients (d vdust / d x)
        Component<shamrock::ComputeField<Tvec>> dx_v_dust;
        /// dust fields gradients (d vdust / d y)
        Component<shamrock::ComputeField<Tvec>> dy_v_dust;
        /// dust fields gradients (d vdust / d z)
        Component<shamrock::ComputeField<Tvec>> dz_v_dust;
        // next time step dust density before drag
        Component<shamrock::ComputeField<Tscal>> rho_d_next_no_drag;
        // next time step dust momentum before drag
        Component<shamrock::ComputeField<Tvec>> rhov_d_next_no_drag;
        /**
         * @brief dust densities +x direction stored at the cells faces
         */
        Component<shambase::DistributedData<
            shammodels::basegodunov::modules::NeighGraphLinkField<std::array<Tscal, 2>>>>
            rho_dust_face_xp;
        /**
         * @brief dust densities -x direction stored at the cells faces
         */
        Component<shambase::DistributedData<
            shammodels::basegodunov::modules::NeighGraphLinkField<std::array<Tscal, 2>>>>
            rho_dust_face_xm;
        /**
         * @brief dust densities +y direction stored at the cells faces
         */
        Component<shambase::DistributedData<
            shammodels::basegodunov::modules::NeighGraphLinkField<std::array<Tscal, 2>>>>
            rho_dust_face_yp;
        /**
         * @brief dust densities -y direction stored at the cells faces
         */
        Component<shambase::DistributedData<
            shammodels::basegodunov::modules::NeighGraphLinkField<std::array<Tscal, 2>>>>
            rho_dust_face_ym;
        /**
         * @brief dust densities +z direction stored at the cells faces
         */
        Component<shambase::DistributedData<
            shammodels::basegodunov::modules::NeighGraphLinkField<std::array<Tscal, 2>>>>
            rho_dust_face_zp;
        /**
         * @brief dust densities -z direction stored at the cells faces
         */
        Component<shambase::DistributedData<
            shammodels::basegodunov::modules::NeighGraphLinkField<std::array<Tscal, 2>>>>
            rho_dust_face_zm;

        /**
         * @brief dust densities reconstruct in +x direction stored at the cells faces
         */
        Component<shambase::DistributedData<
            shammodels::basegodunov::modules::NeighGraphLinkField<std::array<Tscal, 2>>>>
            rho_dust_tilde_xp;
        /**
         * @brief dust densities reconstruct in -x direction stored at the cells faces
         */
        Component<shambase::DistributedData<
            shammodels::basegodunov::modules::NeighGraphLinkField<std::array<Tscal, 2>>>>
            rho_dust_tilde_xm;
        /**
         * @brief dust densities reconstruct in +y direction stored at the cells faces
         */
        Component<shambase::DistributedData<
            shammodels::basegodunov::modules::NeighGraphLinkField<std::array<Tscal, 2>>>>
            rho_dust_tilde_yp;
        /**
         * @brief dust densities reconstruct in -y direction stored at the cells faces
         */
        Component<shambase::DistributedData<
            shammodels::basegodunov::modules::NeighGraphLinkField<std::array<Tscal, 2>>>>
            rho_dust_tilde_ym;
        /**
         * @brief dust densities reconstruct in +z direction stored at the cells faces
         */
        Component<shambase::DistributedData<
            shammodels::basegodunov::modules::NeighGraphLinkField<std::array<Tscal, 2>>>>
            rho_dust_tilde_zp;
        /**
         * @brief dust densities reconstruct in -z direction stored at the cells faces
         */
        Component<shambase::DistributedData<
            shammodels::basegodunov::modules::NeighGraphLinkField<std::array<Tscal, 2>>>>
            rho_dust_tilde_zm;

        /**
         * @brief dust velocities in +x direction stored at the cells faces
         */
        Component<shambase::DistributedData<
            shammodels::basegodunov::modules::NeighGraphLinkField<std::array<Tvec, 2>>>>
            vel_dust_face_xp;
        /**
         * @brief dust velocities in -x direction stored at the cells faces
         */
        Component<shambase::DistributedData<
            shammodels::basegodunov::modules::NeighGraphLinkField<std::array<Tvec, 2>>>>
            vel_dust_face_xm;
        /**
         * @brief dust velocities in +y direction stored at the cells faces
         */
        Component<shambase::DistributedData<
            shammodels::basegodunov::modules::NeighGraphLinkField<std::array<Tvec, 2>>>>
            vel_dust_face_yp;
        /**
         * @brief dust velocities in -y direction stored at the cells faces
         */
        Component<shambase::DistributedData<
            shammodels::basegodunov::modules::NeighGraphLinkField<std::array<Tvec, 2>>>>
            vel_dust_face_ym;
        /**
         * @brief dust velocities in +z direction stored at the cells faces
         */
        Component<shambase::DistributedData<
            shammodels::basegodunov::modules::NeighGraphLinkField<std::array<Tvec, 2>>>>
            vel_dust_face_zp;
        /**
         * @brief dust velocities in -z direction stored at the cells faces
         */
        Component<shambase::DistributedData<
            shammodels::basegodunov::modules::NeighGraphLinkField<std::array<Tvec, 2>>>>
            vel_dust_face_zm;

        /**
         * @brief dust velocities reconstruct in +x direction stored at the cells faces
         */
        Component<shambase::DistributedData<
            shammodels::basegodunov::modules::NeighGraphLinkField<std::array<Tvec, 2>>>>
            vel_dust_tilde_xp;
        /**
         * @brief dust velocities reconstruct in -x direction stored at the cells faces
         */
        Component<shambase::DistributedData<
            shammodels::basegodunov::modules::NeighGraphLinkField<std::array<Tvec, 2>>>>
            vel_dust_tilde_xm;
        /**
         * @brief dust velocities reconstruct in +y direction stored at the cells faces
         */
        Component<shambase::DistributedData<
            shammodels::basegodunov::modules::NeighGraphLinkField<std::array<Tvec, 2>>>>
            vel_dust_tilde_yp;
        /**
         * @brief dust velocities reconstruct in -y direction stored at the cells faces
         */
        Component<shambase::DistributedData<
            shammodels::basegodunov::modules::NeighGraphLinkField<std::array<Tvec, 2>>>>
            vel_dust_tilde_ym;
        /**
         * @brief dust velocities reconstruct in +z direction stored at the cells faces
         */
        Component<shambase::DistributedData<
            shammodels::basegodunov::modules::NeighGraphLinkField<std::array<Tvec, 2>>>>
            vel_dust_tilde_zp;
        /**
         * @brief dust velocities reconstruct in -z direction stored at the cells faces
         */
        Component<shambase::DistributedData<
            shammodels::basegodunov::modules::NeighGraphLinkField<std::array<Tvec, 2>>>>
            vel_dust_tilde_zm;

        /**
         * @brief dust density flux at cells interfaces in +x direction
         */
        Component<
            shambase::DistributedData<shammodels::basegodunov::modules::NeighGraphLinkField<Tscal>>>
            flux_rho_dust_face_xp;
        /**
         * @brief dust density flux at cells interfaces in -x direction
         */
        Component<
            shambase::DistributedData<shammodels::basegodunov::modules::NeighGraphLinkField<Tscal>>>
            flux_rho_dust_face_xm;
        /**
         * @brief dust density flux at cells interfaces in +y direction
         */
        Component<
            shambase::DistributedData<shammodels::basegodunov::modules::NeighGraphLinkField<Tscal>>>
            flux_rho_dust_face_yp;
        /**
         * @brief dust density flux at cells interfaces in -y direction
         */
        Component<
            shambase::DistributedData<shammodels::basegodunov::modules::NeighGraphLinkField<Tscal>>>
            flux_rho_dust_face_ym;
        /**
         * @brief dust density flux at cells interfaces in +z direction
         */
        Component<
            shambase::DistributedData<shammodels::basegodunov::modules::NeighGraphLinkField<Tscal>>>
            flux_rho_dust_face_zp;
        /**
         * @brief dust density flux at cells interfaces in -z direction
         */
        Component<
            shambase::DistributedData<shammodels::basegodunov::modules::NeighGraphLinkField<Tscal>>>
            flux_rho_dust_face_zm;
        /**
         * @brief dust momentum flux at cells interfaces in +x direction
         */
        Component<
            shambase::DistributedData<shammodels::basegodunov::modules::NeighGraphLinkField<Tvec>>>
            flux_rhov_dust_face_xp;
        /**
         * @brief dust momentum flux at cells interfaces in -x direction
         */
        Component<
            shambase::DistributedData<shammodels::basegodunov::modules::NeighGraphLinkField<Tvec>>>
            flux_rhov_dust_face_xm;
        /**
         * @brief dust momentum flux at cells interfaces in +y direction
         */
        Component<
            shambase::DistributedData<shammodels::basegodunov::modules::NeighGraphLinkField<Tvec>>>
            flux_rhov_dust_face_yp;
        /**
         * @brief dust momentum flux at cells interfaces in -y direction
         */
        Component<
            shambase::DistributedData<shammodels::basegodunov::modules::NeighGraphLinkField<Tvec>>>
            flux_rhov_dust_face_ym;
        /**
         * @brief dust momentum flux at cells interfaces in +z direction
         */
        Component<
            shambase::DistributedData<shammodels::basegodunov::modules::NeighGraphLinkField<Tvec>>>
            flux_rhov_dust_face_zp;
        /**
         * @brief dust momentum flux at cells interfaces in -z direction
         */
        Component<
            shambase::DistributedData<shammodels::basegodunov::modules::NeighGraphLinkField<Tvec>>>
            flux_rhov_dust_face_zm;
        // time derivative dust density
        Component<shamrock::ComputeField<Tscal>> dtrho_dust;
        // time derivative dust momemtum
        Component<shamrock::ComputeField<Tvec>> dtrhov_dust;

        struct Timings {
            f64 interface = 0;
            f64 neighbors = 0;
            f64 io        = 0;

            /// Reset the timings logged in the storage
            void reset() { *this = {}; }
        } timings_details;
    };

} // namespace shammodels::basegodunov
