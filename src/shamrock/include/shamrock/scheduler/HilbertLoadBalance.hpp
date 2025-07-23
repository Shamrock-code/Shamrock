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
 * @file HilbertLoadBalance.hpp
 * @author Timothée David--Cléris (tim.shamrock@proton.me)
 * @brief function to run load balancing with the hilbert curve
 *
 */

#include "shammath/sfc/hilbert.hpp"
#include "shamrock/patch/Patch.hpp"
#include <tuple>
#include <vector>

namespace shamrock::scheduler {

    struct LoadBalancingChangeList {
        struct ChangeOp {
            u64 patch_id;
            u64 patch_idx;
            i32 rank_owner_old;
            i32 rank_owner_new;
            i32 tag_comm;
        };

        std::vector<ChangeOp> change_ops;
    };

    /**
     * @brief hilbert load balancing
     *
     */
    template<class hilbert_num>
    class HilbertLoadBalance {

        using SFC = shamrock::sfc::HilbertCurve<hilbert_num, 3>;

        public:
        /**
         * @brief maximal value along an axis for the patch coordinate
         *
         */
        static constexpr u64 max_box_sz = shamrock::sfc::HilbertCurve<hilbert_num, 3>::max_val;

        /**
         * @brief generate the change list from the list of patch to run the load balancing
         *
         * @param global_patch_list the global patch list
         * @return LoadBalancingChangeList list of changes to apply
         *    format = (index of the patch in global list,old owner rank,new owner rank,mpi
         * communication tag)
         */
        static LoadBalancingChangeList
        make_change_list(std::vector<shamrock::patch::Patch> &global_patch_list);
    };

    using HilbertLB     = HilbertLoadBalance<u64>;
    using HilbertLBQuad = HilbertLoadBalance<shamrock::sfc::quad_hilbert_num>;

} // namespace shamrock::scheduler
