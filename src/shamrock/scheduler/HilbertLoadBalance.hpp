// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright(C) 2021-2023 Timothée David--Cléris <timothee.david--cleris@ens-lyon.fr>
// Licensed under CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

#pragma once

/**
 * @file HilbertLoadBalance.hpp
 * @author Timothée David--Cléris (timothee.david--cleris@ens-lyon.fr)
 * @brief function to run load balancing with the hilbert curve
 * 
 */

#include "aliases.hpp"
#include "shamrock/patch/Patch.hpp"
#include "shamrock/sfc/hilbert.hpp"

#include <vector>
#include <tuple>





namespace shamrock::scheduler {

    struct LoadBalancingChangeList{
        struct ChangeOp{
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
    class HilbertLoadBalance{
        
        using SFC = shamrock::sfc::HilbertCurve<hilbert_num, 3>;

        public : 

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
        *    format = (index of the patch in global list,old owner rank,new owner rank,mpi communication tag)
        */
        static LoadBalancingChangeList make_change_list(std::vector<shamrock::patch::Patch> &global_patch_list);

    };

    using HilbertLB = HilbertLoadBalance<u64>;
    using HilbertLBQuad = HilbertLoadBalance<shamrock::sfc::quad_hilbert_num>;

}