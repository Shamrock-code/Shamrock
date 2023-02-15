// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright(C) 2021-2022 Timothée David--Cléris <timothee.david--cleris@ens-lyon.fr>
// Licensed under CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

#pragma once

/**
 * @file loadbalancing_hilbert.hpp
 * @author Timothée David--Cléris (timothee.david--cleris@ens-lyon.fr)
 * @brief function to run load balancing with the hilbert curve
 * @version 1.0
 * @date 2022-02-28
 * 
 * @copyright Copyright (c) 2022
 * 
 */

#include "aliases.hpp"
#include "shamrock/patch/Patch.hpp"
#include "shamrock/sfc/hilbert.hpp"

#include <vector>
#include <tuple>





namespace shamrock::scheduler {
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
        * @return std::vector<std::tuple<u64, i32, i32, i32>> list of changes to apply
        *    format = (index of the patch in global list,old owner rank,new owner rank,mpi communication tag)
        */
        static std::vector<std::tuple<u64, i32, i32, i32>> make_change_list(std::vector<shamrock::patch::Patch> &global_patch_list);

    };

    using HilbertLB = HilbertLoadBalance<u64>;
    using HilbertLBQuad = HilbertLoadBalance<shamrock::sfc::quad_hilbert_num>;

}