// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright(C) 2021-2023 Timothée David--Cléris <timothee.david--cleris@ens-lyon.fr>
// Licensed under CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

#pragma once

/**
 * @file scheduler_patch_list.hpp
 * @author Timothée David--Cléris (timothee.david--cleris@ens-lyon.fr)
 * @brief Class to handle the patch list of the mpi scheduler
 * @version 0.1
 * @date 2022-02-08
 * 
 * @copyright Copyright (c) 2022
 * 
 */


#include <array>
#include <stdexcept>
#include <tuple>
#include <vector>
#include <unordered_map>
#include <unordered_set>

#include "shambase/SourceLocation.hpp"
#include "shambase/exception.hpp"
#include "shamrock/patch/Patch.hpp"

/**
 * @brief Handle the patch list of the mpi scheduler
 * 
 */
class SchedulerPatchList{public:

    //TODO move variable to private
    u64 _next_patch_id = 0;

    /**
     * @brief contain the list of all patches in the simulation
     */
    std::vector<shamrock::patch::Patch> global;

    /**
     * @brief contain the list of patch owned by the current node
     */
    std::vector<shamrock::patch::Patch> local;

    bool is_load_values_up_to_date = false;

    inline void invalidate_load_values(){
        is_load_values_up_to_date = false;
    }
    
    inline void check_load_values_valid(SourceLocation loc = SourceLocation{}){
        if(!is_load_values_up_to_date){
            throw shambase::make_except_with_loc<std::runtime_error>("the load values are invalid please update them",loc);
        }
    }

    /**
     * @brief rebuild global from the local list of each tables
     *  
     * similar to \p global = allgather(\p local)
     */
    void build_global();


    /**
     * @brief select owned patches owned by the node to rebuild local
     * 
     * @return std::unordered_set<u64> 
     */
    [[nodiscard]]
    std::unordered_set<u64> build_local();

    
    /**
     * @brief 
     * 
     * @param old_patchid old owned patch_id list
     * @param to_send_idx vector that will be filled with index in patch global vector of the patchdata that sould be sent
     * @param to_recv_idx vector that will be filled with index in patch global vector of the patchdata that sould be received
     * @param new_patchid new owned patch_id list from global patch list
     */
    void build_local_differantial(
        std::unordered_set<u64> & patch_id_lst, 
        std::vector<u64> & to_send_idx,
        std::vector<u64> & to_recv_idx
        );

    /**
     * @brief id_patch_to_global_idx[patch_id] = index in global patch list
     */
    std::unordered_map<u64,u64> id_patch_to_global_idx;

    /**
     * @brief id_patch_to_local_idx[patch_id] = index in local patch list
     */
    std::unordered_map<u64,u64> id_patch_to_local_idx;

    /**
     * @brief recompute id_patch_to_global_idx
     * 
     */
    void build_global_idx_map();

    /**
     * @brief recompute id_patch_to_local_idx
     */
    void build_local_idx_map();

    /**
     * @brief reset Patch's pack index value
     */
    void reset_local_pack_index();

    /** 
     * @brief split the Patch having id_patch as id and return the index of the 8 subpatches in the global vector
     * 
     * @param id_patch the id of the patch to split
     * @return std::tuple<u64,u64,u64,u64,u64,u64,u64,u64> the index of the 8 splitted in the global vector
     */
    std::tuple<u64,u64,u64,u64,u64,u64,u64,u64> split_patch(u64 id_patch);

    /**
     * @brief merge the 8 given patches index in the global vector    
     * 
     * Note : the first one will contain the merge patch the 7 others will be set with node_owner_id = u32_max, and then be flushed out when doing build local / sync global
     * 
     * @param idx... the 8 patches index
     */
    void merge_patch( u64 idx0, u64 idx1, u64 idx2, u64 idx3, u64 idx4, u64 idx5, u64 idx6, u64 idx7);

    

};


/**
 * @brief generate a fake patch list corresponding to a tree structure
 * 
 * @param total_dtcnt total data count  
 * @param div_limit data count limit to split
 * @return std::vector<Patch> the fake patch list
 */
std::vector<shamrock::patch::Patch> make_fake_patch_list(u32 total_dtcnt,u64 div_limit);