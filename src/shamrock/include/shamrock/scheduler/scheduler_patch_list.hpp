// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright (c) 2021-2025 Timothée David--Cléris <tim.shamrock@proton.me>
// SPDX-License-Identifier: CeCILL Free Software License Agreement v2.1
// Shamrock is licensed under the CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

#pragma once

/**
 * @file scheduler_patch_list.hpp
 * @author Timothée David--Cléris (tim.shamrock@proton.me)
 * @brief Class to handle the patch list of the mpi scheduler
 *
 */

#include "shambase/SourceLocation.hpp"
#include "shambase/exception.hpp"
#include "shamrock/patch/Patch.hpp"
#include <nlohmann/json.hpp>
#include <unordered_map>
#include <unordered_set>
#include <array>
#include <stdexcept>
#include <tuple>
#include <vector>

/**
 * @brief Handle the patch list of the mpi scheduler
 *
 */
class SchedulerPatchList {
    public:
    /**
     * @brief The next available patch id
     *
     * This variable is used to assign a unique id to each patch when adding, splitting
     * or merging patches.
     *
     * @todo make this variable to private
     */
    u64 _next_patch_id = 0;

    /**
     * @brief contain the list of all patches in the simulation
     */
    std::vector<shamrock::patch::Patch> global;

    /**
     * @brief contain the list of patch owned by the current node
     */
    std::vector<shamrock::patch::Patch> local;

    bool is_load_values_up_to_date = false; ///< Are patch load values up to date

    /// Invalidate current load values (To use after a change the patches is made)
    inline void invalidate_load_values() { is_load_values_up_to_date = false; }

    /// Check if the load values are valid, throw otherwise
    inline void check_load_values_valid(SourceLocation loc = SourceLocation{}) {
        if (!is_load_values_up_to_date) {
            throw shambase::make_except_with_loc<std::runtime_error>(
                "the load values are invalid please update them", loc);
        }
    }

    /**
     * @brief rebuild global from the local list of each tables
     *
     * similar to #global = allgather(#local)
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
     * @brief Build the local patch list and create a differential of patches to send / recv since
     * last time
     *
     * The method clears the `local` vector. It then iterates over the `global` vector of
     * `shamrock::patch::Patch` objects. For each `Patch` object, it checks if the `id_patch` is in
     * the `patch_id_lst` set. If it is, it means the patch was previously owned by the current
     * node, so it does nothing. If it was not previously owned, it adds the `id_patch` to the
     * `patch_id_lst` set and adds the index `i` to the `to_recv_idx` vector.
     *
     * If the `node_owner_id` of the `Patch` object is not the same as the current node's rank (as
     * determined by `shamcomm::world_rank()`), it checks if the `id_patch` was previously owned by
     * the current node. If it was, it adds the index `i` to the `to_send_idx` vector and removes
     * the `id_patch` from the `patch_id_lst` set.
     *
     * In summary, this method builds the local patch list by comparing the global patch list with
     * the current node's previously owned patches. It also identifies patches that need to be sent
     * to or received from other nodes.
     *
     * @param patch_id_lst The set of patch ids previously owned by the current node
     * @param to_send_idx The vector of indices of patches that need to be sent to other nodes
     * @param to_recv_idx The vector of indices of patches that need to be received from other nodes
     */
    void build_local_differantial(
        std::unordered_set<u64> &patch_id_lst,
        std::vector<u64> &to_send_idx,
        std::vector<u64> &to_recv_idx);

    /**
     * @brief id_patch_to_global_idx[patch_id] = index in global patch list
     */
    std::unordered_map<u64, u64> id_patch_to_global_idx;

    /**
     * @brief id_patch_to_local_idx[patch_id] = index in local patch list
     */
    std::unordered_map<u64, u64> id_patch_to_local_idx;

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
     * @brief split the Patch having id_patch as id and return the index of the 8 subpatches in the
     * global vector
     *
     * @param id_patch the id of the patch to split
     * @return std::tuple<u64,u64,u64,u64,u64,u64,u64,u64> the index of the 8 splitted in the global
     * vector
     */
    std::tuple<u64, u64, u64, u64, u64, u64, u64, u64> split_patch(u64 id_patch);

    /**
     * @brief merge the 8 given patches index in the global vector
     *
     * Note : the first one will contain the merge patch the 7 others will be set with
     * `node_owner_id = u32_max`, and then be flushed out when doing build local / sync global
     *
     * parameters idx... are the 8 patches index in the global patch metadata vector.
     */
    void
    merge_patch(u64 idx0, u64 idx1, u64 idx2, u64 idx3, u64 idx4, u64 idx5, u64 idx6, u64 idx7);
};

/**
 * @brief Serializes a SchedulerPatchList object to a JSON object.
 *
 * @param j The JSON object to serialize to.
 * @param p The SchedulerPatchList object to serialize.
 */
void to_json(nlohmann::json &j, const SchedulerPatchList &p);

/**
 * @brief Deserializes a JSON object into a SchedulerPatchList object.
 *
 * @param j The JSON object to deserialize from.
 * @param p The SchedulerPatchList object to deserialize into.
 */
void from_json(const nlohmann::json &j, SchedulerPatchList &p);

/**
 * @brief generate a fake patch list corresponding to a tree structure
 *
 * @param total_dtcnt total data count
 * @param div_limit data count limit to split
 * @return std::vector<Patch> the fake patch list
 */
std::vector<shamrock::patch::Patch> make_fake_patch_list(u32 total_dtcnt, u64 div_limit);
