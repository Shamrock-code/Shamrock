// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright (c) 2021-2026 Timothée David--Cléris <tim.shamrock@proton.me>
// SPDX-License-Identifier: CeCILL Free Software License Agreement v2.1
// Shamrock is licensed under the CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

#pragma once

/**
 * @file enum_NeighCacheStrategy.hpp
 * @author Timothée David--Cléris (tim.shamrock@proton.me)
 * @brief Neighbour cache build strategy enum + json serialization/deserialization
 */

#include "shambase/exception.hpp"
#include "nlohmann/json.hpp"
#include "shamrock/io/json_utils.hpp"
#include <string>

namespace shammodels {

    /**
     * @brief Strategy used to build the neighbour cache out of the tree traversal
     *
     * @note This used to be a `use_two_stage_search` boolean, it is an enum so that more
     * strategies can be added without breaking the config API again.
     */
    enum NeighCacheStrategy {
        SingleStage              = 0, ///< Single tree traversal per particle
        TwoStage                 = 1, ///< Two stage neighbours search (see shamrock paper)
        SingleStageSharedOffload = 2, ///< Single tree traversal per particle, shared memory
                                      ///< offload variant
        TwoStageSharedOffload    = 3  ///< Two stage neighbours search, shared memory offload
                                      ///< variant
    };

    SHAMROCK_JSON_SERIALIZE_ENUM(
        NeighCacheStrategy,
        {{NeighCacheStrategy::SingleStage, "single_stage"},
         {NeighCacheStrategy::TwoStage, "two_stage"},
         {NeighCacheStrategy::SingleStageSharedOffload, "single_stage_shared_offload"},
         {NeighCacheStrategy::TwoStageSharedOffload, "two_stage_shared_offload"}});

    /// Json key holding the neighbour cache strategy
    inline constexpr const char *neigh_cache_strategy_json_key = "neigh_cache_strategy";

    /// Legacy json key, which used to hold the strategy as a boolean
    inline constexpr const char *neigh_cache_strategy_legacy_json_key = "use_two_stage_search";

    /// Map the legacy `use_two_stage_search` boolean onto the strategy enum
    inline NeighCacheStrategy neigh_cache_strategy_from_two_stage_search(
        bool use_two_stage_search) {
        return (use_two_stage_search) ? NeighCacheStrategy::TwoStage
                                      : NeighCacheStrategy::SingleStage;
    }

    /**
     * @brief Deserialize the neighbour cache strategy, falling back on the legacy boolean key
     *
     * Reads `neigh_cache_strategy` if the json object has it. Otherwise the legacy
     * `use_two_stage_search` boolean is mapped onto the enum, so that configs dumped before
     * the strategy became an enum can still be loaded. If neither key is there the value is
     * left as is.
     *
     * @param j The json object to deserialize from
     * @param value The strategy to populate
     * @param log_ctx Logger context used for the key update message
     * @param has_used_defaults Set to true if no key was found
     * @param has_updated_config Set to true if the legacy key was used
     */
    inline void get_to_neigh_cache_strategy(
        const nlohmann::json &j,
        NeighCacheStrategy &value,
        const std::string &log_ctx,
        bool &has_used_defaults,
        bool &has_updated_config) {

        if (j.contains(neigh_cache_strategy_json_key)) {
            j.at(neigh_cache_strategy_json_key).get_to(value);
            return;
        }

        if (j.contains(neigh_cache_strategy_legacy_json_key)) {
            value = neigh_cache_strategy_from_two_stage_search(
                j.at(neigh_cache_strategy_legacy_json_key).template get<bool>());
            has_updated_config = true;
            if (shamcomm::world_rank() == 0) {
                shamcomm::logs::warn_ln(
                    log_ctx,
                    "Updating old key [" + std::string(neigh_cache_strategy_legacy_json_key)
                        + "] to new key [" + std::string(neigh_cache_strategy_json_key)
                        + "] in from_json");
            }
            return;
        }

        has_used_defaults = true;
    }

} // namespace shammodels
