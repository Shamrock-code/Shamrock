// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright (c) 2021-2026 Timothée David--Cléris <tim.shamrock@proton.me>
// SPDX-License-Identifier: CeCILL Free Software License Agreement v2.1
// Shamrock is licensed under the CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

/**
 * @file PatchDataLayout.cpp
 * @author Timothée David--Cléris (tim.shamrock@proton.me)
 * @author Yona Lapeyre (yona.lapeyre@ens-lyon.fr) --no git blame--
 * @brief
 */

#include "shamrock/patch/PatchDataLayout.hpp"

namespace shamrock::patch {

    void to_json(nlohmann::json &j, const PatchDataLayout &p) {
        using json = nlohmann::json;
        std::vector<json> layer_entries;

        for (const auto &layer_entry : p.layer_layouts) {
            json layer_json;
            to_json(layer_json, shambase::get_check_ref(layer_entry.layout));
            layer_entries.push_back(json{{"name", layer_entry.name}, {"layout", layer_json}});
        }

        j = layer_entries;
    }

    void from_json(const nlohmann::json &j, PatchDataLayout &p) {
        p.layer_layouts.clear();
        using json = nlohmann::json;

        try {

            std::vector<json> layer_entries = j.get<std::vector<json>>();

            for (const auto &layer_entry : layer_entries) {
                std::string name            = layer_entry.at("name").get<std::string>();
                PatchDataLayerLayout layout = layer_entry.at("layout").get<PatchDataLayerLayout>();
                p.layer_layouts.push_back(
                    PatchDataLayout::LayerEntry{
                        std::make_shared<PatchDataLayerLayout>(layout), name});
            }
        } catch (const json::exception &e) {
            logger::warn_ln(
                "PatchDataLayout",
                shambase::format(
                    "failed to deserialize PatchDataLayout from JSON, try to "
                    "deserialize as single layer :\n error : {}\n json : {}\n",
                    e.what(),
                    j.dump(4)));

            PatchDataLayerLayout pdl_layer;
            from_json(j, pdl_layer);
            p.layer_layouts.push_back(
                PatchDataLayout::LayerEntry{
                    std::make_shared<PatchDataLayerLayout>(pdl_layer), "main"});

            logger::warn_ln("PatchDataLayout", "successfuly deserialized as single layer");
        }
    }

} // namespace shamrock::patch
