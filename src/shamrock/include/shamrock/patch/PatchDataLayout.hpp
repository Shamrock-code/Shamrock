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
 * @file PatchDataLayout.hpp
 * @author Timothée David--Cléris (tim.shamrock@proton.me)
 * @author Yona Lapeyre (yona.lapeyre@ens-lyon.fr)
 * @brief
 */

#include "shambase/memory.hpp"
#include "shamrock/patch/PatchDataLayerLayout.hpp"
#include <initializer_list>
#include <memory>
#include <vector>

namespace shamrock::patch {

    /**
     * @brief PatchDataLayerLayout container class
     * This class store the layout of the layers
     */
    class PatchDataLayout {
        public:
        struct LayerEntry {
            std::shared_ptr<shamrock::patch::PatchDataLayerLayout> layout;
            std::string name;
        };

        std::vector<LayerEntry> layer_layouts;

        PatchDataLayout() = default;

        PatchDataLayout(const std::vector<std::string> &layer_names)
            : layer_layouts{layer_names.size()} {
            for (size_t i = 0; i < layer_layouts.size(); i++) {
                layer_layouts[i].name   = layer_names[i];
                layer_layouts[i].layout = std::make_shared<PatchDataLayerLayout>();
            }
        }

        PatchDataLayout(std::initializer_list<std::string> layer_names)
            : PatchDataLayout(std::vector<std::string>(layer_names)) {}

        size_t get_layer_count() const { return layer_layouts.size(); }

        size_t get_layer_index(const std::string &name) const {
            for (size_t i = 0; i < layer_layouts.size(); i++) {
                if (layer_layouts[i].name == name) {
                    return i;
                }
            }
            throw shambase::make_except_with_loc<std::invalid_argument>(
                "the requested layer does not exists");
        }

        inline std::shared_ptr<PatchDataLayerLayout> &get_layer_ptr(size_t idx) {
            return layer_layouts.at(idx).layout;
        }

        inline const std::shared_ptr<PatchDataLayerLayout> &get_layer_ptr(size_t idx) const {
            return layer_layouts.at(idx).layout;
        }

        inline PatchDataLayerLayout &get_layer_ref(size_t idx) {
            return shambase::get_check_ref(get_layer_ptr(idx));
        }

        inline std::shared_ptr<PatchDataLayerLayout> &get_layer_ptr(const std::string &name) {
            return get_layer_ptr(get_layer_index(name));
        }

        inline const std::shared_ptr<PatchDataLayerLayout> &get_layer_ptr(
            const std::string &name) const {
            return get_layer_ptr(get_layer_index(name));
        }

        inline PatchDataLayerLayout &get_layer_ref(const std::string &name) {
            return get_layer_ref(get_layer_index(name));
        }
    };

    /**
     * @brief Serialize a PatchDataLayout object to a JSON object
     *
     * This function takes a PatchDataLayout object and serializes it to a JSON object.
     * It is used to convert the PatchDataLayout object to a JSON string.
     *
     * @param j The JSON object to serialize the PatchDataLayout object to
     * @param p The PatchDataLayout object to serialize
     */
    inline void to_json(nlohmann::json &j, const PatchDataLayout &p) {
        using json = nlohmann::json;
        std::vector<json> layer_entries;

        for (const auto &layer_entry : p.layer_layouts) {

            json layer_json;
            to_json(layer_json, shambase::get_check_ref(layer_entry.layout));
            layer_entries.push_back(json{{"name", layer_entry.name}, {"layout", layer_json}});
        }

        j = layer_entries;
    }

    /**
     * @brief Deserialize a PatchDataLayout object from a JSON object
     *
     * This function takes a JSON object and deserializes it to a PatchDataLayout object.
     * It is used to convert a JSON string to a PatchDataLayout object.
     *
     * @param j The JSON object to deserialize the PatchDataLayout object from
     * @param p The PatchDataLayout object to deserialize
     */
    inline void from_json(const nlohmann::json &j, PatchDataLayout &p) {
        p.layer_layouts.clear();

        logger::raw_ln(j.dump(4));
    }

} // namespace shamrock::patch
