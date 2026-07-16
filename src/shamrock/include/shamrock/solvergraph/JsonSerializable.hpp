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
 * @file JsonSerializable.hpp
 * @author Timothée David--Cléris (tim.shamrock@proton.me)
 * @brief
 */

#include <nlohmann/json.hpp>
#include <concepts>
#include <functional>
#include <memory>
#include <stdexcept>
#include <string>
#include <unordered_map>

namespace shamrock::solvergraph {
    struct JsonSerializable {
        virtual ~JsonSerializable() {};

        virtual void to_json(nlohmann::json &j) = 0;
        virtual std::string type_name()         = 0;

        static std::unique_ptr<JsonSerializable> from_json(const nlohmann::json &j);
    };

    template<typename T>
    concept JsonDeserializable = std::derived_from<T, JsonSerializable> &&
        requires(const nlohmann::json &j) {
            { T::from_json(j) } -> std::convertible_to<T>;
        };

    class JsonSerializable_registry {

        using Factory = std::function<std::unique_ptr<JsonSerializable>(const nlohmann::json &)>;
        std::unordered_map<std::string, Factory> factories;

        public:
        static JsonSerializable_registry &instance() {
            static JsonSerializable_registry registry;
            return registry;
        }

        template<JsonDeserializable T>
        void registerType(const std::string &name) {
            factories[name] = [](const nlohmann::json &j) -> std::unique_ptr<JsonSerializable> {
                return std::make_unique<T>(T::from_json(j));
            };
        }

        std::unique_ptr<JsonSerializable> create(
            const std::string &type, const nlohmann::json &data) const {
            auto it = factories.find(type);

            if (it == factories.end())
                throw std::runtime_error("Unknown type: " + type);

            return it->second(data);
        }
    };

    std::unique_ptr<JsonSerializable> JsonSerializable::from_json(const nlohmann::json &j) {
        const std::string type = j.at("type").get<std::string>();
        return JsonSerializable_registry::instance().create(type, j);
    }

} // namespace shamrock::solvergraph
