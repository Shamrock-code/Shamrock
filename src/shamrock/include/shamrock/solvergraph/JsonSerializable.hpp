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
 * @brief Polymorphic JSON serialization via a type registry.
 *
 * Provides a base interface and a singleton registry so derived types can be
 * serialized to JSON and reconstructed polymorphically from a `"type"` field.
 */

#include <nlohmann/json.hpp>
#include <unordered_map>
#include <concepts>
#include <functional>
#include <memory>
#include <stdexcept>
#include <string>

namespace shamrock::solvergraph {

    /**
     * @brief Base interface for types that can be serialized to and from JSON.
     *
     * Derived types must implement `to_json` and `type_name`, and also provide a
     * **static** `from_json` that returns an instance of the derived type (enforced
     * by @ref JsonDeserializable when registering).
     *
     * The JSON object used for polymorphic deserialization must contain a `"type"`
     * field whose value matches the name passed to
     * @ref JsonSerializable_registry::register_type and the string returned by
     * `type_name()`.
     *
     * @code{.cpp}
     * struct MyType : shamrock::solvergraph::JsonSerializable {
     *     int value;
     *
     *     void to_json(nlohmann::json &j) const override { j["value"] = value; }
     *     std::string type_name() const override { return "MyType"; }
     *
     *     static MyType from_json(const nlohmann::json &j) {
     *         return MyType{j.at("value").get<int>()};
     *     }
     * };
     *
     * // Register once (name must match type_name() / JSON "type")
     * JsonSerializable_registry::instance().register_type<MyType>("MyType");
     *
     * // Serialize
     * MyType obj{42};
     * nlohmann::json j;
     * obj.to_json(j);
     * j["type"] = obj.type_name();
     *
     * // Deserialize polymorphically
     * auto ptr = JsonSerializable::from_json(j);
     * auto *restored = dynamic_cast<MyType *>(ptr.get());
     * @endcode
     */
    struct JsonSerializable {
        virtual ~JsonSerializable() = default;

        /**
         * @brief Write this object's fields into a JSON object.
         *
         * Does not need to set the `"type"` discriminator; the caller typically
         * adds `j["type"] = type_name()` after this call.
         *
         * @param j JSON object to fill
         */
        virtual void to_json(nlohmann::json &j) const = 0;

        /**
         * @brief Return the type discriminator string for this class.
         *
         * Must match the name used in @ref JsonSerializable_registry::register_type
         * and the `"type"` field in JSON.
         */
        virtual std::string type_name() const = 0;

        /**
         * @brief Reconstruct a registered type from JSON.
         *
         * Reads `j.at("type")` and dispatches to the matching factory in
         * @ref JsonSerializable_registry.
         *
         * @param j JSON object containing at least a string `"type"` field
         * @return Owned instance of the matching registered type
         * @throws std::runtime_error if `j` is not an object with a string `"type"`
         *         field, or if the type name is not registered
         */
        static std::unique_ptr<JsonSerializable> from_json(const nlohmann::json &j);
    };

    /**
     * @brief Concept for types that can be registered for polymorphic JSON deserialization.
     *
     * Requires that `T` derives from @ref JsonSerializable and provides a static
     * `T::from_json(const nlohmann::json &)` convertible to `T`.
     */
    template<typename T>
    concept JsonDeserializable
        = std::derived_from<T, JsonSerializable> && requires(const nlohmann::json &j) {
              { T::from_json(j) } -> std::convertible_to<T>;
          };

    /**
     * @brief Singleton registry mapping type names to JSON factory functions.
     *
     * Types are registered with @ref register_type. Polymorphic reconstruction goes
     * through @ref JsonSerializable::from_json, which calls @ref create.
     */
    class JsonSerializable_registry {

        using Factory = std::function<std::unique_ptr<JsonSerializable>(const nlohmann::json &)>;
        std::unordered_map<std::string, Factory> factories;

        public:
        /**
         * @brief Access the process-wide registry instance.
         */
        static JsonSerializable_registry &instance() {
            static JsonSerializable_registry registry;
            return registry;
        }

        /**
         * @brief Register a type under a string name for polymorphic deserialization.
         *
         * The name must match `T::type_name()` and the JSON `"type"` field.
         * `T` must satisfy @ref JsonDeserializable.
         *
         * @tparam T Concrete type deriving from JsonSerializable
         * @param name Type discriminator string
         */
        template<JsonDeserializable T>
        void register_type(const std::string &name) {
            factories[name] = [](const nlohmann::json &j) -> std::unique_ptr<JsonSerializable> {
                return std::make_unique<T>(T::from_json(j));
            };
        }

        /**
         * @brief Create an instance of a registered type from JSON.
         *
         * @param type Type discriminator (must have been passed to @ref register_type)
         * @param data Full JSON object passed to the type's static `from_json`
         * @return Owned instance of the registered type
         * @throws std::runtime_error if `type` is not registered
         */
        std::unique_ptr<JsonSerializable> create(
            const std::string &type, const nlohmann::json &data) const {
            auto it = factories.find(type);

            if (it == factories.end())
                throw std::runtime_error("Unknown type: " + type);

            return it->second(data);
        }
    };

    inline std::unique_ptr<JsonSerializable> JsonSerializable::from_json(const nlohmann::json &j) {
        if (!j.is_object() || !j.contains("type") || !j["type"].is_string()) {
            throw std::runtime_error(
                "Invalid JSON for deserialization: expected an object with a string 'type' field.");
        }
        const std::string type = j.at("type").get<std::string>();
        return JsonSerializable_registry::instance().create(type, j);
    }

} // namespace shamrock::solvergraph
