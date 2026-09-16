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
 * @file ImplVariant.hpp
 * @author Timothée David--Cléris (tim.shamrock@proton.me)
 * @brief Generic std::variant-based implementation selector.
 *
 * Lets an algorithm expose its set of possible implementations as a plain
 * `std::variant<Alt1, Alt2, ...>` of small structs (mirroring a Rust enum,
 * variants with fields included) instead of hand-writing an enum plus the
 * name <-> enum switch/if-else chains that come with it. Each alternative
 * only needs a `static constexpr std::string_view variant_type_name`
 * (the same convention already used by the std::variant-based configs in
 * shammodels, e.g. AVConfig.hpp).
 *
 * The free functions variant_to_config_string / variant_from_config_string /
 * variant_default_type_names turn any such variant into a single config
 * string ABI: `{"implementation": "<name>", "parameters": <alternative's own json>}`.
 * ImplVariantGlobal builds on top of them to replace the hand-rolled
 * "global variable + enum + name mapping + 3 free functions" pattern used to
 * hold an algorithm's currently selected implementation; see its own doc
 * comment for the two ABI flavors it exposes and how the unset state works.
 *
 * Dispatch on the selected implementation is then a plain
 * `std::visit(shambase::overloaded{...}, variant)` instead of a switch.
 */

#include "shambase/exception.hpp"
#include "sham/format/format.hpp"
#include <fmt/ranges.h>
#include <nlohmann/json.hpp>
#include <string_view>
#include <concepts>
#include <optional>
#include <string>
#include <utility>
#include <variant>
#include <vector>

namespace shamalgs {

    /**
     * @brief Customization point controlling how an alternative's fields (if any) are
     * serialized to / parsed from the "parameters" json value carried alongside its name.
     *
     * The default (used by empty/tag alternatives, i.e. most of them) serializes to an
     * empty json object. Alternatives with tunable fields (e.g. a group size) specialize
     * this trait, for example:
     *
     * @code{.cpp}
     * struct GpuTeamFetching {
     *     static constexpr std::string_view variant_type_name = "gpu_team_fetching";
     *     u32 group_size = 128;
     * };
     *
     * template<>
     * struct shamalgs::ImplVariantParams<GpuTeamFetching> {
     *     static nlohmann::json to_json(const GpuTeamFetching &p) {
     *         return {{"group_size", p.group_size}};
     *     }
     *     static GpuTeamFetching from_json(const nlohmann::json &j) {
     *         GpuTeamFetching p{};
     *         if (j.contains("group_size")) {
     *             p.group_size = j.at("group_size").get<u32>();
     *         }
     *         return p;
     *     }
     * };
     * @endcode
     */
    template<class Alt>
    struct ImplVariantParams {
        /// Serialize the alternative's fields (default: no fields, empty object)
        static inline nlohmann::json to_json(const Alt &) { return nlohmann::json::object(); }
        /// Parse the alternative's fields back (default: no fields, ignored)
        static inline Alt from_json(const nlohmann::json &) { return Alt{}; }
    };

    /**
     * @brief Detects whether Alt opts into exposing more than one default instance of itself
     * (e.g. the same alternative with different tunable field values) via a static
     * `variant_custom_defaults()` method, for example:
     *
     * @code{.cpp}
     * struct GpuTeamFetching {
     *     static constexpr std::string_view variant_type_name = "gpu_team_fetching";
     *     u32 group_size = 128;
     *
     *     static std::vector<GpuTeamFetching> variant_custom_defaults() {
     *         return {GpuTeamFetching{128}, GpuTeamFetching{256}};
     *     }
     * };
     * @endcode
     *
     * Alternatives that do not define it (the common case) keep exposing exactly one,
     * default-constructed instance.
     */
    template<class Alt>
    concept HasCustomDefaults = requires {
        { Alt::variant_custom_defaults() } -> std::convertible_to<std::vector<Alt>>;
    };

    // Forward declaration: defined below, needed by impl_variant_alts::default_config_list()
    template<class Variant>
    inline std::string variant_to_config_string(const Variant &v);

    namespace details {
        /// The instances of Alt to expose as "available implementations": a single
        /// default-constructed one, unless Alt opts into HasCustomDefaults.
        template<class Alt>
        inline std::vector<Alt> alt_default_list() {
            if constexpr (HasCustomDefaults<Alt>) {
                return Alt::variant_custom_defaults();
            } else {
                return {Alt{}};
            }
        }

        /// Partial specialization target: extracts the Alts... pack out of std::variant<Alts...>
        template<class Variant>
        struct impl_variant_alts;

        template<class... Alts>
        struct impl_variant_alts<std::variant<Alts...>> {
            static inline std::vector<std::string> default_type_names() {
                return {std::string(Alts::variant_type_name)...};
            }

            /// List the available implementations as config json strings. Most alternatives
            /// contribute exactly one entry; alternatives with HasCustomDefaults contribute one
            /// entry per instance in their variant_custom_defaults() list.
            static inline std::vector<std::string> default_config_list() {
                std::vector<std::string> out;
                auto add_alt = [&]<class Alt>() {
                    for (auto &alt : alt_default_list<Alt>()) {
                        out.push_back(
                            variant_to_config_string<std::variant<Alts...>>(
                                std::variant<Alts...>{std::move(alt)}));
                    }
                };
                (add_alt.template operator()<Alts>(), ...);
                return out;
            }

            static inline std::variant<Alts...> from_config_string(std::string_view s) {
                nlohmann::json j      = nlohmann::json::parse(s);
                std::string name      = j.at("implementation").get<std::string>();
                nlohmann::json params = j.value("parameters", nlohmann::json::object());

                std::optional<std::variant<Alts...>> result;
                (void) ((name == Alts::variant_type_name
                             ? (result
                                = std::variant<Alts...>{ImplVariantParams<Alts>::from_json(params)},
                                true)
                             : false)
                        || ...);

                if (!result) {
                    throw shambase::make_except_with_loc<std::invalid_argument>(sham::format(
                        "invalid implementation : {}, possible implementations : {}",
                        name,
                        default_type_names()));
                }
                return *result;
            }
        };
    } // namespace details

    /// Serialize the currently active alternative of a variant to a single config json string
    template<class Variant>
    inline std::string variant_to_config_string(const Variant &v) {
        return std::visit(
            [](const auto &alt) {
                using Alt = std::decay_t<decltype(alt)>;
                nlohmann::json j;
                j["implementation"] = std::string(Alt::variant_type_name);
                j["parameters"]     = ImplVariantParams<Alt>::to_json(alt);
                return j.dump();
            },
            v);
    }

    /// Parse a variant back from a config string produced by variant_to_config_string
    template<class Variant>
    inline Variant variant_from_config_string(std::string_view s) {
        return details::impl_variant_alts<Variant>::from_config_string(s);
    }

    /// List the variant_type_name of every alternative of a variant, with no params
    template<class Variant>
    inline std::vector<std::string> variant_default_type_names() {
        return details::impl_variant_alts<Variant>::default_type_names();
    }

    /**
     * @brief Non-template virtual interface exposed by ImplVariantGlobal, for code that needs
     * to hold or pass around an implementation selector without knowing its alternative types.
     */
    class IImplVariant {
        public:
        virtual ~IImplVariant() = default;

        /// Get the currently selected implementation as a single config json string, or a json
        /// null if no implementation has been selected yet
        virtual std::string get_current_config() const = 0;

        /// List the available implementations as config json strings, one per alternative
        virtual std::vector<std::string> get_default_config_list() const = 0;

        /// Select an implementation from a {"implementation": ..., "parameters": ...} json string
        virtual void set(std::string_view config_json) = 0;
    };

    /**
     * @brief Drop-in replacement for the hand-rolled "global variable + enum + name mapping
     * + 3 free functions" implementation-selection pattern.
     *
     * Holds the currently selected implementation as a std::variant<Alts...> and exposes it
     * through a single config json string ABI, so that an algorithm's
     * get_default_impl_list_X / get_current_impl_X / set_impl_X free functions become
     * one-liners:
     *   - get_current_config() / get_default_config_list() / set(string_view) : a single
     *     `{"implementation": ..., "parameters": ...}` json string ABI.
     *
     * No implementation is selected at construction: this class has no notion of a default.
     * is_set() reports whether one has been picked yet. It is up to each call site to decide
     * what to do when unset - typically checking is_set() and calling set() with that
     * algorithm's own default right before dispatching (see e.g.
     * segmented_sort_in_place.cpp). get() assumes is_set(); get_current_config() is the one
     * exception and safely returns a json null instead of dereferencing an unset selection.
     *
     * @tparam Alts the alternative types, each requiring a
     * `static constexpr std::string_view variant_type_name`
     */
    template<class... Alts>
    class ImplVariantGlobal : public IImplVariant {
        public:
        using Variant = std::variant<Alts...>;

        /// Whether an implementation has been selected yet
        inline bool is_set() const { return current.has_value(); }

        /// Get the currently selected implementation. Requires is_set()
        inline const Variant &get() const { return *current; }

        /// Get the currently selected implementation as a single config json string, or a json
        /// null if no implementation has been selected yet (see is_set())
        inline std::string get_current_config() const override {
            if (!is_set()) {
                return nlohmann::json(nullptr).dump();
            }
            return variant_to_config_string(get());
        }

        /// List the available implementations as config json strings. Most alternatives
        /// contribute exactly one entry; alternatives opting into HasCustomDefaults
        /// contribute one entry per instance in their variant_custom_defaults() list.
        inline std::vector<std::string> get_default_config_list() const override {
            return details::impl_variant_alts<Variant>::default_config_list();
        }

        /// Directly select an alternative (e.g. to seed a default at the call site)
        inline void set(Variant v) { current = std::move(v); }

        /// Select an implementation from a {"implementation": ..., "parameters": ...} json string
        inline void set(std::string_view config_json) override {
            current = variant_from_config_string<Variant>(config_json);
        }

        private:
        std::optional<Variant> current;
    };

} // namespace shamalgs
