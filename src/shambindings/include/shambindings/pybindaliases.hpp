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
 * @file pybindaliases.hpp
 * @author Timothée David--Cléris (tim.shamrock@proton.me)
 * @brief Pybind11 include and definitions
 *
 * If we build shamrock executable we embed python in Shamrock
 * hence the include pybind11/embed.h.
 * If we build shamrock as python lib we import pybind11/pybind11.h.
 * Both options defines a similar syntax for the python module definition,
 * we can then wrap them conveniently in a single macro call.
 */

#include "shambase/call_lambda.hpp"
#include "shambase/exception.hpp"
#include "shambase/unique_name_macro.hpp"
#include <pybind11/pybind11.h>
#include <map>
#include <optional>

/// alias to pybind11 namespace
namespace py = pybind11;

// ------------------------------------------------------------------
// Submodule registry
// ------------------------------------------------------------------

namespace shambindings::submodules {

    // ------------------------------------------------------------------
    // Generic registry
    // ------------------------------------------------------------------

    template<typename T>
    struct registry_t {

        // note here that std::less<> is for transparent lookup with string_view
        using map_t                    = std::map<std::string, T, std::less<>>;
        std::unique_ptr<map_t> storage = {};

        using getter_fn = std::function<T &()>;
        std::optional<std::map<std::string, getter_fn, std::less<>>> overrides;

        map_t &data() {
            if (!storage) {
                storage = std::make_unique<map_t>();
            }
            return *storage;
        }

        void reset() {
            storage.reset();
            overrides = std::nullopt;
        }

        // enable override system lazily
        auto &override_map() {
            if (!overrides) {
                overrides.emplace();
            }
            return *overrides;
        }

        void set_override(std::string_view key, getter_fn fn) {
            override_map().insert_or_assign(std::string(key), std::move(fn));
        }

        void clear_overrides() { overrides = std::nullopt; }

        // does not include overrides
        std::vector<std::string> keys() const {
            if (!storage) {
                return {};
            }

            std::vector<std::string> out;
            out.reserve(storage->size());

            for (auto const &kv : *storage) {
                out.push_back(kv.first);
            }

            return out;
        }

        T &get(std::string_view key) {

            // global override first
            if (overrides) {
                if (auto it = overrides->find(key); it != overrides->end()) {
                    return it->second();
                }
            }

            auto &map = data();

            if (auto it = map.find(key); it != map.end()) {
                return it->second;
            }

            throw shambase::make_except_with_loc<std::out_of_range>(
                "registry entry not found: " + std::string(key));
        }

        void insert(std::string_view key, T &&value) {
            auto [it, inserted] = data().emplace(std::string(key), std::move(value));

            if (!inserted) {
                throw shambase::make_except_with_loc<std::runtime_error>(
                    "registry entry already exists: " + std::string(key));
            }
        }
    };

    // ------------------------------------------------------------------
    // Registry types
    // ------------------------------------------------------------------

    using module_factory_t = std::function<py::module()>;

    // ------------------------------------------------------------------
    // Global registries
    // ------------------------------------------------------------------

    registry_t<py::module> &modules();

    registry_t<module_factory_t> &builders();

    // ------------------------------------------------------------------
    // Global build call
    // ------------------------------------------------------------------

    inline void build_all_modules() {
        auto &module_map = modules().data();

        // Get snapshot of builder keys
        std::vector<std::string> keys = builders().keys();

        // Lexicographic order
        std::sort(keys.begin(), keys.end());

        for (auto const &key : keys) {
            auto &builder = builders().get(key);
            module_map.emplace(key, builder());
        }
    }

} // namespace shambindings::submodules

/// function signature used to register python modules
using fct_sig = std::function<void(py::module &)>;

/// Register a python module init function to be ran on init
void register_pybind_init_func(fct_sig);

/**
 * @brief Internal helper that creates static symbols to register a Python init function via a
 * static initializer. It declares `funcname`, creates a `call_lambda` object that calls
 * `register_pybind_init_func(funcname)` at startup, and defines `funcname`.
 *
 * Here the objects/func are static in order to avoid conflicting name in linking. This is similar
 * to anonymous namespaces
 */
#define _internal_register_pybind_init(funcname, lambda_name, varname)                             \
    static void funcname(py::module &varname);                                                     \
    static shambase::call_lambda lambda_name([]() {                                                \
        register_pybind_init_func(funcname);                                                       \
    });                                                                                            \
    static void funcname(py::module &varname)

/**
 * @brief Register a Python module init function using static initialization
 *
 * Generates unique symbols automatically, making it convenient for one-shot initializations in
 * .cpp files.
 *
 * Usage (in a .cpp file) :
 * @code{.cpp}
 *
 * ON_PYTHON_INIT {
 *
 *    // Define things in the python module object `root_module` like so :
 *    root_module.def("hello", []() { return "Hello from SHAMROCK!"; });
 *
 * }
 * @endcode
 */
#define ON_PYTHON_INIT                                                                             \
    _internal_register_pybind_init(                                                                \
        __shamrock_unique_name(pybind_), __shamrock_unique_name(pybind_class_obj_), root_module)


#define Register_pymodsubmodule_int(path, funcname, lambda_name)                                   \
    py::module funcname();                                                                         \
    shambase::call_lambda lambda_name([]() {                                                       \
        shambindings::submodules::builders().insert(path, funcname);                               \
    });                                                                                            \
    py::module funcname()

#define Register_pymodsubmodule(path)                                                              \
    Register_pymodsubmodule_int(                                                                   \
        path, __shamrock_unique_name(pybind_class_obj_), __shamrock_unique_name(pybind_))
