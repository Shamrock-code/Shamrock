// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright (c) 2021-2026 Timothée David--Cléris <tim.shamrock@proton.me>
// SPDX-License-Identifier: CeCILL Free Software License Agreement v2.1
// Shamrock is licensed under the CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

/**
 * @file impl_registry.cpp
 * @author Timothée David--Cléris (tim.shamrock@proton.me)
 * @brief Implementation of the name-keyed registry of implementation selectors.
 */

#include "shamalgs/impl_registry.hpp"
#include "shambase/exception.hpp"
#include "shambase/memory.hpp"
#include "shamcomm/logs.hpp"
#include <functional>
#include <map>
#include <stdexcept>
#include <utility>

namespace {

    using Registry = std::map<std::string, shamalgs::IImplVariant *, std::less<>>;

    /// Meyers singleton, created by the first registration, hence destroyed after every global
    Registry &get_registry() {
        static Registry registry;
        return registry;
    }

    /// Comma-separated list of the registered names, for error messages
    std::string registered_names_str() {
        std::string ret;
        for (const auto &[name, impl] : get_registry()) {
            if (!ret.empty()) {
                ret += ", ";
            }
            ret += name;
        }
        return ret;
    }

    /// Get the selector registered under `alg`, or throw listing the registered names
    shamalgs::IImplVariant &get_impl(std::string_view alg) {
        Registry &registry = get_registry();
        auto it            = registry.find(alg);
        if (it == registry.end()) {
            throw shambase::make_except_with_loc<std::invalid_argument>(
                "no implementation registered under the name \"" + std::string(alg)
                + "\", registered names are: [" + registered_names_str() + "]");
        }
        return *it->second;
    }

} // namespace

namespace shamalgs::impl_registry {

    void register_impl(std::string name, IImplVariant &impl) {
        Registry &registry = get_registry();
        if (registry.find(name) != registry.end()) {
            throw shambase::make_except_with_loc<std::invalid_argument>(
                "an implementation is already registered under the name \"" + name + "\"");
        }
        registry.emplace(std::move(name), &impl);
    }

    std::vector<std::string> get_registered_algs() {
        std::vector<std::string> ret;
        ret.reserve(get_registry().size());
        for (const auto &[name, impl] : get_registry()) {
            ret.push_back(name); // std::map iterates in sorted order
        }
        return ret;
    }

    std::vector<std::string> get_default_impl_list(std::string_view alg) {
        return get_impl(alg).get_default_config_list();
    }

    std::string get_current_impl(std::string_view alg) {
        return get_impl(alg).get_current_config();
    }

    bool is_impl_set(std::string_view alg) { return get_impl(alg).is_set(); }

    void set_impl(std::string_view alg, std::string_view impl) {
        get_impl(alg).set(impl);
        shamlog_info_ln("algs", "setting", alg, "implementation to impl :", impl);
    }

    void autoselect_impl(std::string_view alg, const sham::DeviceScheduler_ptr &sched) {
        IImplVariant &impl = get_impl(alg);
        shambase::get_check_ref(sched);
        impl.autoselect(sched);
        shamlog_info_ln(
            "algs", "defaulting", alg, "implementation to impl :", impl.get_current_config());
    }

} // namespace shamalgs::impl_registry
