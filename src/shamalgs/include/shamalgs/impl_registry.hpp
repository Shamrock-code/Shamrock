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
 * @file impl_registry.hpp
 * @author Timothée David--Cléris (tim.shamrock@proton.me)
 * @brief Name-keyed registry of the implementation selectors (IImplVariant) of every algorithm.
 *
 * Each algorithm registers its selector once, at static initialization, right after defining it:
 *
 * @code{.cpp}
 * constexpr std::string_view my_algo_impl_name = "my_algo";
 *
 * shamalgs::ImplVariantGlobal<AltA, AltB> my_algo_impl{
 *     [](const sham::DeviceScheduler_ptr &, auto &self) {
 *         self.set(AltA{});
 *     }};
 *
 * // same translation unit, so initialized after my_algo_impl
 * SHAMALGS_REGISTER_IMPL(my_algo_impl_name, my_algo_impl);
 * @endcode
 *
 * The selection of any registered algorithm can then be read or changed by name. Every function
 * taking an algorithm name throws std::invalid_argument if no algorithm is registered under it.
 */

#include "shambase/call_lambda.hpp"
#include "shambase/unique_name_macro.hpp"
#include "shamalgs/ImplVariant.hpp"
#include "shambackends/DeviceScheduler.hpp"
#include <string_view>
#include <string>
#include <vector>

namespace shamalgs::impl_registry {

    /**
     * @brief Register an implementation selector under `name`
     *
     * The registry stores the address of `impl`, which must therefore outlive every later use of
     * the registry (in practice, a namespace-scope global). There is no unregister.
     *
     * @throws std::invalid_argument if `name` is already registered (nothing is stored then)
     */
    void register_impl(std::string name, IImplVariant &impl);

    /// Get the names of every registered algorithm, sorted
    std::vector<std::string> get_registered_algs();

    /// Get the list of available implementations of `alg`, as config json strings
    std::vector<std::string> get_default_impl_list(std::string_view alg);

    /// Get the current implementation of `alg` as a config json string, "null" if unset
    std::string get_current_impl(std::string_view alg);

    /// Whether an implementation of `alg` has been selected yet
    bool is_impl_set(std::string_view alg);

    /// Select the implementation of `alg` from a config json string
    void set_impl(std::string_view alg, std::string_view impl);

    /**
     * @brief Select the default implementation of `alg` for the device behind `sched`
     *
     * @throws std::runtime_error if `sched` is null
     */
    void autoselect_impl(std::string_view alg, const sham::DeviceScheduler_ptr &sched);

} // namespace shamalgs::impl_registry

/**
 * @brief Register the implementation selector `impl` under `name` at static initialization
 *
 * Meant for namespace scope in a .cpp file, right after the definition of `impl`: objects of one
 * translation unit are initialized in definition order, so `impl` is constructed by then. See
 * shamalgs::impl_registry::register_impl for the requirements and errors.
 *
 * Usage :
 * @code{.cpp}
 * SHAMALGS_REGISTER_IMPL(my_algo_impl_name, my_algo_impl);
 * @endcode
 *
 * @param name the registry name (string literal or std::string_view constant)
 * @param impl the namespace-scope IImplVariant to register
 */
#define SHAMALGS_REGISTER_IMPL(name, impl)                                                         \
    [[maybe_unused]] static const shambase::call_lambda __shamrock_unique_name(                    \
        shamalgs_impl_registration_) {                                                             \
        [] {                                                                                       \
            shamalgs::impl_registry::register_impl(std::string(name), impl);                       \
        }                                                                                          \
    }
