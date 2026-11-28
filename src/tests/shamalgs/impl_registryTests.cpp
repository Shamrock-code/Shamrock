// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright (c) 2021-2026 Timothée David--Cléris <tim.shamrock@proton.me>
// SPDX-License-Identifier: CeCILL Free Software License Agreement v2.1
// Shamrock is licensed under the CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

#include "shamalgs/ImplVariant.hpp"
#include "shamalgs/impl_registry.hpp"
#include "shamcomm/logs.hpp"
#include "shamsys/NodeInstance.hpp"
#include "shamtest/shamtest.hpp"
#include <algorithm>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
    /// Tag alternative of the dummy selectors built by the test below
    struct DummyAlt {
        static constexpr std::string_view variant_type_name = "dummy";
    };
} // namespace

NEW_TEST(Unittest, "shamalgs/impl_registry", 1) {

    namespace reg = shamalgs::impl_registry;

    std::vector<std::string> algs = reg::get_registered_algs();

    REQUIRE(std::is_sorted(algs.begin(), algs.end()));

    // every algorithm registers itself at static initialization
    for (const char *name :
         {"reduction",
          "is_all_true",
          "scan_exclusive_sum_in_place",
          "segmented_sort_in_place",
          "sort_by_key_pow2_len",
          "sort_by_keys",
          "compute_histogram",
          "clbvh_dual_tree_traversal"}) {
        REQUIRE_NAMED(
            std::string("registered: ") + name,
            std::find(algs.begin(), algs.end(), name) != algs.end());
    }

    // round trip through every implementation of every registered algorithm
    for (const std::string &alg : algs) {
        shamlog_info_ln("tests", "testing registry round trip of:", alg);

        // autoselect first, since a saved "null" could not be restored
        reg::autoselect_impl(alg, shamsys::instance::get_compute_scheduler_ptr());
        REQUIRE_NAMED("is_impl_set: " + alg, reg::is_impl_set(alg));

        std::string saved = reg::get_current_impl(alg);

        for (const std::string &impl : reg::get_default_impl_list(alg)) {
            reg::set_impl(alg, impl);
            REQUIRE_EQUAL_NAMED("set/get round trip: " + alg, reg::get_current_impl(alg), impl);
        }

        reg::set_impl(alg, saved);
        REQUIRE_EQUAL_NAMED("restored: " + alg, reg::get_current_impl(alg), saved);
    }

    // unknown algorithm names are rejected by every function
    const std::string unknown = "not_a_registered_algorithm";
    REQUIRE_EXCEPTION_THROW(reg::get_default_impl_list(unknown), std::invalid_argument);
    REQUIRE_EXCEPTION_THROW(reg::get_current_impl(unknown), std::invalid_argument);
    REQUIRE_EXCEPTION_THROW(reg::is_impl_set(unknown), std::invalid_argument);
    REQUIRE_EXCEPTION_THROW(
        ([&] {
            reg::set_impl(unknown, R"({"implementation":"dummy","parameters":{}})");
        })(),
        std::invalid_argument);
    REQUIRE_EXCEPTION_THROW(
        ([&] {
            reg::autoselect_impl(unknown, shamsys::instance::get_compute_scheduler_ptr());
        })(),
        std::invalid_argument);

    // a null scheduler is rejected before reaching the algorithm's autoselect rule
    REQUIRE_EXCEPTION_THROW(reg::autoselect_impl("reduction", nullptr), std::runtime_error);

    // an empty autoselect rule is rejected at construction
    REQUIRE_EXCEPTION_THROW(
        ([] {
            shamalgs::ImplVariantGlobal<DummyAlt> empty{
                shamalgs::ImplVariantGlobal<DummyAlt>::AutoselectFn{}};
        })(),
        std::invalid_argument);

    // registering an already registered name throws and leaves the registry unchanged.
    // Never register a test-local object under a new name: there is no unregister, so its entry
    // would dangle once the object goes out of scope.
    {
        std::string reduction_before = reg::get_current_impl("reduction");

        shamalgs::ImplVariantGlobal<DummyAlt> dummy{
            [](const sham::DeviceScheduler_ptr &, auto &self) {
                self.set(DummyAlt{});
            }};

        REQUIRE_EXCEPTION_THROW(
            ([&] {
                reg::register_impl("reduction", dummy);
            })(),
            std::invalid_argument);

        REQUIRE(reg::get_registered_algs() == algs);
        REQUIRE_EQUAL(reg::get_current_impl("reduction"), reduction_before);
    }
}
