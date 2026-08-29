// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright (c) 2021-2026 Timothée David--Cléris <tim.shamrock@proton.me>
// SPDX-License-Identifier: CeCILL Free Software License Agreement v2.1
// Shamrock is licensed under the CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

/**
 * @file CLBVHDualTreeTraversal.cpp
 * @author Timothée David--Cléris (tim.shamrock@proton.me)
 * @brief
 */

#include "shambase/exception.hpp"
#include "shambase/overloaded.hpp"
#include "shamalgs/ImplVariant.hpp"
#include "shamtree/details/dtt_parallel_select.hpp"
#include "shamtree/details/dtt_reference.hpp"
#include "shamtree/details/dtt_scan_multipass.hpp"

namespace shamtree {

    /// namespace to control implementation behavior
    namespace impl {

        /// Naive reference CPU implementation of the dual tree traversal
        struct Reference {
            static constexpr std::string_view variant_type_name = "reference";
        };

        /// Parallel-select implementation of the dual tree traversal
        struct ParallelSelect {
            static constexpr std::string_view variant_type_name = "parallel_select";
        };

        /// Scan-multipass implementation of the dual tree traversal
        struct ScanMultipass {
            static constexpr std::string_view variant_type_name = "scan_multipass";
        };

        /// Currently selected dual tree traversal implementation
        shamalgs::ImplVariantGlobal<Reference, ParallelSelect, ScanMultipass> dtt_impl;

        /// Get list of available dual tree traversal implementations
        std::vector<std::string> get_default_impl_list_clbvh_dual_tree_traversal() {
            return dtt_impl.get_default_config_list();
        }

        /// Get the current implementation for dual tree traversal
        std::string get_current_impl_clbvh_dual_tree_traversal_impl() {
            return dtt_impl.get_current_config();
        }

        /// Check if an implementation has been selected for dual tree traversal
        bool is_impl_set_clbvh_dual_tree_traversal() { return dtt_impl.is_set(); }

        /// Set the implementation for dual tree traversal
        void set_impl_clbvh_dual_tree_traversal(const std::string &impl) {
            shamlog_info_ln("tree", "setting dtt implementation to impl :", impl);
            dtt_impl.set(impl);
        }

        /// Select the default implementation for dual tree traversal
        void autoselect_impl_clbvh_dual_tree_traversal() {
            dtt_impl.set(ScanMultipass{});
            shamlog_info_ln(
                "tree",
                "defaulting dtt implementation to impl :",
                get_current_impl_clbvh_dual_tree_traversal_impl());
        }

    } // namespace impl

    template<class Tmorton, class Tvec, u32 dim>
    DTTResult clbvh_dual_tree_traversal(
        sham::DeviceScheduler_ptr dev_sched,
        const CompressedLeafBVH<Tmorton, Tvec, dim> &bvh,
        shambase::VecComponent<Tvec> theta_crit,
        bool ordered_result,
        bool allow_leaf_lowering) {

        if (bvh.is_empty()) {
            throw shambase::make_except_with_loc<std::invalid_argument>(
                "BVH is empty, cannot perform DTT");
        }

        using ImplRef = details::DTTCpuReference<Tmorton, Tvec, dim>;
        using ImplPar = details::DTTParallelSelect<Tmorton, Tvec, dim>;
        using ImplSca = details::DTTScanMultipass<Tmorton, Tvec, dim>;

        if (!impl::dtt_impl.is_set()) {
            impl::autoselect_impl_clbvh_dual_tree_traversal();
        }

        bool ord  = ordered_result;
        bool llow = allow_leaf_lowering;

        return std::visit(
            shambase::overloaded{
                [&](impl::Reference) {
                    return ImplRef::dtt(dev_sched, bvh, theta_crit, ord, llow);
                },
                [&](impl::ParallelSelect) {
                    return ImplPar::dtt(dev_sched, bvh, theta_crit, ord, llow);
                },
                [&](impl::ScanMultipass) {
                    return ImplSca::dtt(dev_sched, bvh, theta_crit, ord, llow);
                },
            },
            impl::dtt_impl.get());
    }

    template DTTResult clbvh_dual_tree_traversal<u64, f64_3, 3>(
        sham::DeviceScheduler_ptr dev_sched,
        const CompressedLeafBVH<u64, f64_3, 3> &bvh,
        shambase::VecComponent<f64_3> theta_crit,
        bool ordered_result,
        bool allow_leaf_lowering);

    template DTTResult clbvh_dual_tree_traversal<u32, f64_3, 3>(
        sham::DeviceScheduler_ptr dev_sched,
        const CompressedLeafBVH<u32, f64_3, 3> &bvh,
        shambase::VecComponent<f64_3> theta_crit,
        bool ordered_result,
        bool allow_leaf_lowering);

} // namespace shamtree
