// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright (c) 2021-2025 Timothée David--Cléris <tim.shamrock@proton.me>
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
#include "shamtree/details/dtt_parallel_select.hpp"
#include "shamtree/details/dtt_reference.hpp"
#include "shamtree/details/dtt_scan_multipass.hpp"

namespace shamtree {

    enum class DTTImpl : u32 { REFERENCE, PARALLEL_SELECT, SCAN_MULTIPASS };
    DTTImpl dtt_impl = DTTImpl::SCAN_MULTIPASS;

    inline DTTImpl dtt_impl_from_params(const std::string &impl) {
        if (impl == "reference") {
            return DTTImpl::REFERENCE;
        } else if (impl == "parallel_select") {
            return DTTImpl::PARALLEL_SELECT;
        } else if (impl == "scan_multipass") {
            return DTTImpl::SCAN_MULTIPASS;
        }
        throw shambase::make_except_with_loc<std::invalid_argument>(shambase::format(
            "invalid implementation : {}, possible implementations : {}",
            impl,
            impl::get_default_impl_list_clbvh_dual_tree_traversal()));
    }

    inline shamalgs::impl_param dtt_impl_to_params(const DTTImpl &impl) {
        if (impl == DTTImpl::REFERENCE) {
            return {"reference", ""};
        } else if (impl == DTTImpl::PARALLEL_SELECT) {
            return {"parallel_select", ""};
        } else if (impl == DTTImpl::SCAN_MULTIPASS) {
            return {"scan_multipass", ""};
        }
        throw shambase::make_except_with_loc<std::invalid_argument>(
            shambase::format("unknow dtt implementation : {}", u32(impl)));
    }

    std::vector<shamalgs::impl_param> impl::get_default_impl_list_clbvh_dual_tree_traversal() {
        std::vector<shamalgs::impl_param> impl_list{
            {"reference", ""}, {"parallel_select", ""}, {"scan_multipass", ""}};
        return impl_list;
    }

    void impl::set_impl_clbvh_dual_tree_traversal(
        const std::string &impl, const std::string &param) {
        shamlog_info_ln("tree", "setting dtt implementation to impl :", impl);
        dtt_impl = dtt_impl_from_params(impl);
    }

    shamalgs::impl_param impl::get_current_impl_clbvh_dual_tree_traversal_impl() {
        return dtt_impl_to_params(dtt_impl);
    }

    template<class Tmorton, class Tvec, u32 dim>
    DTTResult clbvh_dual_tree_traversal(
        sham::DeviceScheduler_ptr dev_sched,
        const CompressedLeafBVH<Tmorton, Tvec, dim> &bvh,
        shambase::VecComponent<Tvec> theta_crit,
        bool ordered_result) {

        if (bvh.is_empty()) {
            throw shambase::make_except_with_loc<std::invalid_argument>(
                "BVH is empty, cannot perform DTT");
        }

        using ImplRef = details::DTTCpuReference<Tmorton, Tvec, dim>;
        using ImplPar = details::DTTParallelSelect<Tmorton, Tvec, dim>;
        using ImplSca = details::DTTScanMultipass<Tmorton, Tvec, dim>;

        bool ord = ordered_result;

        switch (dtt_impl) {
        case DTTImpl::REFERENCE      : return ImplRef::dtt(dev_sched, bvh, theta_crit, ord);
        case DTTImpl::PARALLEL_SELECT: return ImplPar::dtt(dev_sched, bvh, theta_crit, ord);
        case DTTImpl::SCAN_MULTIPASS : return ImplSca::dtt(dev_sched, bvh, theta_crit, ord);
        default                      : shambase::throw_unimplemented();
        }
    }

    template DTTResult clbvh_dual_tree_traversal<u64, f64_3, 3>(
        sham::DeviceScheduler_ptr dev_sched,
        const CompressedLeafBVH<u64, f64_3, 3> &bvh,
        shambase::VecComponent<f64_3> theta_crit,
        bool ordered_result);

} // namespace shamtree
