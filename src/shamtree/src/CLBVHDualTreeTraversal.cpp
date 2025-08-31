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
#include "shamtree/details/dtt_parralel_select.hpp"
#include "shamtree/details/dtt_reference.hpp"
#include "shamtree/details/dtt_scan_multipass.hpp"

namespace shamtree {

    enum class DTTImpl { REFERENCE, PARALLEL_SELECT, SCAN_MULTIPASS };
    DTTImpl dtt_impl = DTTImpl::SCAN_MULTIPASS;

    std::vector<std::string> impl::get_impl_list_clbvh_dual_tree_traversal() {
        return {"reference", "parallel_select", "scan_multipass"};
    }

    void impl::set_impl_clbvh_dual_tree_traversal(
        const std::string &impl, const std::string &param) {
        if (impl == "reference") {
            dtt_impl = DTTImpl::REFERENCE;
        } else if (impl == "parallel_select") {
            dtt_impl = DTTImpl::PARALLEL_SELECT;
        } else if (impl == "scan_multipass") {
            dtt_impl = DTTImpl::SCAN_MULTIPASS;
        } else {
            throw std::invalid_argument("invalid implementation");
        }
    }

    template<class Tmorton, class Tvec, u32 dim>
    inline DTTResult clbvh_dual_tree_traversal(
        sham::DeviceScheduler_ptr dev_sched,
        const CompressedLeafBVH<Tmorton, Tvec, dim> &bvh,
        shambase::VecComponent<Tvec> theta_crit) {

        using ImplRef = details::DTTCpuReference<Tmorton, Tvec, dim>;
        using ImplPar = details::DTTParallelSelect<Tmorton, Tvec, dim>;
        using ImplSca = details::DTTScanMultipass<Tmorton, Tvec, dim>;

        switch (dtt_impl) {
        case DTTImpl::REFERENCE      : return ImplRef::dtt(dev_sched, bvh, theta_crit);
        case DTTImpl::PARALLEL_SELECT: return ImplPar::dtt(dev_sched, bvh, theta_crit);
        case DTTImpl::SCAN_MULTIPASS : return ImplSca::dtt(dev_sched, bvh, theta_crit);
        default                      : shambase::throw_unimplemented();
        }
    }

    template DTTResult clbvh_dual_tree_traversal<u64, f64_3, 3>(
        sham::DeviceScheduler_ptr dev_sched,
        const CompressedLeafBVH<u64, f64_3, 3> &bvh,
        shambase::VecComponent<f64_3> theta_crit);

} // namespace shamtree
