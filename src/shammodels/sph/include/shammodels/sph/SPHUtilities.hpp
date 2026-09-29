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
 * @file SPHUtilities.hpp
 * @author Timothée David--Cléris (tim.shamrock@proton.me)
 * @brief
 *
 */

#include "shammodels/sph/BasicSPHGhosts.hpp"
#include "shammodels/sph/math/density.hpp"
#include "shamrock/scheduler/PatchScheduler.hpp"
#include "shamtree/RadixTree.hpp"
#include "shamtree/TreeTraversal.hpp"

namespace shammodels::sph {

    template<class vec, class SPHKernel, class u_morton>
    class SPHTreeUtilities {

        public:
        using flt = shambase::VecComponent<vec>;

        static constexpr flt Rkern = SPHKernel::Rkern;

        using GhostHndl = BasicSPHGhostHandler<vec>;
        using InterfBuildCache
            = shambase::DistributedDataShared<typename GhostHndl::InterfaceIdTable>;

        PatchScheduler &sched;

        SPHTreeUtilities(PatchScheduler &sched) : sched(sched) {}

        static void iterate_smoothing_length_tree(

            sycl::buffer<vec> &merged_r,
            sycl::buffer<flt> &hnew,
            sycl::buffer<flt> &hold,
            sycl::buffer<flt> &eps_h,
            sycl::range<1> update_range,
            RadixTree<u_morton, vec> &tree,

            flt gpart_mass,
            flt h_evol_max,
            flt h_evol_iter_max

        );
    };

    /**
     * @brief handle basic utilities dealing with SPH
     *
     * @tparam vec
     */
    template<class vec, class SPHKernel>
    class SPHUtilities {
        public:
        using flt = shambase::VecComponent<vec>;

        static constexpr flt Rkern = SPHKernel::Rkern;

        using GhostHndl = BasicSPHGhostHandler<vec>;

        PatchScheduler &sched;

        SPHUtilities(PatchScheduler &sched) : sched(sched) {}

        static void iterate_smoothing_length_cache(

            sham::DeviceBuffer<vec> &merged_r,
            sham::DeviceBuffer<flt> &hnew,
            sham::DeviceBuffer<flt> &hold,
            sham::DeviceBuffer<flt> &eps_h,
            sycl::range<1> update_range,
            shamrock::tree::ObjectCache &neigh_cache,

            flt gpart_mass,
            flt h_evol_max,
            flt h_evol_iter_max

        );

        template<class u_morton>
        static void iterate_smoothing_length_tree(

            sycl::buffer<vec> &merged_r,
            sycl::buffer<flt> &hnew,
            sycl::buffer<flt> &hold,
            sycl::buffer<flt> &eps_h,
            sycl::range<1> update_range,
            RadixTree<u_morton, vec> &tree,

            flt gpart_mass,
            flt h_evol_max,
            flt h_evol_iter_max

        ) {
            SPHTreeUtilities<vec, SPHKernel, u_morton>::iterate_smoothing_length_tree(
                merged_r,
                hnew,
                hold,
                eps_h,
                update_range,
                tree,
                gpart_mass,
                h_evol_max,
                h_evol_iter_max);
        }

        static void compute_omega(
            sham::DeviceBuffer<vec> &merged_r,
            sham::DeviceBuffer<flt> &h_part,
            sham::DeviceBuffer<flt> &omega_h,
            sycl::range<1> part_range,
            shamrock::tree::ObjectCache &neigh_cache,
            flt gpart_mass);
    };

} // namespace shammodels::sph
