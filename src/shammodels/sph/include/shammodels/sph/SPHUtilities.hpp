// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright (c) 2021-2025 Timothée David--Cléris <tim.shamrock@proton.me>
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
        using InterfBuildCache
            = shambase::DistributedDataShared<typename GhostHndl::InterfaceIdTable>;

        PatchScheduler &sched;

        SPHUtilities(PatchScheduler &sched) : sched(sched) {}

        inline InterfBuildCache
        build_interf_cache(GhostHndl &interf_handle, SerialPatchTree<vec> &sptree, flt h_evol_max) {

            using namespace shamrock::patch;

            const u32 ihpart = sched.pdl().template get_field_idx<flt>("hpart");

            PatchField<flt> interactR_patch = sched.map_owned_to_patch_field_simple<flt>(
                [&](const Patch p, PatchDataLayer &pdat) -> flt {
                    if (!pdat.is_empty()) {
                        return pdat.get_field<flt>(ihpart).compute_max() * h_evol_max * Rkern;
                    } else {
                        return shambase::VectorProperties<flt>::get_min();
                    }
                });

            PatchtreeField<flt> interactR_mpi_tree = sptree.make_patch_tree_field(
                sched,
                shamsys::instance::get_compute_queue(),
                interactR_patch,
                [](flt h0, flt h1, flt h2, flt h3, flt h4, flt h5, flt h6, flt h7) {
                    return sham::max_8points(h0, h1, h2, h3, h4, h5, h6, h7);
                });

            return interf_handle.make_interface_cache(sptree, interactR_mpi_tree, interactR_patch);
        }

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
