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
 * @file GSPHUtilities.hpp
 * @author Guo Yansong (guo.yansong.ngy@gmail.com)
 * @brief GSPH-specific utilities for ghost handling
 *
 * This file provides GSPH-specific versions of utility functions that use
 * the centralized field name constants from FieldNames.hpp.
 */

#include "shammodels/gsph/modules/GSPHGhostHandler.hpp"
#include "shamrock/scheduler/PatchScheduler.hpp"
#include "shamsys/NodeInstance.hpp"

namespace shammodels::gsph {

    /**
     * @brief GSPH-specific utilities for ghost particle handling
     *
     * This class provides utility functions for GSPH that use the centralized
     * field name constants from FieldNames.hpp.
     *
     * @tparam Tvec Vector type (e.g., f64_3)
     * @tparam SPHKernel Kernel type (e.g., M4)
     */
    template<class Tvec, class SPHKernel>
    class GSPHUtilities {
        public:
        using Tscal = shambase::VecComponent<Tvec>;

        static constexpr Tscal Rkern = SPHKernel::Rkern;

        using GhostHndl = gsph::GSPHGhostHandler<Tvec>;
        using InterfBuildCache
            = shambase::DistributedDataShared<typename GhostHndl::InterfaceIdTable>;

        PatchScheduler &sched;

        GSPHUtilities(PatchScheduler &sched) : sched(sched) {}

        /**
         * @brief Build interface cache for ghost particle communication
         *
         * Uses the field names from FieldNames.hpp.
         *
         * @param interf_handle Ghost handler
         * @param sptree Serial patch tree
         * @param h_evol_max Maximum smoothing length evolution factor
         * @return InterfBuildCache Interface build cache
         */
        inline InterfBuildCache build_interf_cache(
            GhostHndl &interf_handle, SerialPatchTree<Tvec> &sptree, Tscal h_evol_max) {

            using namespace shamrock::patch;

            const u32 ihpart = sched.pdl().template get_field_idx<Tscal>(names::common::hpart);

            PatchField<Tscal> interactR_patch = sched.map_owned_to_patch_field_simple<Tscal>(
                [&](const Patch p, PatchDataLayer &pdat) -> Tscal {
                    if (!pdat.is_empty()) {
                        return pdat.get_field<Tscal>(ihpart).compute_max() * h_evol_max * Rkern;
                    } else {
                        return shambase::VectorProperties<Tscal>::get_min();
                    }
                });

            PatchtreeField<Tscal> interactR_mpi_tree = sptree.make_patch_tree_field(
                sched,
                shamsys::instance::get_compute_queue(),
                interactR_patch,
                [](Tscal h0, Tscal h1, Tscal h2, Tscal h3, Tscal h4, Tscal h5, Tscal h6, Tscal h7) {
                    return sham::max_8points(h0, h1, h2, h3, h4, h5, h6, h7);
                });

            return interf_handle.make_interface_cache(sptree, interactR_mpi_tree, interactR_patch);
        }
    };

} // namespace shammodels::gsph
