// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright (c) 2021-2024 Timothée David--Cléris <tim.shamrock@proton.me>
// SPDX-License-Identifier: CeCILL Free Software License Agreement v2.1
// Shamrock is licensed under the CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

/**
 * @file KillParticles.cpp
 * @author Timothée David--Cléris (timothee.david--cleris@ens-lyon.fr)
 * @brief
 *
 */

#include "shammodels/sph/modules/KillParticles.hpp"

namespace shammodels::sph::modules {

    void KillParticles::_impl_evaluate_internal() {
        auto edges = get_edges();

        std::vector<u64> ids;
        edges.patchdatas.patchdatas.for_each(
            [&ids](u64 id_patch, shamrock::patch::PatchData &patchdata) {
                ids.push_back(id_patch);
            });

        edges.part_to_remove.check_allocated(ids);

        edges.patchdatas.patchdatas.for_each(
            [&](u64 id_patch, shamrock::patch::PatchData &patchdata) {
                auto &buf = edges.part_to_remove.buffers.get(id_patch);
                patchdata.remove_ids(buf, buf.get_size());
            });
    }

} // namespace shammodels::sph::modules
