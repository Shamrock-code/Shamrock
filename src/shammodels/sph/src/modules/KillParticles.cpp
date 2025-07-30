// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright (c) 2021-2025 Timothée David--Cléris <tim.shamrock@proton.me>
// SPDX-License-Identifier: CeCILL Free Software License Agreement v2.1
// Shamrock is licensed under the CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

/**
 * @file KillParticles.cpp
 * @author Timothée David--Cléris (tim.shamrock@proton.me)
 * @brief Implements the KillParticles module, which removes particles based on provided indices.
 *
 */

#include "shammodels/sph/modules/KillParticles.hpp"

namespace shammodels::sph::modules {

    void KillParticles::_impl_evaluate_internal() {
        StackEntry stack_loc{};
        auto edges = get_edges();

        edges.part_to_remove.check_allocated(edges.patchdatas.patchdatas.get_ids());

        edges.patchdatas.patchdatas.for_each(
            [&](u64 id_patch, shamrock::patch::PatchData &patchdata) {
                auto &buf = edges.part_to_remove.buffers.get(id_patch);
                u32 bsize = buf.get_size();
                if (bsize > 0) {
                    patchdata.remove_ids(buf, bsize);
                }
            });
    }

    std::string KillParticles::_impl_get_tex() {

        auto part_to_remove = get_ro_edge_base(0).get_tex_symbol();
        auto patchdatas     = get_rw_edge_base(0).get_tex_symbol();

        std::string tex = R"tex(
            Particle killing:

            Remove particles ${part_to_remove}$ from ${patchdatas}$
        )tex";

        shambase::replace_all(tex, "{part_to_remove}", part_to_remove);
        shambase::replace_all(tex, "{patchdatas}", patchdatas);

        return tex;
    }

} // namespace shammodels::sph::modules
