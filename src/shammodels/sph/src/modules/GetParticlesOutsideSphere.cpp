// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright (c) 2021-2024 Timothée David--Cléris <tim.shamrock@proton.me>
// SPDX-License-Identifier: CeCILL Free Software License Agreement v2.1
// Shamrock is licensed under the CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

/**
 * @file GetParticlesOutsideSphere.cpp
 * @author Timothée David--Cléris (tim.shamrock@proton.me)
 * @brief Implements the GetParticlesOutsideSphere module, which identifies particles outside a
 * given sphere.
 *
 */

#include "shammodels/sph/modules/GetParticlesOutsideSphere.hpp"

namespace shammodels::sph::modules {

    template<typename Tvec>
    void GetParticlesOutsideSphere<Tvec>::_impl_evaluate_internal() {
        auto edges = get_edges();

        const shamrock::solvergraph::DDPatchDataFieldRef<Tvec> &pos_refs = edges.pos.get_refs();

        edges.part_ids_outside_sphere.ensure_allocated(pos_refs.get_ids());

        pos_refs.for_each([&](u64 id_patch, const PatchDataField<Tvec> &pos) {
            auto tmp = pos.get_ids_where(
                [](Tvec pos, Tvec sphere_center, Tscal sphere_radius) {
                    return sycl::length(pos - sphere_center) > sphere_radius;
                },
                sphere_center,
                sphere_radius);

            edges.part_ids_outside_sphere.buffers.get(id_patch).append(tmp);
        });
    }

} // namespace shammodels::sph::modules

template class shammodels::sph::modules::GetParticlesOutsideSphere<f64_3>;
