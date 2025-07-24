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

#include "shambase/string.hpp"
#include "shammodels/sph/modules/GetParticlesOutsideSphere.hpp"

namespace shammodels::sph::modules {

    template<typename Tvec>
    void GetParticlesOutsideSphere<Tvec>::_impl_evaluate_internal() {
        auto edges = get_edges();

        const shamrock::solvergraph::DDPatchDataFieldRef<Tvec> &pos_refs = edges.pos.get_refs();

        edges.part_ids_outside_sphere.ensure_allocated(pos_refs.get_ids());

        pos_refs.for_each([&](u64 id_patch, const PatchDataField<Tvec> &pos) {
            auto tmp = pos.get_ids_where(
                [](const Tvec *__restrict pos, u32 i, Tvec sphere_center, Tscal sphere_radius) {
                    return sycl::length(pos[i] - sphere_center) > sphere_radius;
                },
                sphere_center,
                sphere_radius);

            edges.part_ids_outside_sphere.buffers.get(id_patch).append(tmp);
        });
    }

    template<typename Tvec>
    std::string GetParticlesOutsideSphere<Tvec>::_impl_get_tex() {
        auto pos                     = get_ro_edge_base(0).get_tex_symbol();
        auto part_ids_outside_sphere = get_rw_edge_base(0).get_tex_symbol();

        std::string tex = R"tex(
        Get particles outside of the sphere

        \begin{align}
        {part_ids_outside_sphere} &= \{i \text{ where } \vert\vert{pos}_i - c\vert\vert < r\}\\
        c &= {center}\\
        r &= {radius}
        \end{align}
        )tex";

        shambase::replace_all(tex, "{pos}", pos);
        shambase::replace_all(tex, "{part_ids_outside_sphere}", part_ids_outside_sphere);
        shambase::replace_all(tex, "{center}", shambase::format("{}", sphere_center));
        shambase::replace_all(tex, "{radius}", shambase::format("{}", sphere_radius));

        return "TODO";
    }

} // namespace shammodels::sph::modules

template class shammodels::sph::modules::GetParticlesOutsideSphere<f64_3>;
