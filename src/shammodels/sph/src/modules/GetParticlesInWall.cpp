// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright (c) 2021-2026 Timothée David--Cléris <tim.shamrock@proton.me>
// SPDX-License-Identifier: CeCILL Free Software License Agreement v2.1
// Shamrock is licensed under the CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

/**
 * @file GetParticlesInWall.cpp
 * @author Yona Lapeyre (yona.lapeyre@ens-lyon.fr)
 * @brief Implements the GetParticlesInWall module, which identifies particles in a wall.
 *
 */

#include "shambase/stacktrace.hpp"
#include "shambase/string.hpp"
#include "shammodels/sph/modules/GetParticlesInWall.hpp"

namespace shammodels::sph::modules {

    template<typename Tvec>
    void GetParticlesInWall<Tvec>::_impl_evaluate_internal() {
        StackEntry stack_loc{};
        auto edges = get_edges();

        const shamrock::solvergraph::DDPatchDataFieldRef<Tvec> &pos_refs = edges.pos.get_refs();

        edges.part_ids_in_wall.ensure_allocated(pos_refs.get_ids());

        pos_refs.for_each([&](u64 id_patch, const PatchDataField<Tvec> &pos) {
            auto tmp = pos.get_ids_where(
                [](const Tvec *__restrict pos, u32 i, Tvec wall_pos, Tscal wall_thickness) {
                    return sycl::length(pos[i] - wall_pos) > wall_thickness;
                },
                wall_pos,
                wall_thickness);

            edges.part_ids_in_wall.buffers.get(id_patch).append(tmp);
        });
    }

    template<typename Tvec>
    std::string GetParticlesInWall<Tvec>::_impl_get_tex() const {
        auto pos              = get_ro_edge_base(0).get_tex_symbol();
        auto part_ids_in_wall = get_rw_edge_base(0).get_tex_symbol();

        std::string tex = R"tex(
        Get particles outside of the sphere

        \begin{align}
        {part_ids_in_wall} &= \{i \text{ where } \vert\vert{pos}_i - c\vert\vert > r\}\\
        c &= {center}\\
        r &= {radius}
        \end{align}
        )tex";

        shambase::replace_all(tex, "{pos}", pos);
        shambase::replace_all(tex, "{part_ids_in_wall}", part_ids_in_wall);
        shambase::replace_all(tex, "{center}", shambase::format("{}", wall_pos));
        shambase::replace_all(tex, "{radius}", shambase::format("{}", wall_thickness));

        return tex;
    }

} // namespace shammodels::sph::modules

template class shammodels::sph::modules::GetParticlesInWall<f64_3>;
