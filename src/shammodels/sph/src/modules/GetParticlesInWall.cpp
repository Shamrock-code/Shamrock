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

        // edges.part_ids_in_wall.ensure_allocated(pos_refs.get_ids());

        pos_refs.for_each([&](u64 id_patch, const PatchDataField<Tvec> &pos) {
            auto tmp = pos.get_ids_where(
                [](const Tvec *__restrict pos,
                   u32 i,
                   Tvec wall_pos,
                   Tscal wall_length,
                   Tscal wall_width,
                   Tscal wall_thickness) {
                    Tscal x = pos[i][0];
                    Tscal y = pos[i][1];
                    Tscal z = pos[i][2];

                    Tscal x0 = wall_pos[0];
                    Tscal y0 = wall_pos[1];
                    Tscal z0 = wall_pos[2];

                    bool in_wall = (x - x0 < wall_length) && (x - x0 > 0) && (y - y0 < wall_width)
                                   && (y - y0 > 0) && (z - z0 < wall_thickness) && (z - z0 > 0);

                    return in_wall;
                },
                wall_pos,
                wall_length,
                wall_width,
                wall_thickness);

            // edges.part_ids_in_wall.buffers.get(id_patch).append(tmp);
        });
    }

    template<typename Tvec>
    std::string GetParticlesInWall<Tvec>::_impl_get_tex() const {
        auto pos              = get_ro_edge_base(0).get_tex_symbol();
        auto part_ids_in_wall = get_rw_edge_base(0).get_tex_symbol();

        std::string tex = R"tex(
    Get particles inside the rectangular wall

    \begin{align}
    {part_ids_in_wall} &= \{i \text{ where } \vert\vert{x}_i - {x0}\vert\vert < {wall_length} \text{ and } \vert\vert{y}_i - {y0}\vert\vert < {wall_width} \text{ and } \vert\vert{z}_i - {z0}\vert\vert < {wall_thickness}\}\\
    \end{align}
    )tex";

        shambase::replace_all(tex, "{x0}", std::to_string(wall_pos[0]));
        shambase::replace_all(tex, "{y0}", std::to_string(wall_pos[1]));
        shambase::replace_all(tex, "{z0}", std::to_string(wall_pos[2]));
        shambase::replace_all(tex, "{wall_length}", std::to_string(wall_length));
        shambase::replace_all(tex, "{wall_width}", std::to_string(wall_width));
        shambase::replace_all(tex, "{wall_thickness}", std::to_string(wall_thickness));

        return tex;
    }

} // namespace shammodels::sph::modules

template class shammodels::sph::modules::GetParticlesInWall<f64_3>;
