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
#include "shambackends/kernel_call_distrib.hpp"
#include "shammodels/sph/modules/GetParticlesInWall.hpp"

namespace shammodels::sph::modules {

    template<typename Tvec>
    void GetParticlesInWall<Tvec>::_impl_evaluate_internal() {
        StackEntry stack_loc{};

        auto edges = get_edges();

        auto &thread_counts = edges.sizes.indexes;

        edges.pos.check_sizes(thread_counts);
        edges.part_ids_in_wall.ensure_sizes(thread_counts);

        auto &positions        = edges.pos.get_spans();
        auto &part_ids_in_wall = edges.part_ids_in_wall.get_spans();

        auto dev_sched = shamsys::instance::get_compute_scheduler_ptr();

        sham::distributed_data_kernel_call(
            dev_sched,
            sham::DDMultiRef{positions},
            sham::DDMultiRef{part_ids_in_wall},
            thread_counts,
            [wall_pos       = this->wall_pos,
             wall_length    = this->wall_length,
             wall_width     = this->wall_width,
             wall_thickness = this->wall_thickness](
                u32 i, const Tvec *__restrict pos, u32 *__restrict part_ids_in_wall) {
                Tscal x = pos[i][0];
                Tscal y = pos[i][1];
                Tscal z = pos[i][2];

                Tscal x0 = wall_pos[0];
                Tscal y0 = wall_pos[1];
                Tscal z0 = wall_pos[2];

                bool in_wall = (x - x0 < wall_length) && (x - x0 > 0) && (y - y0 < wall_width)
                               && (y - y0 > 0) && (z - z0 < wall_thickness) && (z - z0 > 0);

                if (in_wall) {
                    part_ids_in_wall[i] = 1;
                } else {
                    part_ids_in_wall[i] = 0;
                }
            });
    }

    /**
     * @brief Returns the tex string for the GetParticlesInWall module.
     *
     * This module identifies particles inside a rectangular wall.
     *
     * @param[in] pos The position field.
     * @param[in] part_ids_in_wall The particle id field.
     * @return The tex string.
     */
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
