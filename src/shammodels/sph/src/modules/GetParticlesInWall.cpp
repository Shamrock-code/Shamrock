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
#include "shamcomm/logs.hpp"
#include "shammodels/sph/modules/GetParticlesInWall.hpp"

namespace shammodels::sph::modules {

    template<typename Tvec>
    void GetParticlesInWall<Tvec>::_impl_evaluate_internal() {
        StackEntry stack_loc{};
        auto edges = get_edges();

        auto &thread_counts = edges.sizes.indexes;
        edges.pos.check_sizes(thread_counts);
        edges.ghost_mask.ensure_sizes(thread_counts);

        auto &positions  = edges.pos.get_spans();
        auto &ghost_mask = edges.ghost_mask.get_spans();

        auto dev_sched = shamsys::instance::get_compute_scheduler_ptr();
        sham::distributed_data_kernel_call(
            dev_sched,
            sham::DDMultiRef{positions},
            sham::DDMultiRef{ghost_mask},
            thread_counts,
            [wall_func
             = this->wall_func](u32 i, const Tvec *__restrict pos, u32 *__restrict ghost_mask) {
                bool in_wall = wall_func(pos[i]);

                if (in_wall) {
                    ghost_mask[i] = 1;
                } else {
                    ghost_mask[i] = 0;
                }
            });
    }

    /**
     * @brief Returns the tex string for the GetParticlesInWall module.
     *
     * This module identifies particles inside a rectangular wall.
     *
     * @param[in] pos The position field.
     * @param[in] ghost_mask The particle id field.
     * @return The tex string.
     */
    template<typename Tvec>
    std::string GetParticlesInWall<Tvec>::_impl_get_tex() const {

        auto positions  = get_ro_edge_base(0).get_tex_symbol();
        auto ghost_mask = get_rw_edge_base(0).get_tex_symbol();

        std::string tex = R"tex(
    Identify particles inside the wall.

    For each particle \(i\), evaluate the user provided predicate \(\texttt{wall\_func}\) on its position.
    \begin{align}
    {ghost_mask}_i &=
    \begin{cases}
    1 & \text{if } \texttt{wall\_func}({positions}_i) = \text{true} \\
    0 & \text{otherwise}
    \end{cases}
    \end{align}
    )tex";

        shambase::replace_all(tex, "{positions}", positions);
        shambase::replace_all(tex, "{ghost_mask}", ghost_mask);

        return tex;
    }

} // namespace shammodels::sph::modules

template class shammodels::sph::modules::GetParticlesInWall<f64_3>;
