// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright (c) 2021-2026 Timothée David--Cléris <tim.shamrock@proton.me>
// SPDX-License-Identifier: CeCILL Free Software License Agreement v2.1
// Shamrock is licensed under the CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

/**
 * @file  NodeGetNextConsVar.cpp
 * @author Léodasce Sewanou (leodasce.sewanou@ens-lyon.fr)
 * @author Timothée David--Cléris (tim.shamrock@proton.me) --no git blame--
 * @brief
 *
 */

#include "shambase/stacktrace.hpp"
#include "shambackends/kernel_call.hpp"
#include "shammodels/common/amr/NeighGraph.hpp"
#include "shammodels/ramses/SolverConfig.hpp"
#include "shammodels/ramses/modules/NodeGetNextConsVar.hpp"
#include <type_traits>

namespace {
    using Direction = shammodels::basegodunov::modules::Direction;

    template<class Tvec>
    struct KernelNextConsVar {

        using Tscal = sham::VecComponent<Tvec>;

        inline static void kernel(
            const shambase::DistributedData<u32> &sizes,
            const shambase::DistributedData<shamrock::PatchDataFieldSpanPointer<Tscal>>
                &spans_rho_old,
            const shambase::DistributedData<shamrock::PatchDataFieldSpanPointer<Tscal>>
                &spans_rho_next,
            const shambase::DistributedData<shamrock::PatchDataFieldSpanPointer<Tvec>>
                &spans_rhov_old,
            const shambase::DistributedData<shamrock::PatchDataFieldSpanPointer<Tscal>>
                &spans_rhoe_old,
            const shambase::DistributedData<shamrock::PatchDataFieldSpanPointer<Tvec>>
                &spans_phi_g_old,
            const shambase::DistributedData<shamrock::PatchDataFieldSpanPointer<Tvec>>
                &spans_phi_g_next,
            shambase::DistributedData<shamrock::PatchDataFieldSpanPointer<Tvec>> &spans_rhov_next,
            shambase::DistributedData<shamrock::PatchDataFieldSpanPointer<Tscal>> &spans_rhoe_next,
            const f64 dt_over_2,
            u32 block_size)

        {

            shambase::DistributedData<u32> cell_counts
                = sizes.map<u32>([&](u64 id, u32 block_count) {
                      u32 cell_count = block_count * block_size;
                      return cell_count;
                  });

            sham::distributed_data_kernel_call(
                shamsys::instance::get_compute_scheduler_ptr(),
                sham::DDMultiRef{
                    spans_rho_old,
                    spans_rho_next,
                    spans_rhov_old,
                    spans_rhoe_old,
                    spans_phi_g_old,
                    spans_phi_g_next},
                sham::DDMultiRef{spans_rhov_next, spans_rhoe_next},
                cell_counts,
                [dt_over_2](
                    u32 i,
                    const Tscal *__restrict rho_old,
                    const Tscal *__restrict rho_next,
                    const Tvec *__restrict rhov_old,
                    const Tscal *__restrict rhoe_old,
                    const Tvec *__restrict phi_g_old,
                    const Tvec *__restrict phi_g_next,
                    const Tvec *__restrict dt_rhov_old,
                    const Tscal *__restrict dt_rhoe_old,
                    Tvec *__restrict rhov_new,
                    Tscal *__restrict rhoe_new) {
                    Tvec tmp_rhov
                        = rhov_old[i] + (2. * dt_over_2) * dt_rhov_old[i]
                          + dt_over_2 * (rho_old[i] * phi_g_old[i] + rho_next[i] * phi_g_next[i]);
                    rhov_new[i] = tmp_rhov;

                    rhoe_new[i]
                        = rhoe_old[i] + (2. * dt_over_2) * dt_rhoe_old[i]
                          + dt_over_2
                                * (rho_old[i] * sham::dot((rhov_old[i] / rho_old[i]), phi_g_old[i])
                                   + rho_next[i]
                                         * sham::dot((tmp_rhov / rho_next[i]), phi_g_next[i]));
                });
        }
    };

} // namespace

namespace shammodels::basegodunov::modules {
    template<class Tvec>
    void NodeGetNextConsVar<Tvec>::_impl_evaluate_internal() {
        StackEntry stack_loc{};
        auto edges = get_edges();

        {

            edges.spans_rho_old.check_sizes(edges.sizes.indexes);
            edges.spans_rho_next.check_sizes(edges.sizes.indexes);
            edges.spans_rhov_old.check_sizes(edges.sizes.indexes);
            edges.spans_rhoe_old.check_sizes(edges.sizes.indexes);

            edges.spans_phi_g_old.check_sizes(edges.sizes.indexes);
            edges.spans_phi_g_next.check_sizes(edges.sizes.indexes);

            edges.spans_dt_rhoe_old.check_sizes(edges.sizes.indexes);
            edges.spans_dt_rhov_old.check_sizes(edges.sizes.indexes);

            edges.spans_rhov_next.ensure_sizes(edges.sizes.indexes);
            edges.spans_rhoe_next.ensure_sizes(edges.sizes.indexes);

            KernelNextConsVar<Tvec>::kernel(
                edges.sizes.indexes,
                edges.spans_rho_old.get_spans(),
                edges.spans_rho_next.get_spans(),
                edges.spans_rhov_old.get_spans(),
                edges.spans_rhoe_old.get_spans(),
                edges.spans_phi_g_old.get_spans(),
                edges.spans_phi_g_next.get_spans(),
                edges.spans_rhov_next.get_spans(),
                edges.spans_rhoe_next.get_spans(),
                edges.dt_over2.value,
                block_size);
        }
    }
} // namespace shammodels::basegodunov::modules

template class shammodels::basegodunov::modules::NodeGetNextConsVar<f64_3>;
