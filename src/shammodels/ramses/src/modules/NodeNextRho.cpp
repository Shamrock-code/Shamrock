// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright (c) 2021-2026 Timothée David--Cléris <tim.shamrock@proton.me>
// SPDX-License-Identifier: CeCILL Free Software License Agreement v2.1
// Shamrock is licensed under the CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

/**
 * @file NodeNextRho.cpp
 * @author Léodasce Sewanou (leodasce.sewanou@ens-lyon.fr)
 * @author Timothée David--Cléris (tim.shamrock@proton.me) --no git blame--
 * @brief
 *
 */

#include "shambase/stacktrace.hpp"
#include "shambackends/kernel_call.hpp"
#include "shamcomm/logs.hpp"
#include "shammodels/common/amr/NeighGraph.hpp"
#include "shammodels/ramses/SolverConfig.hpp"
#include "shammodels/ramses/modules/NodeNextRho.hpp"
#include <type_traits>

namespace {

    using Direction = shammodels::basegodunov::modules::Direction;

    template<class Tvec>
    struct KernelNextRho {

        using Tscal = sham::VecComponent<Tvec>;

        inline static void kernel(
            const shambase::DistributedData<u32> &sizes,
            const shambase::DistributedData<shamrock::PatchDataFieldSpanPointer<Tscal>>
                &spans_rho_old,
            const shambase::DistributedData<shamrock::PatchDataFieldSpanPointer<Tscal>>
                &spans_dt_rho_old,
            const f64 dt_over_2,
            shambase::DistributedData<shamrock::PatchDataFieldSpanPointer<Tscal>> &spans_rho_new,
            u32 block_size) {

            shambase::DistributedData<u32> cell_counts
                = sizes.map<u32>([&](u64 id, u32 block_count) {
                      u32 cell_count = block_count * block_size;
                      return cell_count;
                  });
            sham::distributed_data_kernel_call(
                shamsys::instance::get_compute_scheduler_ptr(),
                sham::DDMultiRef{spans_rho_old, spans_dt_rho_old},
                sham::DDMultiRef{spans_rho_new},
                cell_counts,
                [dt_over_2](
                    u32 i,
                    const Tscal *__restrict rho_old,
                    const Tscal *__restrict dt_rho_old,
                    Tscal *__restrict rho_new) {
                    rho_new[i] = rho_old[i] + (2. * dt_over_2) * dt_rho_old[i];
                });
        }
    };

} // namespace

namespace shammodels::basegodunov::modules {

    template<class Tvec>
    void NodeNextRho<Tvec>::_impl_evaluate_internal() {
        StackEntry stack_loc{};
        auto edges = get_edges();
        edges.spans_dt_rho_old.check_sizes(edges.sizes.indexes);
        edges.spans_rho_next.ensure_sizes(edges.sizes.indexes);
        edges.spans_rho_old.check_sizes(edges.sizes.indexes);

        KernelNextRho<Tvec>::kernel(
            edges.sizes.indexes,
            edges.spans_rho_old.get_spans(),
            edges.spans_dt_rho_old.get_spans(),
            edges.dt_over2.value,
            edges.spans_rho_next.get_spans(),
            block_size);
    }

} // namespace shammodels::basegodunov::modules

template class shammodels::basegodunov::modules::NodeNextRho<f64_3>;
