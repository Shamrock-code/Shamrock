// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright (c) 2021-2025 Timothée David--Cléris <tim.shamrock@proton.me>
// SPDX-License-Identifier: CeCILL Free Software License Agreement v2.1
// Shamrock is licensed under the CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

/**
 * @file NodeUpdateMomentum.cpp
 * @author Léodasce Sewanou (leodasce.sewanou@ens-lyon.fr)
 * @brief
 *
 */

#include "shammodels/ramses/modules/NodeUpdateMomentum.hpp"
#include "shambackends/kernel_call_distrib.hpp"
#include "shambackends/typeAliasVec.hpp"
#include "shamsys/NodeInstance.hpp"

namespace {

    template<class Tvec>
    struct KernelUpdateMomentum {
        using Tscal = shambase::VecComponent<Tvec>;

        inline static void kernel(
            const shambase::DistributedData<shamrock::PatchDataFieldSpanPointer<Tscal>> &spans_rho,
            const shambase::DistributedData<shamrock::PatchDataFieldSpanPointer<Tvec>> &spans_g,
            shambase::DistributedData<shamrock::PatchDataFieldSpanPointer<Tvec>> &spans_rhovel,
            shambase::DistributedData<shamrock::PatchDataFieldSpanPointer<Tscal>> &spans_rhoe,

            const Tscal dt,
            const shambase::DistributedData<u32> &sizes,
            const shambase::DistributedData<u32> &sizes_no_gz,
            u32 block_size) {

            shambase::DistributedData<u32> cell_counts
                = sizes.map<u32>([&](u64 id, u32 block_count) {
                      u32 cell_count = block_count * block_size;
                      return cell_count;
                  });

            shambase::DistributedData<u32> cell_counts_no_ghost
                = sizes_no_gz.map<u32>([&](u64 id, u32 block_count) {
                      u32 cell_count = block_count * block_size;
                      return cell_count;
                  });

            sham::distributed_data_kernel_call(
                shamsys::instance::get_compute_scheduler_ptr(),
                sham::DDMultiRef{spans_rho, spans_g},
                sham::DDMultiRef{spans_rhovel, spans_rhoe},
                // cell_counts,
                cell_counts_no_ghost,
                [dt](
                    u32 i,
                    const Tscal *__restrict rho,
                    const Tvec *__restrict g,
                    Tvec *__restrict rhovel,
                    Tscal *__restrict rhoe) {
                    auto tmp  = rhovel[i];
                    rhovel[i] = tmp + dt * rho[i] * g[i];
                    // rhoe[i] += dt * sham::dot(tmp, g[i]);
                });
        }
    };

} // namespace

namespace shammodels::basegodunov::modules {
    template<class Tvec>
    void NodeUpdateMomentum<Tvec>::_impl_evaluate_internal() {
        auto edges = get_edges();

        if (edges.dt.value != 0)

        {
            edges.spans_rho.check_sizes(edges.sizes.indexes);
            edges.spans_g.check_sizes(edges.sizes.indexes);
            edges.spans_rhovel.ensure_sizes(edges.sizes.indexes);

            KernelUpdateMomentum<Tvec>::kernel(
                edges.spans_rho.get_spans(),
                edges.spans_g.get_spans(),
                edges.spans_rhovel.get_spans(),
                edges.spans_rhoe.get_spans(),
                edges.dt.value,
                edges.sizes.indexes,
                edges.sizes_no_gz.indexes,
                block_size);
        }

        else {
            return;
        }
    }

} // namespace shammodels::basegodunov::modules

template class shammodels::basegodunov::modules::NodeUpdateMomentum<f64_3>;
