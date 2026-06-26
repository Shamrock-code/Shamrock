// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright (c) 2021-2026 Timothée David--Cléris <tim.shamrock@proton.me>
// SPDX-License-Identifier: CeCILL Free Software License Agreement v2.1
// Shamrock is licensed under the CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

/**
 * @file NodeAYPX.cpp
 * @author Léodasce Sewanou (leodasce.sewanou@ens-lyon.fr)
 * @brief
 *
 */
#include "shammodels/ramses/modules/NodeAYPX.hpp"
#include "shambackends/kernel_call_distrib.hpp"
#include "shambackends/typeAliasVec.hpp"
#include "shambackends/vec.hpp"
#include "shammodels/ramses/SolverConfig.hpp"
#include "shamsys/NodeInstance.hpp"

namespace {

    template<class T>
    struct KernelAYPX {
        inline static void kernel(
            const shambase::DistributedData<shamrock::PatchDataFieldSpanPointer<T>> &spans_x,
            shambase::DistributedData<shamrock::PatchDataFieldSpanPointer<T>> &spans_y,
            const f64 alpha,
            const shambase::DistributedData<u32> &sizes,
            u32 block_size) {

            shambase::DistributedData<u32> cell_counts
                = sizes.map<u32>([&](u64 id, u32 block_count) {
                      u32 cell_count = block_count * block_size;
                      return cell_count;
                  });

            sham::distributed_data_kernel_call(
                shamsys::instance::get_compute_scheduler_ptr(),
                sham::DDMultiRef{spans_x},
                sham::DDMultiRef{spans_y},
                cell_counts,
                [alpha](u32 i, const T *__restrict x, T *__restrict y) {
                    y[i] = alpha * y[i] + x[i];
                });
        }
    };

} // namespace

namespace shammodels::basegodunov::modules {

    template<class T>
    void NodeAYPX<T>::_impl_evaluate_internal() {
        auto edges = get_edges();

        edges.spans_x.check_sizes(edges.sizes.indexes);
        edges.spans_y.ensure_sizes(edges.sizes.indexes);

        KernelAYPX<T>::kernel(
            edges.spans_x.get_spans(),
            edges.spans_y.get_spans(),
            edges.alpha.value,
            edges.sizes.indexes,
            block_size);
    }

} // namespace shammodels::basegodunov::modules

template class shammodels::basegodunov::modules::NodeAYPX<f64>;
template class shammodels::basegodunov::modules::NodeAYPX<f64_3>;
