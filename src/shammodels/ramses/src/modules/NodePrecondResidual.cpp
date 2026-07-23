// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright (c) 2021-2026 Timothée David--Cléris <tim.shamrock@proton.me>
// SPDX-License-Identifier: CeCILL Free Software License Agreement v2.1
// Shamrock is licensed under the CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

/**
 * @file NodePrecondResidual.cpp
 * @author Léodasce Sewanou (leodasce.sewanou@ens-lyon.fr)
 * @brief
 *
 */
#include "shammodels/ramses/modules/NodePrecondResidual.hpp"
#include "shambackends/kernel_call_distrib.hpp"
#include "shambackends/typeAliasVec.hpp"
#include "shambackends/vec.hpp"
#include "shammodels/ramses/SolverConfig.hpp"
#include "shamrock/patch/PatchDataFieldSpan.hpp"
#include "shamsys/NodeInstance.hpp"

template<class T>
struct KernelPrecondRes {

    inline static void kernel(
        const shambase::DistributedData<u32> &sizes,
        const shambase::DistributedData<shamrock::PatchDataFieldSpanPointer<T>>
            &spans_block_cell_sizes,
        const shambase::DistributedData<shamrock::PatchDataFieldSpanPointer<T>> &spans_x,
        shambase::DistributedData<shamrock::PatchDataFieldSpanPointer<T>> &spans_y,
        u32 block_size) {

        shambase::DistributedData<u32> cell_counts = sizes.map<u32>([&](u64 id, u32 block_count) {
            u32 cell_count = block_count * block_size;
            return cell_count;
        });

        sham::distributed_data_kernel_call(
            shamsys::instance::get_compute_scheduler_ptr(),
            sham::DDMultiRef{spans_block_cell_sizes, spans_x},
            sham::DDMultiRef{spans_y},
            cell_counts,
            [block_size](u32 i, const T *__restrict bsize, const T *__restrict x, T *__restrict y) {
                const u32 block_id = i / block_size;
                auto delta_cell    = bsize[block_id];
                // y[i]               = x[i] / (6. * delta_cell);
                y[i] = (x[i] * delta_cell * delta_cell) / (6.);
            });
    }
};

template<class T>
void shammodels::basegodunov::modules::NodePrecondRes<T>::_impl_evaluate_internal() {
    auto edges = get_edges();

    edges.spans_y.ensure_sizes(edges.sizes.indexes);

    KernelPrecondRes<T>::kernel(
        edges.sizes.indexes,
        edges.block_cell_sizes.get_spans(),
        edges.spans_x.get_spans(),
        edges.spans_y.get_spans(),
        block_size);
}

template class shammodels::basegodunov::modules::NodePrecondRes<f64>;
template class shammodels::basegodunov::modules::NodePrecondRes<f64_3>;
