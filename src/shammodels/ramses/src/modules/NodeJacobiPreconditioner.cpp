// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright (c) 2021-2025 Timothée David--Cléris <tim.shamrock@proton.me>
// SPDX-License-Identifier: CeCILL Free Software License Agreement v2.1
// Shamrock is licensed under the CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

/**
 * @file NodeJacobiPreconditioner.cpp
 * @author Léodasce Sewanou (leodasce.sewanou@ens-lyon.fr)
 * @brief
 *
 */

#include "shammodels/ramses/modules/NodeJacobiPreconditioner.hpp"
#include "shambackends/kernel_call_distrib.hpp"
#include "shamsys/NodeInstance.hpp"

namespace {

    template<class T>
    struct KernelJacobiPreconditioner {

        inline static void kernel(
            const shambase::DistributedData<shamrock::PatchDataFieldSpanPointer<T>>
                &spans_cell_sizes,
            const shambase::DistributedData<shamrock::PatchDataFieldSpanPointer<T>> &spans_x,
            shambase::DistributedData<shamrock::PatchDataFieldSpanPointer<T>> &spans_y,
            const shambase::DistributedData<u32> &sizes,
            u32 block_size) {

            shambase::DistributedData<u32> cell_counts
                = sizes.map<u32>([&](u64 id, u32 block_count) {
                      u32 cell_count = block_count * block_size;
                      return cell_count;
                  });

            sham::distributed_data_kernel_call(
                shamsys::instance::get_compute_scheduler_ptr(),
                sham::DDMultiRef{spans_cell_sizes, spans_x},
                sham::DDMultiRef{spans_y},
                cell_counts,
                [block_size](
                    u32 i, const T *__restrict csize, const T *__restrict x, T *__restrict y) {
                    u32 block_id = i / block_size;
                    T dV         = csize[block_id];
                    // T aii = 6/(dV*dV);
                    y[i] = (dV * dV) * (x[i]) / 6;
                });
        }
    };

} // namespace

namespace shammodels::basegodunov::modules {

    template<class T>
    void NodeJacobiPreconditioner<T>::_impl_evaluate_internal() {
        auto edges = get_edges();
        edges.spans_block_cell_sizes.check_sizes(edges.sizes.indexes);
        edges.spans_x.check_sizes(edges.sizes.indexes);
        edges.spans_y.ensure_sizes(edges.sizes.indexes);

        KernelJacobiPreconditioner<T>::kernel(
            edges.spans_block_cell_sizes.get_spans(),
            edges.spans_x.get_spans(),
            edges.spans_y.get_spans(),
            edges.sizes.indexes,
            block_size);
    }

    template<class T>
    std::string NodeJacobiPreconditioner<T>::_impl_get_tex() {

        std::string tex = R"tex(
            JacobiPreconditioner
        )tex";

        return tex;
    }

} // namespace shammodels::basegodunov::modules

template class shammodels::basegodunov::modules::NodeJacobiPreconditioner<f64>;
