// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright (c) 2021-2025 Timothée David--Cléris <tim.shamrock@proton.me>
// SPDX-License-Identifier: CeCILL Free Software License Agreement v2.1
// Shamrock is licensed under the CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

/**
 * @file NodeLinCombThreeVectors.cpp
 * @author Léodasce Sewanou (leodasce.sewanou@ens-lyon.fr)
 * @brief
 *
 */

#include "shammodels/ramses/modules/NodeLinCombThreeVectors.hpp"
#include "shambackends/kernel_call_distrib.hpp"
#include "shamsys/NodeInstance.hpp"

namespace {

    template<class T>
    struct KernelLinCombThreeVectors {

        inline static void kernel(
            const shambase::DistributedData<shamrock::PatchDataFieldSpanPointer<T>> &spans_x,
            const shambase::DistributedData<shamrock::PatchDataFieldSpanPointer<T>> &spans_y,
            shambase::DistributedData<shamrock::PatchDataFieldSpanPointer<T>> &spans_z,
            const T alpha_0,
            const T alpha_1,
            const T alpha_2,
            const shambase::DistributedData<u32> &sizes,
            u32 block_size) {

            shambase::DistributedData<u32> cell_counts
                = sizes.map<u32>([&](u64 id, u32 block_count) {
                      u32 cell_count = block_count * block_size;
                      return cell_count;
                  });

            sham::distributed_data_kernel_call(
                shamsys::instance::get_compute_scheduler_ptr(),
                sham::DDMultiRef{spans_x, spans_y},
                sham::DDMultiRef{spans_z},
                cell_counts,
                [alpha_0, alpha_1, alpha_2](
                    u32 i, const T *__restrict x, const T *__restrict y, T *__restrict z) {
                    z[i] = alpha_0 * z[i] + alpha_1 * x[i] + alpha_2 * y[i];
                });
        }
    };

} // namespace

namespace shammodels::basegodunov::modules {

    template<class T>
    void NodeLinCombThreeVectors<T>::_impl_evaluate_internal() {
        auto edges = get_edges();

        edges.spans_x.check_sizes(edges.sizes.indexes);
        edges.spans_y.check_sizes(edges.sizes.indexes);
        edges.spans_z.ensure_sizes(edges.sizes.indexes);

        KernelLinCombThreeVectors<T>::kernel(
            edges.spans_x.get_spans(),
            edges.spans_y.get_spans(),
            edges.spans_z.get_spans(),
            edges.alpha_0.value,
            edges.alpha_1.value,
            edges.alpha_2.value,
            edges.sizes.indexes,
            block_size);
    }

    template<class T>
    std::string NodeLinCombThreeVectors<T>::_impl_get_tex() {

        std::string tex = R"tex(
            LinComb(3 vectors)
        )tex";

        return tex;
    }

} // namespace shammodels::basegodunov::modules

template class shammodels::basegodunov::modules::NodeLinCombThreeVectors<f64>;
