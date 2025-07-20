// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright (c) 2021-2024 Timothée David--Cléris <tim.shamrock@proton.me>
// SPDX-License-Identifier: CeCILL Free Software License Agreement v2.1
// Shamrock is licensed under the CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

/**
 * @file CGComputeNewSearchDir.cpp
 * @author Léodasce Sewanou (leodasce.sewanou@ens-lyon.fr)
 * @brief
 *
 */

#include "shammodels/ramses/modules/CGComputeNewSearchDir.hpp"
#include "shambackends/kernel_call_distrib.hpp"
#include "shamrock/patch/PatchDataField.hpp"
#include "shamsys/NodeInstance.hpp"

namespace {

    template<class T>
    struct KernelComputeNewSearchDir {

        inline static void kernel(
            const shambase::DistributedData<shamrock::PatchDataFieldSpanPointer<T>> &spans_phi_res,
            shambase::DistributedData<shamrock::PatchDataFieldSpanPointer<T>> &spans_phi_p,
            const shambase::DistributedData<u32> &sizes,
            u32 block_size,
            T beta) {

            shambase::DistributedData<u32> cell_counts
                = sizes.map<u32>([&](u64 id, u32 block_count) {
                      u32 cell_count = block_count * block_size;
                      return cell_count;
                  });

            sham::distributed_data_kernel_call(
                shamsys::instance::get_compute_scheduler_ptr(),
                sham::DDMultiRef{spans_phi_res},
                sham::DDMultiRef{spans_phi_p},
                cell_counts,
                [beta](u32 i, const T *__restrict phi_res, T *__restrict phi_p) {
                    phi_p[i] = phi_res[i] + beta * phi_p[i];
                });
        }
    };

} // namespace

namespace shammodels::basegodunov::modules {

    template<class T>
    void NodeComputeNewSearchDir<T>::_impl_evaluate_internal() {
        auto edges = get_edges();

        edges.spans_phi_res.check_sizes(edges.sizes.indexes);

        edges.spans_phi_p.ensure_sizes(edges.sizes.indexes);

        KernelComputeNewSearchDir<T>::kernel(
            edges.spans_phi_res.get_spans(),
            edges.spans_phi_p.get_spans(),
            edges.sizes.indexes,
            block_size,
            beta);
    }

    template<class T>
    std::string NodeComputeNewSearchDir<T>::_impl_get_tex() {

        std::string tex = R"tex(
            Compute the search direction vector for the next iteration of CG
        )tex";

        return tex;
    }

} // namespace shammodels::basegodunov::modules

template class shammodels::basegodunov::modules::NodeComputeNewSearchDir<f64>;
