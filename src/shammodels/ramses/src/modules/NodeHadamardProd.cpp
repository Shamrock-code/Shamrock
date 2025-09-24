// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright (c) 2021-2025 Timothée David--Cléris <tim.shamrock@proton.me>
// SPDX-License-Identifier: CeCILL Free Software License Agreement v2.1
// Shamrock is licensed under the CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

/**
 * @file NodeHadamardProd.cpp
 * @author Léodasce Sewanou (leodasce.sewanou@ens-lyon.fr)
 * @brief
 *
 */

#include "shammodels/ramses/modules/NodeHadamardProd.hpp"
#include "shambackends/kernel_call_distrib.hpp"
#include "shamrock/patch/PatchDataField.hpp"
#include "shamsys/NodeInstance.hpp"

namespace {

    template<class T>
    struct KernelHadamardProd {

        inline static void kernel(
            const shambase::DistributedData<shamrock::PatchDataFieldSpanPointer<T>> &spans_in1,
            const shambase::DistributedData<shamrock::PatchDataFieldSpanPointer<T>> &spans_in2,
            shambase::DistributedData<shamrock::PatchDataFieldSpanPointer<T>> &spans_out,
            const shambase::DistributedData<u32> &sizes,
            u32 block_size) {

            shambase::DistributedData<u32> cell_counts
                = sizes.map<u32>([&](u64 id, u32 block_count) {
                      u32 cell_count = block_count * block_size;
                      return cell_count;
                  });

            sham::distributed_data_kernel_call(
                shamsys::instance::get_compute_scheduler_ptr(),
                sham::DDMultiRef{spans_in1, spans_in2},
                sham::DDMultiRef{spans_out},
                cell_counts,
                [block_size](
                    u32 i, const T *__restrict in1, const T *__restrict in2, T *__restrict out) {
                    out[i] = in1[i] * in2[i];
                });
        }
    };

} // namespace

namespace shammodels::basegodunov::modules {

    template<class T>
    void NodeHadamardProd<T>::_impl_evaluate_internal() {
        auto edges = get_edges();

        edges.spans_in1.check_sizes(edges.sizes.indexes);
        edges.spans_in2.check_sizes(edges.sizes.indexes);

        edges.spans_out.ensure_sizes(edges.sizes.indexes);

        KernelHadamardProd<T>::kernel(
            edges.spans_in1.get_spans(),
            edges.spans_in2.get_spans(),
            edges.spans_out.get_spans(),
            edges.sizes.indexes,
            block_size);
    }

    template<class T>
    std::string NodeHadamardProd<T>::_impl_get_tex() {

        auto block_count = get_ro_edge_base(0).get_tex_symbol();
        auto in1         = get_ro_edge_base(1).get_tex_symbol();
        auto in2         = get_ro_edge_base(2).get_tex_symbol();
        auto out         = get_rw_edge_base(0).get_tex_symbol();

        std::string tex = R"tex(
            Compute Hadamard product
        )tex";

        return tex;
    }

} // namespace shammodels::basegodunov::modules

template class shammodels::basegodunov::modules::NodeHadamardProd<f64>;
