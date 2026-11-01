// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright (c) 2021-2026 Timothée David--Cléris <tim.shamrock@proton.me>
// SPDX-License-Identifier: CeCILL Free Software License Agreement v2.1
// Shamrock is licensed under the CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

/**
 * @file NodeHadamardProd.cpp
 * @author Léodasce Sewanou (leodasce.sewanou@ens-lyon.fr)
 * @author Timothée David--Cléris (tim.shamrock@proton.me)
 * @brief
 *
 * Adapted from the PhD work of Léodasce Sewanou.
 */

#include "shambase/string.hpp"
#include "shambackends/kernel_call_distrib.hpp"
#include "shamrock/patch/PatchDataField.hpp"
#include "shamrock/solvergraph/NodeHadamardProd.hpp"
#include "shamsys/NodeInstance.hpp"

namespace shamrock::solvergraph {

    template<class T>
    void NodeHadamardProd<T>::_impl_evaluate_internal() {
        auto edges = get_edges();

        edges.spans_in1.check_sizes(edges.sizes.indexes);
        edges.spans_in2.check_sizes(edges.sizes.indexes);
        edges.spans_out.ensure_sizes(edges.sizes.indexes);

        shambase::DistributedData<u32> val_cnts
            = edges.sizes.indexes.template map<u32>([&](u64 id, u32 obj_cnt) {
                  return obj_cnt * nvar;
              });

        sham::distributed_data_kernel_call(
            shamsys::instance::get_compute_scheduler_ptr(),
            sham::DDMultiRef{edges.spans_in1.get_spans(), edges.spans_in2.get_spans()},
            sham::DDMultiRef{edges.spans_out.get_spans()},
            val_cnts,
            [](u32 i, const T *__restrict in1, const T *__restrict in2, T *__restrict out) {
                out[i] = in1[i] * in2[i];
            });
    }

    template<class T>
    std::string NodeHadamardProd<T>::_impl_get_tex() const {
        std::string tex = R"tex(
            Hadamard (elementwise) product
            \begin{equation}
            {spans_out}_i = {spans_in1}_i \, {spans_in2}_i
            \end{equation}
        )tex";

        replace_edges_tex_symbols(tex);

        return tex;
    }

} // namespace shamrock::solvergraph

template class shamrock::solvergraph::NodeHadamardProd<f64>;
