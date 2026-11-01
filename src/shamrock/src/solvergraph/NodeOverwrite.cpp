// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright (c) 2021-2026 Timothée David--Cléris <tim.shamrock@proton.me>
// SPDX-License-Identifier: CeCILL Free Software License Agreement v2.1
// Shamrock is licensed under the CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

/**
 * @file NodeOverwrite.cpp
 * @author Léodasce Sewanou (leodasce.sewanou@ens-lyon.fr)
 * @author Timothée David--Cléris (tim.shamrock@proton.me)
 * @brief
 *
 * Adapted from the PhD work of Léodasce Sewanou.
 */

#include "shambase/string.hpp"
#include "shambackends/kernel_call_distrib.hpp"
#include "shamrock/patch/PatchDataField.hpp"
#include "shamrock/solvergraph/NodeOverwrite.hpp"
#include "shamsys/NodeInstance.hpp"

namespace shamrock::solvergraph {

    template<class T>
    void NodeOverwrite<T>::_impl_evaluate_internal() {
        auto edges = get_edges();

        edges.spans_x.check_sizes(edges.sizes.indexes);
        edges.spans_y.ensure_sizes(edges.sizes.indexes);

        shambase::DistributedData<u32> val_cnts
            = edges.sizes.indexes.template map<u32>([&](u64 id, u32 obj_cnt) {
                  return obj_cnt * nvar;
              });

        sham::distributed_data_kernel_call(
            shamsys::instance::get_compute_scheduler_ptr(),
            sham::DDMultiRef{edges.spans_x.get_spans()},
            sham::DDMultiRef{edges.spans_y.get_spans()},
            val_cnts,
            [](u32 i, const T *__restrict x, T *__restrict y) {
                y[i] = x[i];
            });
    }

    template<class T>
    std::string NodeOverwrite<T>::_impl_get_tex() const {
        std::string tex = R"tex(
            Overwrite a field with another one
            \begin{equation}
            {spans_y}_i = {spans_x}_i
            \end{equation}
        )tex";

        replace_edges_tex_symbols(tex);

        return tex;
    }

} // namespace shamrock::solvergraph

template class shamrock::solvergraph::NodeOverwrite<f64>;
