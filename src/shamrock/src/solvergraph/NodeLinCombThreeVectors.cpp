// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright (c) 2021-2026 Timothée David--Cléris <tim.shamrock@proton.me>
// SPDX-License-Identifier: CeCILL Free Software License Agreement v2.1
// Shamrock is licensed under the CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

/**
 * @file NodeLinCombThreeVectors.cpp
 * @author Léodasce Sewanou (leodasce.sewanou@ens-lyon.fr)
 * @author Timothée David--Cléris (tim.shamrock@proton.me)
 * @brief
 *
 * Adapted from the PhD work of Léodasce Sewanou.
 */

#include "shambase/string.hpp"
#include "shambackends/kernel_call_distrib.hpp"
#include "shamrock/patch/PatchDataField.hpp"
#include "shamrock/solvergraph/NodeLinCombThreeVectors.hpp"
#include "shamsys/NodeInstance.hpp"

namespace shamrock::solvergraph {

    template<class T>
    void NodeLinCombThreeVectors<T>::_impl_evaluate_internal() {
        auto edges = get_edges();

        edges.spans_x.check_sizes(edges.sizes.indexes);
        edges.spans_y.check_sizes(edges.sizes.indexes);
        edges.spans_z.ensure_sizes(edges.sizes.indexes);

        Tscal alpha_0 = edges.alpha_0.data;
        Tscal alpha_1 = edges.alpha_1.data;
        Tscal alpha_2 = edges.alpha_2.data;

        // as in BLAS, z is not read when alpha_0 == 0, so that it can be freshly allocated
        bool read_z = alpha_0 != 0;

        shambase::DistributedData<u32> val_cnts
            = edges.sizes.indexes.template map<u32>([&](u64 id, u32 obj_cnt) {
                  return obj_cnt * nvar;
              });

        sham::distributed_data_kernel_call(
            shamsys::instance::get_compute_scheduler_ptr(),
            sham::DDMultiRef{edges.spans_x.get_spans(), edges.spans_y.get_spans()},
            sham::DDMultiRef{edges.spans_z.get_spans()},
            val_cnts,
            [alpha_0, alpha_1, alpha_2, read_z](
                u32 i, const T *__restrict x, const T *__restrict y, T *__restrict z) {
                // keep the left to right evaluation order, so that the rounding does not depend on
                // read_z
                z[i] = read_z ? alpha_0 * z[i] + alpha_1 * x[i] + alpha_2 * y[i]
                              : alpha_1 * x[i] + alpha_2 * y[i];
            });
    }

    template<class T>
    std::string NodeLinCombThreeVectors<T>::_impl_get_tex() const {
        std::string tex = R"tex(
            Linear combination of three vectors
            \begin{equation}
            {spans_z}_i \leftarrow {alpha_0} \, {spans_z}_i + {alpha_1} \, {spans_x}_i + {alpha_2} \, {spans_y}_i
            \end{equation}
        )tex";

        replace_edges_tex_symbols(tex);

        return tex;
    }

} // namespace shamrock::solvergraph

template class shamrock::solvergraph::NodeLinCombThreeVectors<f64>;
