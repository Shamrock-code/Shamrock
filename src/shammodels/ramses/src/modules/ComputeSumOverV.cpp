// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright (c) 2021-2024 Timothée David--Cléris <tim.shamrock@proton.me>
// SPDX-License-Identifier: CeCILL Free Software License Agreement v2.1
// Shamrock is licensed under the CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

/**
 * @file ComputeSumOverV.cpp
 * @author Timothée David--Cléris (timothee.david--cleris@ens-lyon.fr)
 * @brief
 *
 */

#include "shammodels/ramses/modules/ComputeSumOverV.hpp"
#include "shambackends/kernel_call_distrib.hpp"
#include "shamcomm/logs.hpp"
#include "shammath/riemann.hpp"
#include "shamrock/patch/PatchDataField.hpp"
#include "shamsys/NodeInstance.hpp"

namespace shammodels::basegodunov::modules {

    template<class T>
    void NodeComputeSumOverV<T>::_impl_evaluate_internal() {
        auto edges = get_edges();

        edges.spans_field.check_sizes(edges.sizes.indexes);

        T loc_val = {};
        edges.spans_field.get_refs().for_each([&](u32 i, PatchDataField<T> &field_ref) {
            loc_val += field_ref.compute_sum();
        });

        T global_sum = shamalgs::collective::allreduce_sum(loc_val);

        edges.mean_val.value = (global_sum / edges.total_volume.value);

        // logger::raw_ln(loc_val, global_sum, edges.mean_val.value,edges.total_volume.value);
    }

    template<class T>
    std::string NodeComputeSumOverV<T>::_impl_get_tex() {

        auto block_count  = get_ro_edge_base(0).get_tex_symbol();
        auto field        = get_ro_edge_base(1).get_tex_symbol();
        auto total_volume = get_ro_edge_base(2).get_tex_symbol();
        auto mean         = get_rw_edge_base(0).get_tex_symbol();

        std::string tex = R"tex(
            Compute cell mass

            \begin{align}
            {mean} &=\sum_{i\in \Omega} {field}_i / {total_volume} \\
            \Omega = [0,{block_count} * N_{\rm cell/block}) \\
            N_{\rm cell/block} & = {block_size}
            \end{align}
        )tex";

        shambase::replace_all(tex, "{total_volume}", total_volume);
        shambase::replace_all(tex, "{field}", field);
        shambase::replace_all(tex, "{mean}", mean);
        shambase::replace_all(tex, "{block_count}", block_count);
        shambase::replace_all(tex, "{block_size}", shambase::format("{}", block_size));

        return tex;
    }

} // namespace shammodels::basegodunov::modules

template class shammodels::basegodunov::modules::NodeComputeSumOverV<f64>;
