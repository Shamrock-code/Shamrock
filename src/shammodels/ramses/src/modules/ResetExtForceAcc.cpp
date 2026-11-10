// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright (c) 2021-2026 Timothée David--Cléris <tim.shamrock@proton.me>
// SPDX-License-Identifier: CeCILL Free Software License Agreement v2.1
// Shamrock is licensed under the CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

/**
 * @file ResetExtForceAcc.cpp
 * @author Léodasce Sewanou (leodasce.sewanou@ens-lyon.fr)
 * @author Timothée David--Cléris (tim.shamrock@proton.me)
 * @brief
 *
 */

#include "shambase/string.hpp"
#include "shambackends/kernel_call_distrib.hpp"
#include "shammodels/ramses/modules/ResetExtForceAcc.hpp"
#include "shamrock/patch/PatchDataField.hpp"
#include "shamsys/NodeInstance.hpp"

namespace shammodels::basegodunov::modules {

    template<class Tvec>
    void ResetExtForceAcc<Tvec>::_impl_evaluate_internal() {
        auto edges = get_edges();

        // (only reallocated if the block counts have changed)
        edges.spans_a_ext.ensure_sizes(edges.sizes.indexes);

        shambase::DistributedData<u32> cell_counts
            = edges.sizes.indexes.template map<u32>([&](u64 id, u32 block_count) {
                  return block_count * block_size;
              });

        sham::distributed_data_kernel_call(
            shamsys::instance::get_compute_scheduler_ptr(),
            sham::DDMultiRef{},
            sham::DDMultiRef{edges.spans_a_ext.get_spans()},
            cell_counts,
            [](u32 i, Tvec *__restrict a_ext) {
                a_ext[i] = Tvec{};
            });
    }

    template<class Tvec>
    std::string ResetExtForceAcc<Tvec>::_impl_get_tex() const {
        auto a_ext = get_rw_edge_base(0).get_tex_symbol();

        std::string tex = R"tex(
            Reset external forces acceleration

            \begin{align}
            {a_ext}_i &= 0
            \end{align}
        )tex";

        shambase::replace_all(tex, "{a_ext}", a_ext);

        return tex;
    }

} // namespace shammodels::basegodunov::modules

template class shammodels::basegodunov::modules::ResetExtForceAcc<f64_3>;
