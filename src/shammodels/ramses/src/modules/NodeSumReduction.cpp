// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright (c) 2021-2026 Timothée David--Cléris <tim.shamrock@proton.me>
// SPDX-License-Identifier: CeCILL Free Software License Agreement v2.1
// Shamrock is licensed under the CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

/**
 * @file NodeSumReduction.cpp
 * @author Léodasce Sewanou (leodasce.sewanou@ens-lyon.fr)
 * @brief
 *
 */

#include "shammodels/ramses/modules/NodeSumReduction.hpp"
#include "shamalgs/collective/reduction.hpp"
#include "shamrock/patch/PatchDataField.hpp"

namespace shammodels::basegodunov::modules {

    template<class T>
    void NodeSumReduction<T>::_impl_evaluate_internal() {
        auto edges = get_edges();

        edges.spans_in.check_sizes(edges.sizes.indexes);
        T loc_val = {};

        edges.spans_in.get_refs().for_each([&](u32 patch_id, PatchDataField<T> &res_field_ref) {
            auto buf_length = block_size * edges.sizes.indexes.get(patch_id);
            // loc_val += res_field_ref.compute_sum();
            auto dev_sched = shamsys::instance::get_compute_scheduler_ptr();
            loc_val += shamalgs::primitives::sum(dev_sched, res_field_ref.get_buf(), 0, buf_length);
        });
        edges.out_scal.value = shamalgs::collective::allreduce_sum(loc_val);
    }

} // namespace shammodels::basegodunov::modules

template class shammodels::basegodunov::modules::NodeSumReduction<f64>;
