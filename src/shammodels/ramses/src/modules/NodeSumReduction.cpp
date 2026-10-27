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
 * @author Timothée David--Cléris (tim.shamrock@proton.me)
 * @brief
 *
 * Adapted from the PhD work of Léodasce Sewanou.
 */

#include "shambase/exception.hpp"
#include "shambase/string.hpp"
#include "shamalgs/collective/reduction.hpp"
#include "shamalgs/primitives/reduction.hpp"
#include "shammodels/ramses/modules/NodeSumReduction.hpp"
#include "shamrock/patch/PatchDataField.hpp"
#include "shamsys/NodeInstance.hpp"

namespace shammodels::basegodunov::modules {

    template<class T>
    void NodeSumReduction<T>::_impl_evaluate_internal() {
        auto edges = get_edges();

        edges.spans_in.check_sizes(edges.sizes.indexes);

        auto dev_sched = shamsys::instance::get_compute_scheduler_ptr();

        T loc_val = {};

        // floating-point addition is not associative: accumulate the patches in a fixed order
        edges.spans_in.get_refs().for_each_ordered(
            [&](u64 id_patch, const std::reference_wrapper<PatchDataField<T>> &field_ref) {
                const PatchDataField<T> &field = field_ref.get();
                u32 cell_count                 = edges.sizes.indexes.get(id_patch) * block_size;

                if (field.get_val_cnt() != cell_count) {
                    shambase::throw_with_loc<std::runtime_error>(sham::format(
                        "field size mismatch on patch {}: expected {} values, got {}",
                        id_patch,
                        cell_count,
                        field.get_val_cnt()));
                }

                if (cell_count == 0) {
                    return;
                }

                loc_val += shamalgs::primitives::sum(dev_sched, field.get_buf(), 0, cell_count);
            });

        edges.out_scal.value = shamalgs::collective::allreduce_sum(loc_val);
    }

    template<class T>
    std::string NodeSumReduction<T>::_impl_get_tex() const {
        std::string tex = R"tex(
            Global sum reduction
            \begin{equation}
            {out_scal} = \sum_{i} {spans_in}_i
            \end{equation}
        )tex";

        replace_edges_tex_symbols(tex);

        return tex;
    }

} // namespace shammodels::basegodunov::modules

template class shammodels::basegodunov::modules::NodeSumReduction<f64>;
