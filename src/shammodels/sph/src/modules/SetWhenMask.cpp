// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright (c) 2021-2026 Timothée David--Cléris <tim.shamrock@proton.me>
// SPDX-License-Identifier: CeCILL Free Software License Agreement v2.1
// Shamrock is licensed under the CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

/**
 * @file SetWhenMask.cpp
 * @author Timothée David--Cléris (tim.shamrock@proton.me)
 * @author Yona Lapeyre (yona.lapeyre@ens-lyon.fr)
 * @brief Implementation of the SetWhenMask solver graph node.
 *
 */

#include "shambase/stacktrace.hpp"
#include "shambase/string.hpp"
#include "shambackends/kernel_call_distrib.hpp"
#include "shammodels/sph/modules/SetWhenMask.hpp"
#include "shamrock/patch/PatchDataField.hpp"
#include "shamsys/NodeInstance.hpp"

namespace shammodels::sph::modules {

    template<class T>
    void SetWhenMask<T>::_impl_evaluate_internal() {
        __shamrock_stack_entry();

        auto edges = get_edges();

        auto dev_sched = shamsys::instance::get_compute_scheduler_ptr();

        edges.mask.check_sizes(edges.part_counts.indexes);
        edges.field_to_set.ensure_sizes(edges.part_counts.indexes);

        auto var_counts = edges.part_counts.indexes.template map<u32>([&](u64, u32 count) {
            return count * nvar;
        });

        sham::distributed_data_kernel_call(
            dev_sched,
            sham::DDMultiRef{edges.mask.get_spans()},
            sham::DDMultiRef{edges.field_to_set.get_spans()},
            var_counts,
            [val_to_set = this->val_to_set,
             nvar       = this->nvar](u32 id_var, const u32 *mask, T *field_to_set) {
                if (mask[id_var / nvar] != 0) {
                    field_to_set[id_var] = val_to_set;
                }
            });
    }

    template<class T>
    std::string SetWhenMask<T>::_impl_get_tex() const {
        auto mask         = get_ro_edge_base(1).get_tex_symbol();
        auto field_to_set = get_rw_edge_base(0).get_tex_symbol();

        std::string tex = R"tex(
    Set the field to a constant value \(c\) where the mask is non-zero
    \begin{align}
    {field}_{i,k} &= c \quad \text{if } {mask}_i \neq 0, \quad k = 0, \dots, {nvar} - 1
    \end{align}
    )tex";

        shambase::replace_all(tex, "{field}", field_to_set);
        shambase::replace_all(tex, "{mask}", mask);
        shambase::replace_all(tex, "{nvar}", std::to_string(nvar));

        return tex;
    }

} // namespace shammodels::sph::modules

template class shammodels::sph::modules::SetWhenMask<f64>;
template class shammodels::sph::modules::SetWhenMask<f64_3>;
