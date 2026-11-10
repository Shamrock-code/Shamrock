// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright (c) 2021-2026 Timothée David--Cléris <tim.shamrock@proton.me>
// SPDX-License-Identifier: CeCILL Free Software License Agreement v2.1
// Shamrock is licensed under the CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

/**
 * @file ResidualDot.cpp
 * @author Léodasce Sewanou (leodasce.sewanou@ens-lyon.fr)
 * @author Timothée David--Cléris (tim.shamrock@proton.me)
 * @brief Squared L2 norm of a field, summed over all patches and ranks.
 *
 * Adapted from the PhD work of Léodasce Sewanou.
 */

#include "shambase/exception.hpp"
#include "shambase/string.hpp"
#include "shamalgs/collective/reduction.hpp"
#include "shamalgs/primitives/dot_sum.hpp"
#include "shammodels/ramses/modules/ResidualDot.hpp"
#include "shamrock/patch/PatchDataField.hpp"
#include <functional>

namespace shammodels::basegodunov::modules {

    template<class T>
    void ResidualDot<T>::_impl_evaluate_internal() {
        auto edges = get_edges();

        Tscal loc_val = {};

        // floating-point addition is not associative: accumulate the patches in a fixed order
        edges.spans_phi_res.get_refs().for_each_ordered(
            [&](u64 id_patch, const std::reference_wrapper<PatchDataField<T>> &field_ref) {
                PatchDataField<T> &field = field_ref.get();
                u32 val_cnt              = edges.sizes.indexes.get(id_patch) * nvar;

                // the field may hold more values than summed (ghost blocks), but never less
                if (field.get_val_cnt() < val_cnt) {
                    shambase::throw_with_loc<std::runtime_error>(sham::format(
                        "field too small on patch {}: expected at least {} values, got {}",
                        id_patch,
                        val_cnt,
                        field.get_val_cnt()));
                }

                loc_val += shamalgs::primitives::dot_sum(field.get_buf(), 0, val_cnt);
            });

        edges.res_ddot.data = shamalgs::collective::allreduce_sum(loc_val);
    }

    template<class T>
    std::string ResidualDot<T>::_impl_get_tex() const {
        auto field        = get_ro_edge_base(1).get_tex_symbol();
        auto residual_dot = get_rw_edge_base(0).get_tex_symbol();

        std::string tex = R"tex(
            Compute L2-norm squared of residual vector
            \begin{equation}
            {residual_dot} &= \sum_{i \in [0,N_{field})} {field}_i \cdot {field}_i
            \end{equation}
        )tex";

        shambase::replace_all(tex, "{field}", field);
        shambase::replace_all(tex, "{residual_dot}", residual_dot);

        return tex;
    }

} // namespace shammodels::basegodunov::modules

template class shammodels::basegodunov::modules::ResidualDot<f64>;
template class shammodels::basegodunov::modules::ResidualDot<f64_3>;
