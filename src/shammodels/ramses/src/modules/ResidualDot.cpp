// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright (c) 2021-2024 Timothée David--Cléris <tim.shamrock@proton.me>
// SPDX-License-Identifier: CeCILL Free Software License Agreement v2.1
// Shamrock is licensed under the CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

/**
 * @file ResidualDot.cpp
 * @author Léodasce Sewanou (leodasce.sewanou@ens-lyon.fr)
 * @brief
 *
 */

#include "shammodels/ramses/modules/ResidualDot.hpp"
#include "shamalgs/collective/reduction.hpp"
#include "shamrock/patch/PatchDataField.hpp"

namespace shammodels::basegodunov::modules {

    template<class T>
    void ResidualDot<T>::_impl_evaluate_internal() {
        auto edges = get_edges();

        T loc_val = {};
        edges.spans_phi_res.get_refs().for_each([&](u32 i, PatchDataField<T> &res_field_ref) {
            loc_val += res_field_ref.compute_dot_sum();
        });

        edges.res_ddot.value = shamalgs::collective::allreduce_sum(loc_val);
    }

    template<class T>
    std::string ResidualDot<T>::_impl_get_tex() {
        auto field        = get_ro_edge_base(0).get_tex_symbol();
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
