// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright (c) 2021-2024 Timothée David--Cléris <tim.shamrock@proton.me>
// SPDX-License-Identifier: CeCILL Free Software License Agreement v2.1
// Shamrock is licensed under the CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

/**
 * @file CGResidualDdot.cpp
 * @author Léodasce Sewanou (leodasce.sewanou@ens-lyon.fr)
 * @brief
 *
 */

#include "shammodels/ramses/modules/CGResidualDdot.hpp"
#include "shamalgs/collective/reduction.hpp"
#include "shambackends/kernel_call_distrib.hpp"
#include "shamcomm/logs.hpp"
#include "shammath/riemann.hpp"
#include "shamrock/patch/PatchDataField.hpp"
#include "shamsys/NodeInstance.hpp"

namespace shammodels::basegodunov::modules {

    template<class T>
    void CGResidualDdot<T>::_impl_evaluate_internal() {
        auto edges = get_edges();

        edges.spans_phi_res.check_sizes(edges.sizes.indexes);

        T loc_val = {};
        edges.spans_phi_res.get_refs().for_each([&](u32 i, PatchDataField<T> &res_field_ref) {
            loc_val += res_field_ref.compute_dot_sum();
        });

        edges.res_ddot.value = shamalgs::collective::allreduce_sum(loc_val);
    }

    template<class T>
    std::string CGResidualDdot<T>::_impl_get_tex() {
        auto block_count   = get_ro_edge_base(0).get_tex_symbol();
        auto phi_res_field = get_ro_edge_base(1).get_tex_symbol();
        auto ddot_res      = get_rw_edge_base(0).get_tex_symbol();

        std::string tex = R"tex(
        Compute L2-norm squred of residual vector

        \begin{equation}
        {\mathbf{r}} = \mathbf{4\pi\mathbf{\mathrm{\rho}}} - \mathbf{A}\mathbf{\Phi}
        \end{equation}
    )tex";

        return tex;
    }

} // namespace shammodels::basegodunov::modules

template class shammodels::basegodunov::modules::CGResidualDdot<f64>;
