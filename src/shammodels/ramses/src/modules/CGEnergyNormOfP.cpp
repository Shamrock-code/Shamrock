// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright (c) 2021-2024 Timothée David--Cléris <tim.shamrock@proton.me>
// SPDX-License-Identifier: CeCILL Free Software License Agreement v2.1
// Shamrock is licensed under the CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

/**
 * @file CGEnergyNormOfP.cpp
 * @author Léodasce Sewanou (leodasce.sewanou@ens-lyon.fr)
 * @brief
 *
 */

#include "shammodels/ramses/modules/CGEnergyNormOfP.hpp"
#include "shamalgs/collective/reduction.hpp"
#include "shamcomm/logs.hpp"
#include "shamrock/patch/PatchDataField.hpp"
#include "shamsys/NodeInstance.hpp"

namespace shammodels::basegodunov::modules {

    template<class T>
    void NodeCGEnergyNormOfP<T>::_impl_evaluate_internal() {
        auto edges = get_edges();

        edges.spans_phi_hadamard_prod.check_sizes(edges.sizes.indexes);

        T loc_val = {};
        edges.spans_phi_hadamard_prod.get_refs().for_each(
            [&](u32 i, PatchDataField<T> &res_field_ref) {
                loc_val += res_field_ref.compute_sum();
            });

        edges.e_norm.value = shamalgs::collective::allreduce_sum(loc_val);
    }

    template<class T>
    std::string NodeCGEnergyNormOfP<T>::_impl_get_tex() {
        auto block_count       = get_ro_edge_base(0).get_tex_symbol();
        auto phi_hadamard_prod = get_ro_edge_base(1).get_tex_symbol();
        auto e_norm            = get_rw_edge_base(0).get_tex_symbol();

        std::string tex = R"tex(
        Compute A-norm of the vector p

    )tex";

        return tex;
    }

} // namespace shammodels::basegodunov::modules

template class shammodels::basegodunov::modules::NodeCGEnergyNormOfP<f64>;
