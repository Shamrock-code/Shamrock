// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright (c) 2021-2025 Timothée David--Cléris <tim.shamrock@proton.me>
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
        // logger::raw_ln("SRed: \t", &edges.spans_in,"\n");
        // edges.spans_in.check_sizes(edges.sizes_no_gz.indexes);
        edges.spans_in.check_sizes(edges.sizes.indexes);
        T loc_val = {};

        edges.spans_in.get_refs().for_each([&](u32 i, PatchDataField<T> &res_field_ref) {
            loc_val += res_field_ref.compute_sum();
            // logger::raw_ln("id_a = [ ", i, " ] : ", loc_val, "\n");
        });
        edges.out_scal.value = shamalgs::collective::allreduce_sum(loc_val);

        logger::raw_ln("global = [ ", edges.out_scal.value, "\n");
    }

    template<class T>
    std::string NodeSumReduction<T>::_impl_get_tex() {
        auto block_count = get_ro_edge_base(0).get_tex_symbol();
        auto in          = get_ro_edge_base(1).get_tex_symbol();
        auto out_scal    = get_rw_edge_base(0).get_tex_symbol();

        std::string tex = R"tex(
        Compute sum-reduction of a given vector overall the computational grid
    )tex";

        return tex;
    }

} // namespace shammodels::basegodunov::modules

template class shammodels::basegodunov::modules::NodeSumReduction<f64>;
