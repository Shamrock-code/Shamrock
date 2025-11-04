// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright (c) 2021-2025 Timothée David--Cléris <tim.shamrock@proton.me>
// SPDX-License-Identifier: CeCILL Free Software License Agreement v2.1
// Shamrock is licensed under the CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

/**
 * @file ComputeMass.cpp
 * @author Timothée David--Cléris (tim.shamrock@proton.me)
 * @brief
 *
 */

#include "shammodels/ramses/modules/ComputeMass.hpp"
#include "shambackends/kernel_call_distrib.hpp"
#include "shammath/riemann.hpp"
#include "shamrock/patch/PatchDataField.hpp"
#include "shamsys/NodeInstance.hpp"

namespace {

    template<class Tscal>
    struct KernelComputeMass {

        inline static void kernel(
            const shambase::DistributedData<shamrock::PatchDataFieldSpanPointer<Tscal>>
                &spans_cell_sizes,
            const shambase::DistributedData<shamrock::PatchDataFieldSpanPointer<Tscal>> &spans_rho,
            shambase::DistributedData<shamrock::PatchDataFieldSpanPointer<Tscal>> &spans_mass,
            const shambase::DistributedData<u32> &sizes,
            u32 block_size) {

            shambase::DistributedData<u32> cell_counts
                = sizes.map<u32>([&](u64 id, u32 block_count) {
                      u32 cell_count = block_count * block_size;
                      return cell_count;
                  });

            sham::distributed_data_kernel_call(
                shamsys::instance::get_compute_scheduler_ptr(),
                sham::DDMultiRef{spans_cell_sizes, spans_rho},
                sham::DDMultiRef{spans_mass},
                cell_counts,
                [block_size](
                    u32 i,
                    const Tscal *__restrict csize,
                    const Tscal *__restrict rho,
                    Tscal *__restrict mass) {
                    u32 block_id = i / block_size;
                    Tscal dV     = csize[block_id];
                    dV           = dV * dV * dV;

                    mass[i] = rho[i] * dV;
                });
        }
    };

} // namespace

namespace shammodels::basegodunov::modules {

    template<class Tvec, class TgridVec>
    void NodeComputeMass<Tvec, TgridVec>::_impl_evaluate_internal() {
        auto edges = get_edges();

        edges.spans_block_cell_sizes.check_sizes(edges.sizes.indexes);
        edges.spans_rhos.check_sizes(edges.sizes.indexes);

        edges.spans_mass.ensure_sizes(edges.sizes.indexes);

        KernelComputeMass<Tscal>::kernel(
            edges.spans_block_cell_sizes.get_spans(),
            edges.spans_rhos.get_spans(),
            edges.spans_mass.get_spans(),
            edges.sizes.indexes,
            block_size);
    }

    template<class Tvec, class TgridVec>
    std::string NodeComputeMass<Tvec, TgridVec>::_impl_get_tex() {

        auto block_count = get_ro_edge_base(0).get_tex_symbol();
        auto cell_size   = get_ro_edge_base(1).get_tex_symbol();
        auto rho         = get_ro_edge_base(2).get_tex_symbol();
        auto mass        = get_rw_edge_base(0).get_tex_symbol();

        std::string tex = R"tex(
            Compute cell mass

            \begin{align}
            {mass}_i &= {rho}_i {cell_size}_i^3 \\
            i &\in [0,{block_count} * N_{\rm cell/block}) \\
            N_{\rm cell/block} & = {block_size}
            \end{align}
        )tex";

        shambase::replace_all(tex, "{cell_size}", cell_size);
        shambase::replace_all(tex, "{rho}", rho);
        shambase::replace_all(tex, "{mass}", mass);
        shambase::replace_all(tex, "{block_count}", block_count);
        shambase::replace_all(tex, "{block_size}", shambase::format("{}", block_size));

        return tex;
    }

} // namespace shammodels::basegodunov::modules

template class shammodels::basegodunov::modules::NodeComputeMass<f64_3, i64_3>;
