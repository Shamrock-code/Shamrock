// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright (c) 2021-2024 Timothée David--Cléris <tim.shamrock@proton.me>
// SPDX-License-Identifier: CeCILL Free Software License Agreement v2.1
// Shamrock is licensed under the CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

/**
 * @file ComputeCubeCellSizes.cpp
 * @author Timothée David--Cléris (timothee.david--cleris@ens-lyon.fr)
 * @brief
 *
 */

#include "shammodels/ramses/modules/ComputeCubeCellSizes.hpp"
#include "shambackends/kernel_call_distrib.hpp"
#include "shamrock/patch/PatchDataField.hpp"
#include "shamsys/NodeInstance.hpp"

namespace {

    template<class TgridVec>
    struct KernelComputeCubeCellSizes {
        using TgridScal = typename std::make_unsigned<shambase::VecComponent<TgridVec>>::type;

        inline static void kernel(
            const shambase::DistributedData<shamrock::PatchDataFieldSpanPointer<TgridVec>>
                &spans_block_min,
            const shambase::DistributedData<shamrock::PatchDataFieldSpanPointer<TgridVec>>
                &spans_block_max,
            shambase::DistributedData<shamrock::PatchDataFieldSpanPointer<TgridScal>>
                &spans_block_cell_sizes,
            const shambase::DistributedData<u32> &sizes) {

            const shambase::DistributedData<u32> &block_counts = sizes;

            sham::distributed_data_kernel_call(
                shamsys::instance::get_compute_scheduler_ptr(),
                sham::DDMultiRef{spans_block_min, spans_block_max},
                sham::DDMultiRef{spans_block_cell_sizes},
                block_counts,
                [](u32 i,
                   const TgridVec *__restrict acc_block_min,
                   const TgridVec *__restrict acc_block_max,
                   TgridScal *__restrict bsize) {
                    TgridVec lower = acc_block_min[i];
                    TgridVec upper = acc_block_max[i];

                    TgridVec block_cell_size = (upper - lower);

                    TgridScal res = TgridScal(block_cell_size.x());

                    bsize[i] = res;
                });
        }
    };

} // namespace

namespace shammodels::basegodunov::modules {

    template<class TgridVec>
    void NodeComputeCubeCellSizes<TgridVec>::_impl_evaluate_internal() {
        auto edges = get_edges();

        edges.spans_block_min.check_sizes(edges.sizes.indexes);
        edges.spans_block_max.check_sizes(edges.sizes.indexes);

        edges.spans_block_cell_sizes.ensure_sizes(edges.sizes.indexes);

        KernelComputeCubeCellSizes<TgridVec>::kernel(
            edges.spans_block_min.get_spans(),
            edges.spans_block_max.get_spans(),
            edges.spans_block_cell_sizes.get_spans(),
            edges.sizes.indexes);
    }

    template<class TgridVec>
    std::string NodeComputeCubeCellSizes<TgridVec>::_impl_get_tex() {

        auto block_count      = get_ro_edge_base(0).get_tex_symbol();
        auto block_min        = get_ro_edge_base(1).get_tex_symbol();
        auto block_max        = get_ro_edge_base(2).get_tex_symbol();
        auto block_cell_sizes = get_rw_edge_base(0).get_tex_symbol();

        std::string tex = R"tex(
            Compute cell AABBs

            \begin{align}
            {block_cell_sizes}_i &= \mathbf{e}_x \cdot ({block_max}_i - {block_min}_i) \\
            i &\in [0,{block_count})
            \end{align}
        )tex";

        shambase::replace_all(tex, "{block_count}", block_count);
        shambase::replace_all(tex, "{block_min}", block_min);
        shambase::replace_all(tex, "{block_max}", block_max);
        shambase::replace_all(tex, "{block_cell_sizes}", block_cell_sizes);

        return tex;
    }

} // namespace shammodels::basegodunov::modules

template class shammodels::basegodunov::modules::NodeComputeCubeCellSizes<i64_3>;
