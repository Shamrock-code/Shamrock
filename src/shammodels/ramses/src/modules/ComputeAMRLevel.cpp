// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright (c) 2021-2025 Timothée David--Cléris <tim.shamrock@proton.me>
// SPDX-License-Identifier: CeCILL Free Software License Agreement v2.1
// Shamrock is licensed under the CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

/**
 * @file ComputeAMRLevel.cpp
 * @author Timothée David--Cléris (tim.shamrock@proton.me)
 * @brief
 */

#include "shambase/DistributedData.hpp"
#include "shambase/assert.hpp"
#include "shambase/integer.hpp"
#include "shambase/logs/loglevels.hpp"
#include "shambackends/kernel_call.hpp"
#include "shambackends/kernel_call_distrib.hpp"
#include "shambackends/math.hpp"
#include "shammodels/ramses/modules/ComputeAMRLevel.hpp"
#include "shamrock/patch/PatchDataField.hpp"
#include "shamsys/NodeInstance.hpp"

namespace shammodels::basegodunov::modules {

    template<class TgridVec>
    void ComputeAMRLevel<TgridVec>::_impl_evaluate_internal() {
        auto edges = get_edges();

        edges.block_min.check_sizes(edges.sizes.indexes);
        edges.block_max.check_sizes(edges.sizes.indexes);

        edges.block_level.ensure_sizes(edges.sizes.indexes);

        auto &q = shamsys::instance::get_compute_scheduler().get_queue();

        const auto &spans_block_min = edges.block_min.get_spans();
        const auto &spans_block_max = edges.block_max.get_spans();
        auto &spans_block_level     = edges.block_level.get_spans();

        edges.sizes.indexes.for_each([&](u64 id, const u64 &n) {
            TgridVec l0_ref = edges.level0_size.values.get(id);

            sham::kernel_call(
                q,
                sham::MultiRef{spans_block_min.get(id), spans_block_max.get(id)},
                sham::MultiRef{spans_block_level.get(id)},
                n,
                [l0_ref](
                    u64 id,
                    const TgridVec *__restrict block_min,
                    const TgridVec *__restrict block_max,
                    TgridUint *__restrict block_level) {
                    TgridVec block_size = block_max[id] - block_min[id];

                    SHAM_ASSERT(block_size.x() > 0 && block_size.y() > 0 && block_size.z() > 0);

                    Tgridscal l0 = (Tgridscal) l0_ref.x();
                    Tgridscal s  = (Tgridscal) block_size.x();

                    SHAM_ASSERT(l0 >= s);

                    Tgridscal fact = l0 / s;

                    SHAM_ASSERT(shambase::is_pow_of_two(fact));

                    block_level[id] = sham::log2_pow2_num<Tgridscal>(fact);
                });
        });
    }

    template<class TgridVec>
    std::string ComputeAMRLevel<TgridVec>::_impl_get_tex() {
        return "TODO";
    }

} // namespace shammodels::basegodunov::modules

template class shammodels::basegodunov::modules::ComputeAMRLevel<i64_3>;
