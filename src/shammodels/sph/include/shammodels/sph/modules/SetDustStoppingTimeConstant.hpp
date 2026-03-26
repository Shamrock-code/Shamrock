// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright (c) 2021-2026 Timothée David--Cléris <tim.shamrock@proton.me>
// SPDX-License-Identifier: CeCILL Free Software License Agreement v2.1
// Shamrock is licensed under the CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

#pragma once

/**
 * @file SetDustStoppingTimeConstant.hpp
 * @author Timothée David--Cléris (tim.shamrock@proton.me)
 * @brief
 *
 */

#include "shambackends/DeviceBuffer.hpp"
#include "shambackends/kernel_call_distrib.hpp"
#include "shambackends/vec.hpp"
#include "shammodels/sph/math/density.hpp"
#include "shamrock/solvergraph/IFieldSpan.hpp"
#include "shamrock/solvergraph/INode.hpp"
#include "shamrock/solvergraph/Indexes.hpp"
#include "shamrock/solvergraph/ScalarEdge.hpp"
#include "shamsys/NodeInstance.hpp"
#include <vector>

#define NODE_SET_DUST_STOPPING_TIME_CTE_EDGES(X_RO, X_RW)                                          \
    /* scalars */                                                                                  \
    X_RO(shamrock::solvergraph::ScalarEdge<std::vector<Tscal>>, t_j_0)                             \
                                                                                                   \
    /* counts */                                                                                   \
    X_RO(shamrock::solvergraph::Indexes<u32>, part_counts)                                         \
                                                                                                   \
    /* fields */                                                                                   \
    X_RW(shamrock::solvergraph::IFieldSpan<Tscal>, t_j)

namespace shammodels::sph::modules {

    template<class Tvec, template<class> class SPHKernel>
    class SetDustStoppingTimeConstant : public shamrock::solvergraph::INode {

        using Tscal  = shambase::VecComponent<Tvec>;
        using Kernel = SPHKernel<Tscal>;

        static constexpr Tscal kernel_radius = SPHKernel<Tscal>::Rkern;

        u32 ndust;

        public:
        SetDustStoppingTimeConstant(u32 ndust) : ndust(ndust) {}

        EXPAND_NODE_EDGES(NODE_SET_DUST_STOPPING_TIME_CTE_EDGES)

        inline void _impl_evaluate_internal() {

            __shamrock_stack_entry();

            auto edges = get_edges();

            auto &part_counts                   = edges.part_counts.indexes;
            const std::vector<Tscal> &inputs_tj = edges.t_j_0.value;

            // ensure that the output edges are of size part_counts
            edges.t_j.ensure_sizes(part_counts);

            sham::DeviceBuffer<Tscal> t_j_0(ndust, shamsys::instance::get_compute_scheduler_ptr());
            t_j_0.copy_from_stdvec(inputs_tj);

            auto &q = shamsys::instance::get_compute_scheduler().get_queue();

            part_counts.for_each([&](u64 id, u32 count) {
                // call the kernel for each patches with part_counts.get(id_patch) threads of patch
                // id_patch
                sham::kernel_call(
                    q,
                    sham::MultiRef{t_j_0},
                    sham::MultiRef{edges.t_j.get_spans().get(id)},
                    part_counts.get(id) * ndust,
                    [ndust
                     = ndust](u32 thread_id, const Tscal *__restrict t_j_0, Tscal *__restrict t_j) {
                        u32 jdust      = thread_id % ndust;
                        t_j[thread_id] = t_j_0[jdust];
                    });
            });
        }

        inline virtual std::string _impl_get_label() const { return "ComputeDustTtilde"; };

        inline virtual std::string _impl_get_tex() const { return "TODO"; };
    };
} // namespace shammodels::sph::modules
