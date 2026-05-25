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
 * @file NodeMonofluidTVIAddSourceTerm.hpp
 * @author Timothée David--Cléris (tim.shamrock@proton.me)
 * @brief
 *
 */

#include "shambackends/kernel_call_distrib.hpp"
#include "shambackends/math.hpp"
#include "shambackends/vec.hpp"
#include "shamcomm/logs.hpp"
#include "shamrock/solvergraph/IFieldSpan.hpp"
#include "shamrock/solvergraph/INode.hpp"
#include "shamrock/solvergraph/Indexes.hpp"
#include "shamrock/solvergraph/ScalarEdge.hpp"
#include "shamsys/NodeInstance.hpp"
#include <experimental/mdspan>

#define NODE_EDGES(X_RO, X_RW)                                                                     \
                                                                                                   \
    /* counts */                                                                                   \
    X_RO(shamrock::solvergraph::Indexes<u32>, part_counts)                                         \
    X_RO(shamrock::solvergraph::ScalarEdge<Tscal>, rhodust_eps)                                    \
                                                                                                   \
    /* fields */                                                                                   \
    X_RO(shamrock::solvergraph::IFieldSpan<Tscal>, S)                                              \
    X_RO(shamrock::solvergraph::IFieldSpan<Tscal>, s_j)                                            \
                                                                                                   \
    /* outputs */                                                                                  \
    X_RW(shamrock::solvergraph::IFieldSpan<Tscal>, ds_j_dt)

namespace shammodels::sph::modules {

    template<class Tvec>
    class NodeMonofluidTVIAddSourceTerm : public shamrock::solvergraph::INode {

        using Tscal = shambase::VecComponent<Tvec>;

        u32 nbins;

        public:
        NodeMonofluidTVIAddSourceTerm(u32 nbins) : nbins(nbins) {}

        EXPAND_NODE_EDGES(NODE_EDGES)

        inline void _impl_evaluate_internal() {

            __shamrock_stack_entry();

            auto edges = get_edges();

            edges.S.check_sizes(edges.part_counts.indexes);
            edges.s_j.check_sizes(edges.part_counts.indexes);
            edges.ds_j_dt.check_sizes(edges.part_counts.indexes);

            auto rhodust_eps = edges.rhodust_eps.value;

            shambase::DistributedData<u32> counts = edges.part_counts.indexes.template map<u32>(
                [nbins = this->nbins](u64 /**/, u32 count) -> u32 {
                    return count * nbins;
                });

            sham::distributed_data_kernel_call(
                shamsys::instance::get_compute_scheduler_ptr(),
                sham::DDMultiRef{edges.S.get_spans(), edges.s_j.get_spans()},
                sham::DDMultiRef{edges.ds_j_dt.get_spans()},
                counts,
                [rhodust_eps](
                    u32 id,
                    const Tscal *__restrict S,
                    const Tscal *__restrict s_j,
                    Tscal *__restrict ds_j_dt) {
                    auto sj = s_j[id];

                    Tscal ds_j_dt_val = 0;

                    if (sj * sj > rhodust_eps) {
                        ds_j_dt_val = S[id] / (2 * sycl::sqrt(sj));
                    } else {
                        // we need this trick otherwise the bin would never start to get filled
                        // because of the cuttof, so the trick is to add the threshold in the
                        // denominator. yeah dirty i know i know  ...
                        ds_j_dt_val = S[id] / (2 * sycl::sqrt(sj + sycl::sqrt(rhodust_eps)));
                    }

                    ds_j_dt[id] += ds_j_dt_val;
                });
        }

        inline virtual std::string _impl_get_label() const {
            return "NodeMonofluidTVIAddSourceTerm";
        };

        inline virtual std::string _impl_get_tex() const { return "TODO"; };
    };
} // namespace shammodels::sph::modules

#undef NODE_EDGES
