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
 * @brief Module to add a source term on the dust density to the monofluid s_j derivative
 */

#include "shambase/string.hpp"
#include "shambackends/kernel_call_distrib.hpp"
#include "shambackends/vec.hpp"
#include "shamrock/solvergraph/IFieldSpan.hpp"
#include "shamrock/solvergraph/INode.hpp"
#include "shamrock/solvergraph/Indexes.hpp"
#include "shamrock/solvergraph/ScalarEdge.hpp"
#include "shamsys/NodeInstance.hpp"
#include <experimental/mdspan>

#define NODE_MONOFLUID_TVI_ADD_SOURCE_TERM_EDGES(X_RO, X_RW)                                       \
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

        EXPAND_NODE_EDGES(NODE_MONOFLUID_TVI_ADD_SOURCE_TERM_EDGES)

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
                    u32 id,                      // = part_id * nbins + jbin
                    const Tscal *__restrict S,   // [part_counts * nbins]
                    const Tscal *__restrict s_j, // [part_counts * nbins]
                    Tscal *__restrict ds_j_dt    // [part_counts * nbins]
                ) {
                    auto sj = s_j[id];

                    bool valid_div = sj * sj > rhodust_eps;

                    auto ds_j_dt_val = (valid_div) ? S[id] / (2 * sycl::sqrt(sj)) : 0;

                    ds_j_dt[id] += ds_j_dt_val;
                });
        }

        inline virtual std::string _impl_get_label() const {
            return "NodeMonofluidTVIAddSourceTerm";
        };

        inline virtual std::string _impl_get_tex() const {
            auto part_counts = get_ro_edge_base(0).get_tex_symbol();
            auto rhodust_eps = get_ro_edge_base(1).get_tex_symbol();
            auto S           = get_ro_edge_base(2).get_tex_symbol();
            auto s_j         = get_ro_edge_base(3).get_tex_symbol();
            auto ds_j_dt     = get_rw_edge_base(0).get_tex_symbol();

            std::string tex = R"tex(
            Monofluid TVI source term added to the dust surface-density evolution

            \begin{align}
            {ds_j_dt}_{j,a} &\mathrel{+}= \begin{cases}
                \dfrac{ {S}_{j,a} }{ 2 \sqrt{ {s_j}_{j,a} } }
                & \text{if } {s_j}_{j,a}^2 > \rho_{\rm eps} \\
                0 & \text{otherwise}
            \end{cases} \\
            a &\in [0, {part_counts}), \quad j \in [0, N_{\rm bins}) \\
            \rho_{\rm eps} &= {rhodust_eps}, \quad N_{\rm bins} = {nbins}
            \end{align}
            )tex";

            shambase::replace_all(tex, "{part_counts}", part_counts);
            shambase::replace_all(tex, "{rhodust_eps}", rhodust_eps);
            shambase::replace_all(tex, "{S}", S);
            shambase::replace_all(tex, "{s_j}", s_j);
            shambase::replace_all(tex, "{ds_j_dt}", ds_j_dt);
            shambase::replace_all(tex, "{nbins}", shambase::format("{}", nbins));

            return tex;
        };
    };
} // namespace shammodels::sph::modules

#undef NODE_MONOFLUID_TVI_ADD_SOURCE_TERM_EDGES
