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
 * @file SetDustStoppingTimeEpstein.hpp
 * @author Timothée David--Cléris (tim.shamrock@proton.me)
 * @brief
 *
 */

#include "shambase/string.hpp"
#include "shambackends/DeviceBuffer.hpp"
#include "shambackends/kernel_call_distrib.hpp"
#include "shambackends/vec.hpp"
#include "shammodels/sph/math/density.hpp"
#include "shamphys/Dust.hpp"
#include "shamrock/solvergraph/IFieldSpan.hpp"
#include "shamrock/solvergraph/INode.hpp"
#include "shamrock/solvergraph/Indexes.hpp"
#include "shamrock/solvergraph/ScalarEdge.hpp"
#include "shamsys/NodeInstance.hpp"
#include <vector>

#define NODE_EDGES(X_RO, X_RW)                                                                     \
    /* scalars */                                                                                  \
    X_RO(shamrock::solvergraph::ScalarEdge<Tscal>, gpart_mass)                                     \
    X_RO(shamrock::solvergraph::ScalarEdge<Tscal>, gamma)                                          \
    X_RO(shamrock::solvergraph::ScalarEdge<std::vector<Tscal>>, sgrain_j)                          \
    X_RO(shamrock::solvergraph::ScalarEdge<std::vector<Tscal>>, rho_grain_j)                       \
                                                                                                   \
    /* counts */                                                                                   \
    X_RO(shamrock::solvergraph::Indexes<u32>, part_counts)                                         \
                                                                                                   \
    /* fields */                                                                                   \
    X_RO(shamrock::solvergraph::IFieldSpan<Tscal>, hpart)                                          \
    X_RO(shamrock::solvergraph::IFieldSpan<Tscal>, cs)                                             \
                                                                                                   \
    X_RW(shamrock::solvergraph::IFieldSpan<Tscal>, t_j)

namespace shammodels::sph::modules {

    template<class Tvec, template<class> class SPHKernel>
    class SetDustStoppingTimeEpstein : public shamrock::solvergraph::INode {

        using Tscal  = shambase::VecComponent<Tvec>;
        using Kernel = SPHKernel<Tscal>;

        u32 ndust;
        std::unique_ptr<sham::DeviceBuffer<Tscal>> sgrain_j;
        std::unique_ptr<sham::DeviceBuffer<Tscal>> rho_grain_j;

        public:
        SetDustStoppingTimeEpstein(u32 ndust) : ndust(ndust) {}

        EXPAND_NODE_EDGES(NODE_EDGES)

        inline void _impl_evaluate_internal() {

            __shamrock_stack_entry();

            auto edges = get_edges();

            auto &part_counts                            = edges.part_counts.indexes;
            const std::vector<Tscal> &inputs_sgrain_j    = edges.sgrain_j.value;
            const std::vector<Tscal> &inputs_rho_grain_j = edges.rho_grain_j.value;
            SHAM_ASSERT(inputs_sgrain_j.size() == ndust);
            SHAM_ASSERT(inputs_rho_grain_j.size() == ndust);

            // ensure that the output edges are of size part_counts
            edges.t_j.ensure_sizes(part_counts);

            auto dev_sched = shamsys::instance::get_compute_scheduler_ptr();

            if (!sgrain_j) {
                sgrain_j = std::make_unique<sham::DeviceBuffer<Tscal>>(ndust, dev_sched);
            }
            if (!rho_grain_j) {
                rho_grain_j = std::make_unique<sham::DeviceBuffer<Tscal>>(ndust, dev_sched);
            }

            sgrain_j->resize(ndust);
            rho_grain_j->resize(ndust);
            sgrain_j->copy_from_stdvec(inputs_sgrain_j);
            rho_grain_j->copy_from_stdvec(inputs_rho_grain_j);

            auto &q = shamsys::instance::get_compute_scheduler().get_queue();

            const Tscal pmass = edges.gpart_mass.value;
            const Tscal gamma = edges.gamma.value;

            part_counts.for_each([&](u64 id, u32 count) {
                // call the kernel for each patches with part_counts.get(id_patch) threads of patch
                // id_patch
                sham::kernel_call(
                    q,
                    sham::MultiRef{
                        *sgrain_j,
                        *rho_grain_j,
                        edges.hpart.get_spans().get(id),
                        edges.cs.get_spans().get(id)},
                    sham::MultiRef{edges.t_j.get_spans().get(id)},
                    count * ndust,
                    [ndust = ndust, pmass, gamma](
                        u32 thread_id,
                        const Tscal *__restrict sgrain_j,
                        const Tscal *__restrict rho_grain_j,
                        const Tscal *__restrict hpart,
                        const Tscal *__restrict cs,
                        Tscal *__restrict t_j) {
                        u32 jdust = thread_id % ndust;
                        u32 id_a  = thread_id / ndust;

                        Tscal sgrain    = sgrain_j[jdust];
                        Tscal rho_grain = rho_grain_j[jdust];

                        Tscal h_a  = hpart[id_a];
                        Tscal cs_a = cs[id_a];

                        using namespace shamrock::sph;
                        Tscal rho_a = rho_h(pmass, h_a, Kernel::hfactd);

                        auto ts = shamphys::epstein_stopping_time(
                            rho_grain, sgrain, rho_a, cs_a, gamma);

                        t_j[thread_id] = ts;
                    });
            });
        }

        inline virtual std::string _impl_get_label() const { return "SetDustStoppingTimeEpstein"; };

        inline virtual std::string _impl_get_tex() const {

            auto gpart_mass  = get_ro_edge_base(0).get_tex_symbol();
            auto gamma       = get_ro_edge_base(1).get_tex_symbol();
            auto sgrain_j    = get_ro_edge_base(2).get_tex_symbol();
            auto rho_grain_j = get_ro_edge_base(3).get_tex_symbol();
            auto part_counts = get_ro_edge_base(4).get_tex_symbol();
            auto hpart       = get_ro_edge_base(5).get_tex_symbol();
            auto cs          = get_ro_edge_base(6).get_tex_symbol();
            auto t_j         = get_rw_edge_base(0).get_tex_symbol();

            std::string tex = R"tex(
                SetDustStoppingTimeEpstein (PHANTOM eq.~250, subsonic)

                \begin{align}
                \rho_i &= {gpart_mass} \left( \frac{h_{\rm fact}}{ {hpart}_i } \right)^3 \\
                {t_j}_{i,j} &= \frac{ {rho_grain_j}_j \, {sgrain_j}_j }{ \rho_i \, {cs}_i }
                    \sqrt{\frac{\pi \, {gamma}}{8}} \\
                i &\in [0,{part_counts}) \\
                j &\in [0,{ndust}) \\
                h_{\rm fact} &= {hfact}
                \end{align}
            )tex";

            shambase::replace_all(tex, "{gpart_mass}", gpart_mass);
            shambase::replace_all(tex, "{gamma}", gamma);
            shambase::replace_all(tex, "{sgrain_j}", sgrain_j);
            shambase::replace_all(tex, "{rho_grain_j}", rho_grain_j);
            shambase::replace_all(tex, "{part_counts}", part_counts);
            shambase::replace_all(tex, "{ndust}", shambase::format("{}", ndust));
            shambase::replace_all(tex, "{hpart}", hpart);
            shambase::replace_all(tex, "{cs}", cs);
            shambase::replace_all(tex, "{t_j}", t_j);
            shambase::replace_all(tex, "{hfact}", shambase::format("{}", Kernel::hfactd));

            return tex;
        };
    };
} // namespace shammodels::sph::modules

#undef NODE_EDGES
