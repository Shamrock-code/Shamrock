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
 * @file ComputeDustTtilde.hpp
 * @author Timothée David--Cléris (tim.shamrock@proton.me)
 * @brief
 *
 */

#include "shambackends/kernel_call_distrib.hpp"
#include "shambackends/vec.hpp"
#include "shamcomm/logs.hpp"
#include "shammodels/sph/math/density.hpp"
#include "shamrock/solvergraph/IFieldSpan.hpp"
#include "shamrock/solvergraph/INode.hpp"
#include "shamrock/solvergraph/Indexes.hpp"
#include "shamrock/solvergraph/ScalarEdge.hpp"
#include "shamsys/NodeInstance.hpp"

#define NODE_COMPUTE_DUST_TTILDE_EDGES(X_RO, X_RW)                                                 \
    /* scalars */                                                                                  \
    X_RO(shamrock::solvergraph::ScalarEdge<Tscal>, gpart_mass)                                     \
                                                                                                   \
    /* counts */                                                                                   \
    X_RO(shamrock::solvergraph::Indexes<u32>, part_counts)                                         \
                                                                                                   \
    /* fields */                                                                                   \
    X_RO(shamrock::solvergraph::IFieldSpan<Tscal>, hpart)                                          \
    X_RO(shamrock::solvergraph::IFieldSpan<Tscal>, s_j)                                            \
    X_RO(shamrock::solvergraph::IFieldSpan<Tscal>, t_j)                                            \
                                                                                                   \
    /* outputs */                                                                                  \
    X_RW(shamrock::solvergraph::IFieldSpan<Tscal>, Ttilde_sj)

namespace shammodels::sph::modules {

    template<class Tvec, template<class> class SPHKernel>
    class ComputeDustTtilde : public shamrock::solvergraph::INode {

        using Tscal  = shambase::VecComponent<Tvec>;
        using Kernel = SPHKernel<Tscal>;

        static constexpr Tscal kernel_radius = SPHKernel<Tscal>::Rkern;

        u32 ndust;

        public:
        ComputeDustTtilde(u32 ndust) : ndust(ndust) {}

        EXPAND_NODE_EDGES(NODE_COMPUTE_DUST_TTILDE_EDGES)

        inline void _impl_evaluate_internal() {

            __shamrock_stack_entry();

            auto edges = get_edges();

            auto &part_counts = edges.part_counts.indexes;

            // check that all input edges have the particles with ghosts zones
            edges.hpart.check_sizes(part_counts);
            edges.s_j.check_sizes(part_counts);
            edges.t_j.check_sizes(part_counts);

            // ensure that the output edges are of size part_counts (output without ghosts zones)
            edges.Ttilde_sj.ensure_sizes(part_counts);

            const Tscal pmass = edges.gpart_mass.value;

            auto total_specie_count = part_counts.template map<u32>([&](u64 id, u32 count) {
                return count * ndust;
            });

            // call the kernel for each patches with part_counts.get(id_patch) threads of patch
            // id_patch
            sham::distributed_data_kernel_call(
                shamsys::instance::get_compute_scheduler_ptr(),
                sham::DDMultiRef{
                    edges.hpart.get_spans(), edges.s_j.get_spans(), edges.t_j.get_spans()},
                sham::DDMultiRef{edges.Ttilde_sj.get_spans()},
                total_specie_count,
                [pmass, ndust = ndust](
                    u32 thread_id,
                    const Tscal *__restrict hpart,
                    const Tscal *__restrict s_j,
                    const Tscal *__restrict t_j,
                    Tscal *__restrict Ttilde_sj) {
                    u32 id_a  = thread_id / ndust;
                    u32 jdust = thread_id % ndust;

                    Tscal h_a  = hpart[id_a];
                    Tscal sj_a = s_j[thread_id];
                    Tscal tj_a = t_j[thread_id];

                    using namespace shamrock::sph;
                    Tscal rho_a = rho_h(pmass, h_a, Kernel::hfactd);

                    auto epsilon = [&](Tscal sj) {
                        return sj * sj / rho_a;
                    };
                    /*
                     * Hutchison 2018 eq 15
                     * \tilde{T}_{sj} = \epsilon_j t_j - \sum_{k} \epsilon_k^2 t_k
                     */

                    Tscal eps_j_a     = epsilon(sj_a);
                    Tscal Ttilde_sj_a = eps_j_a * tj_a;

                    for (u32 k = 0; k < ndust; k++) {
                        Tscal sk_a = s_j[id_a * ndust + k];
                        Tscal tk_a = t_j[id_a * ndust + k];

                        Tscal eps_k_a = epsilon(sk_a);
                        Ttilde_sj_a -= eps_k_a * eps_k_a * tk_a;
                    }

                    // logger::raw_ln("Ttilde_sj_a", jdust, Ttilde_sj_a, eps_j_a, tj_a);

                    Ttilde_sj[thread_id] = Ttilde_sj_a;
                });
        }

        inline virtual std::string _impl_get_label() const { return "ComputeDustTtilde"; };

        inline virtual std::string _impl_get_tex() const { return "TODO"; };
    };
} // namespace shammodels::sph::modules
