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
 * @file MonoFluidTVIDeltav.hpp
 * @author Timothée David--Cléris (tim.shamrock@proton.me)
 * @brief
 *
 */

#include "shambackends/kernel_call_distrib.hpp"
#include "shambackends/vec.hpp"
#include "shammodels/sph/math/density.hpp"
#include "shamrock/solvergraph/IFieldSpan.hpp"
#include "shamrock/solvergraph/INode.hpp"
#include "shamrock/solvergraph/Indexes.hpp"
#include "shamrock/solvergraph/ScalarEdge.hpp"
#include "shamsys/NodeInstance.hpp"

#define MONO_FLUID_TVI_DELTAV_EDGES(X_RO, X_RW)                                                    \
    /* scalars */                                                                                  \
    X_RO(shamrock::solvergraph::ScalarEdge<Tscal>, gpart_mass)                                     \
                                                                                                   \
    /* counts */                                                                                   \
    X_RO(shamrock::solvergraph::Indexes<u32>, part_counts)                                         \
                                                                                                   \
    /* fields */                                                                                   \
    X_RO(shamrock::solvergraph::IFieldSpan<Tscal>, hpart)                                          \
    X_RO(shamrock::solvergraph::IFieldSpan<Tvec>, grad_pressure)                                   \
    X_RO(shamrock::solvergraph::IFieldSpan<Tscal>, s_j)                                            \
    X_RO(shamrock::solvergraph::IFieldSpan<Tscal>, t_j)                                            \
                                                                                                   \
    /* outputs */                                                                                  \
    X_RW(shamrock::solvergraph::IFieldSpan<Tvec>, delta_v)

namespace shammodels::sph::modules {

    template<class Tvec, template<class> class SPHKernel>
    class MonoFluidTVIDeltav : public shamrock::solvergraph::INode {

        using Tscal  = shambase::VecComponent<Tvec>;
        using Kernel = SPHKernel<Tscal>;

        u32 ndust;

        public:
        MonoFluidTVIDeltav(u32 ndust) : ndust(ndust) {}

        EXPAND_NODE_EDGES(MONO_FLUID_TVI_DELTAV_EDGES)

        inline void _impl_evaluate_internal() {

            __shamrock_stack_entry();

            auto edges = get_edges();

            auto &part_counts = edges.part_counts.indexes;

            // check that all input edges have the particles with ghosts zones
            edges.hpart.check_sizes(part_counts);
            edges.grad_pressure.check_sizes(part_counts);
            edges.s_j.check_sizes(part_counts);
            edges.t_j.check_sizes(part_counts);

            // ensure that the output edges are of size part_counts (output without ghosts zones)
            edges.delta_v.ensure_sizes(part_counts);

            const Tscal pmass = edges.gpart_mass.value;

            auto total_specie_count = part_counts.template map<u32>([&](u64 id, u32 count) {
                return count * ndust;
            });

            // call the kernel for each patches with part_counts.get(id_patch) threads of patch
            // id_patch
            sham::distributed_data_kernel_call(
                shamsys::instance::get_compute_scheduler_ptr(),
                sham::DDMultiRef{
                    edges.hpart.get_spans(),
                    edges.grad_pressure.get_spans(),
                    edges.s_j.get_spans(),
                    edges.t_j.get_spans()},
                sham::DDMultiRef{edges.delta_v.get_spans()},
                total_specie_count,
                [pmass, ndust = ndust](
                    u32 thread_id,
                    const Tscal *__restrict hpart,
                    const Tvec *__restrict grad_pressure,
                    const Tscal *__restrict s_j,
                    const Tscal *__restrict t_j,
                    Tvec *__restrict delta_v) {
                    u32 id_a  = thread_id / ndust;
                    u32 jdust = thread_id % ndust;

                    Tscal h_a     = hpart[id_a];
                    Tvec grad_P_a = grad_pressure[id_a];
                    Tscal sj_a    = s_j[thread_id];
                    Tscal tj_a    = t_j[thread_id];

                    using namespace shamrock::sph;
                    Tscal rho_a = rho_h(pmass, h_a, Kernel::hfactd);

                    auto epsilon = [&](Tscal sj) {
                        return sj * sj / rho_a;
                    };

                    /*
                     * Hutchison 2018 eq 15
                     * T_{sj} = \epsilon_j (1 - \epsilon_j) t_j
                     * delta_v = \epsilon_j t_j \nabla P / rho = T_{sj} \nabla P / rho_g
                     */

                    Tscal eps_j_a = epsilon(sj_a);

                    delta_v[id_a] = (eps_j_a * tj_a / rho_a) * grad_P_a;
                });
        }

        inline virtual std::string _impl_get_label() const { return "MonoFluidTVIDeltav"; };

        inline virtual std::string _impl_get_tex() const {

            auto gpart_mass    = get_ro_edge_base(0).get_tex_symbol();
            auto part_counts   = get_ro_edge_base(1).get_tex_symbol();
            auto hpart         = get_ro_edge_base(2).get_tex_symbol();
            auto grad_pressure = get_ro_edge_base(3).get_tex_symbol();
            auto s_j           = get_ro_edge_base(4).get_tex_symbol();
            auto t_j           = get_ro_edge_base(5).get_tex_symbol();
            auto delta_v       = get_rw_edge_base(0).get_tex_symbol();

            std::string tex = R"tex(
                MonoFluidTVIDeltav

                \begin{align}
                \epsilon_{i,j} = \frac{{s_j}_{i,j}^2}{{rho}_i ({hpart}_i)} \\
                {delta_v}_{i,j} = \epsilon_{i,j} {t_j}_{i,j} {grad_pressure}_i / {rho}_i \\
                i \in [0,{part_counts}] \\
                j \in [0,{ndust}]
                \end{align}
            )tex";

            shambase::replace_all(tex, "{gpart_mass}", gpart_mass);
            shambase::replace_all(tex, "{part_counts}", part_counts);
            shambase::replace_all(tex, "{hpart}", hpart);
            shambase::replace_all(tex, "{grad_pressure}", grad_pressure);
            shambase::replace_all(tex, "{s_j}", s_j);
            shambase::replace_all(tex, "{t_j}", t_j);
            shambase::replace_all(tex, "{delta_v}", delta_v);

            return tex;
        };
    };
} // namespace shammodels::sph::modules
