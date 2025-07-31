// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright (c) 2021-2025 Timothée David--Cléris <tim.shamrock@proton.me>
// SPDX-License-Identifier: CeCILL Free Software License Agreement v2.1
// Shamrock is licensed under the CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

/**
 * @file ConsToPrimDust.cpp
 * @author Timothée David--Cléris (tim.shamrock@proton.me)
 * @brief
 *
 */

#include "shammodels/ramses/modules/ConsToPrimDust.hpp"
#include "shambackends/kernel_call_distrib.hpp"
#include "shammath/riemann_dust.hpp"
#include "shamrock/patch/PatchDataField.hpp"
#include "shamsys/NodeInstance.hpp"

namespace {

    template<class Tvec>
    struct KernelConsToPrimDust {
        using Tscal = shambase::VecComponent<Tvec>;

        inline static void kernel(
            const shambase::DistributedData<shamrock::PatchDataFieldSpanPointer<Tscal>>
                &spans_rho_dust,
            const shambase::DistributedData<shamrock::PatchDataFieldSpanPointer<Tvec>>
                &spans_rhov_dust,

            shambase::DistributedData<shamrock::PatchDataFieldSpanPointer<Tvec>> &spans_vel_dust,
            const shambase::DistributedData<u32> &sizes,
            u32 block_size,
            u32 ndust) {

            shambase::DistributedData<u32> cell_counts
                = sizes.map<u32>([&](u64 id, u32 block_count) {
                      u32 cell_count = block_count * block_size * ndust;
                      return cell_count;
                  });

            sham::distributed_data_kernel_call(
                shamsys::instance::get_compute_scheduler_ptr(),
                sham::DDMultiRef{spans_rho_dust, spans_rhov_dust},
                sham::DDMultiRef{spans_vel_dust},
                cell_counts,
                [](u32 i,
                   const Tscal *__restrict rho_dust,
                   const Tvec *__restrict rhov_dust,
                   Tvec *__restrict vel_dust) {
                    auto d_conststate = shammath::DustConsState<Tvec>{rho_dust[i], rhov_dust[i]};
                    auto d_prim_state = shammath::d_cons_to_prim(d_conststate);

                    vel_dust[i] = d_prim_state.vel;
                });
        }
    };

} // namespace

namespace shammodels::basegodunov::modules {

    template<class Tvec>
    void NodeConsToPrimDust<Tvec>::_impl_evaluate_internal() {
        auto edges = get_edges();

        edges.spans_rho_dust.check_sizes(edges.sizes.indexes);
        edges.spans_rhov_dust.check_sizes(edges.sizes.indexes);

        edges.spans_vel_dust.ensure_sizes(edges.sizes.indexes);

        KernelConsToPrimDust<Tvec>::kernel(
            edges.spans_rho_dust.get_spans(),
            edges.spans_rhov_dust.get_spans(),
            edges.spans_vel_dust.get_spans(),
            edges.sizes.indexes,
            block_size,
            ndust);
    }

    template<class Tvec>
    std::string NodeConsToPrimDust<Tvec>::_impl_get_tex() {

        auto block_count = get_ro_edge_base(0).get_tex_symbol();
        auto rho         = get_ro_edge_base(1).get_tex_symbol();
        auto rhov        = get_ro_edge_base(2).get_tex_symbol();
        auto vel         = get_rw_edge_base(0).get_tex_symbol();

        std::string tex = R"tex(
            Conservative to primitive variable (dust)

            \begin{align}
            {vel}_{i,j} &= \frac{ {rhov}_{i,j} }{ {rho}_{i,j} } \\
            i &\in [0,{block_count} * N_{\rm cell/block}) \\
            j &\in [0,n_{\rm dust}) \\
            n_{\rm dust} & = {ndust} \\
            N_{\rm cell/block} & = {block_size}
            \end{align}
        )tex";

        shambase::replace_all(tex, "{vel}", vel);
        shambase::replace_all(tex, "{rho}", rho);
        shambase::replace_all(tex, "{rhov}", rhov);
        shambase::replace_all(tex, "{block_count}", block_count);
        shambase::replace_all(tex, "{ndust}", shambase::format("{}", ndust));
        shambase::replace_all(tex, "{block_size}", shambase::format("{}", block_size));

        return tex;
    }

} // namespace shammodels::basegodunov::modules

template class shammodels::basegodunov::modules::NodeConsToPrimDust<f64_3>;
