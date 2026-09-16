// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright (c) 2021-2026 Timothée David--Cléris <tim.shamrock@proton.me>
// SPDX-License-Identifier: CeCILL Free Software License Agreement v2.1
// Shamrock is licensed under the CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

/**
 * @file EulerTimeDerivativeDust.cpp
 * @author Timothée David--Cléris (tim.shamrock@proton.me)
 * @brief
 *
 */

#include "shambase/string.hpp"
#include "shambackends/kernel_call_distrib.hpp"
#include "shambackends/math.hpp"
#include "shammodels/ramses/modules/EulerTimeDerivativeDust.hpp"
#include "shamrock/patch/PatchDataField.hpp"
#include "shamsys/NodeInstance.hpp"

namespace {

    template<class Tvec>
    struct KernelEulerTimeDerivativeDust {
        using Tscal = shambase::VecComponent<Tvec>;

        inline static void kernel(
            const shambase::DistributedData<shamrock::PatchDataFieldSpanPointer<Tscal>>
                &spans_rho_dust,
            const shambase::DistributedData<shamrock::PatchDataFieldSpanPointer<Tvec>>
                &spans_vel_dust,
            const shambase::DistributedData<shamrock::PatchDataFieldSpanPointer<Tvec>>
                &spans_grad_rho_dust,
            const shambase::DistributedData<shamrock::PatchDataFieldSpanPointer<Tvec>>
                &spans_dx_v_dust,
            const shambase::DistributedData<shamrock::PatchDataFieldSpanPointer<Tvec>>
                &spans_dy_v_dust,
            const shambase::DistributedData<shamrock::PatchDataFieldSpanPointer<Tvec>>
                &spans_dz_v_dust,

            shambase::DistributedData<shamrock::PatchDataFieldSpanPointer<Tscal>>
                &spans_dt_rho_dust,
            shambase::DistributedData<shamrock::PatchDataFieldSpanPointer<Tvec>> &spans_dt_vel_dust,
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
                sham::DDMultiRef{
                    spans_rho_dust,
                    spans_vel_dust,
                    spans_grad_rho_dust,
                    spans_dx_v_dust,
                    spans_dy_v_dust,
                    spans_dz_v_dust},
                sham::DDMultiRef{spans_dt_rho_dust, spans_dt_vel_dust},
                cell_counts,
                [](u32 i,
                   const Tscal *__restrict rho_dust,
                   const Tvec *__restrict vel_dust,
                   const Tvec *__restrict grad_rho_dust,
                   const Tvec *__restrict dx_v_dust,
                   const Tvec *__restrict dy_v_dust,
                   const Tvec *__restrict dz_v_dust,
                   Tscal *__restrict dt_rho_dust,
                   Tvec *__restrict dt_vel_dust) {
                    Tscal rho_dust_i     = rho_dust[i];
                    Tvec v_dust_i        = vel_dust[i];
                    Tvec grad_rho_dust_i = grad_rho_dust[i];
                    Tvec dx_v_dust_i     = dx_v_dust[i];
                    Tvec dy_v_dust_i     = dy_v_dust[i];
                    Tvec dz_v_dust_i     = dz_v_dust[i];

                    dt_rho_dust[i]
                        = -(sham::dot(v_dust_i, grad_rho_dust_i)
                            + rho_dust_i * (dx_v_dust_i[0] + dy_v_dust_i[1] + dz_v_dust_i[2]));

                    // Dust is pressureless, only the advection term remains
                    dt_vel_dust[i]
                        = -(v_dust_i[0] * dx_v_dust_i + v_dust_i[1] * dy_v_dust_i
                            + v_dust_i[2] * dz_v_dust_i);
                });
        }
    };

} // namespace

namespace shammodels::basegodunov::modules {

    template<class Tvec>
    void NodeEulerTimeDerivativeDust<Tvec>::_impl_evaluate_internal() {
        auto edges = get_edges();

        edges.spans_rho_dust.check_sizes(edges.sizes.indexes);
        edges.spans_vel_dust.check_sizes(edges.sizes.indexes);
        edges.spans_grad_rho_dust.check_sizes(edges.sizes.indexes);
        edges.spans_dx_v_dust.check_sizes(edges.sizes.indexes);
        edges.spans_dy_v_dust.check_sizes(edges.sizes.indexes);
        edges.spans_dz_v_dust.check_sizes(edges.sizes.indexes);

        edges.spans_dt_rho_dust.ensure_sizes(edges.sizes.indexes);
        edges.spans_dt_vel_dust.ensure_sizes(edges.sizes.indexes);

        KernelEulerTimeDerivativeDust<Tvec>::kernel(
            edges.spans_rho_dust.get_spans(),
            edges.spans_vel_dust.get_spans(),
            edges.spans_grad_rho_dust.get_spans(),
            edges.spans_dx_v_dust.get_spans(),
            edges.spans_dy_v_dust.get_spans(),
            edges.spans_dz_v_dust.get_spans(),
            edges.spans_dt_rho_dust.get_spans(),
            edges.spans_dt_vel_dust.get_spans(),
            edges.sizes.indexes,
            block_size,
            ndust);
    }

    template<class Tvec>
    std::string NodeEulerTimeDerivativeDust<Tvec>::_impl_get_tex() const {

        auto block_count   = get_ro_edge_base(0).get_tex_symbol();
        auto rho_dust      = get_ro_edge_base(1).get_tex_symbol();
        auto vel_dust      = get_ro_edge_base(2).get_tex_symbol();
        auto grad_rho_dust = get_ro_edge_base(3).get_tex_symbol();
        auto dx_v_dust     = get_ro_edge_base(4).get_tex_symbol();
        auto dy_v_dust     = get_ro_edge_base(5).get_tex_symbol();
        auto dz_v_dust     = get_ro_edge_base(6).get_tex_symbol();
        auto dt_rho_dust   = get_rw_edge_base(0).get_tex_symbol();
        auto dt_vel_dust   = get_rw_edge_base(1).get_tex_symbol();

        std::string tex = R"tex(
            Euler time derivatives of the dust primitive state (pressureless)

            \begin{align}
            {dt_rho_dust}_i &= - \left( {vel_dust}_i \cdot {grad_rho_dust}_i
                + {rho_dust}_i \left( {dx_v_dust}_{i,x} + {dy_v_dust}_{i,y}
                + {dz_v_dust}_{i,z} \right) \right) \\
            {dt_vel_dust}_i &= - \left( {vel_dust}_{i,x} {dx_v_dust}_i
                + {vel_dust}_{i,y} {dy_v_dust}_i + {vel_dust}_{i,z} {dz_v_dust}_i \right) \\
            i &\in [0,{block_count} * N_{\rm cell/block} * N_{\rm dust}) \\
            N_{\rm cell/block} & = {block_size} \\
            N_{\rm dust} & = {ndust}
            \end{align}
        )tex";

        shambase::replace_all(tex, "{dt_rho_dust}", dt_rho_dust);
        shambase::replace_all(tex, "{dt_vel_dust}", dt_vel_dust);
        shambase::replace_all(tex, "{grad_rho_dust}", grad_rho_dust);
        shambase::replace_all(tex, "{rho_dust}", rho_dust);
        shambase::replace_all(tex, "{vel_dust}", vel_dust);
        shambase::replace_all(tex, "{dx_v_dust}", dx_v_dust);
        shambase::replace_all(tex, "{dy_v_dust}", dy_v_dust);
        shambase::replace_all(tex, "{dz_v_dust}", dz_v_dust);
        shambase::replace_all(tex, "{block_count}", block_count);
        shambase::replace_all(tex, "{block_size}", sham::format("{}", block_size));
        shambase::replace_all(tex, "{ndust}", sham::format("{}", ndust));

        return tex;
    }

} // namespace shammodels::basegodunov::modules

template class shammodels::basegodunov::modules::NodeEulerTimeDerivativeDust<f64_3>;
