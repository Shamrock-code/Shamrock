// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright (c) 2021-2026 Timothée David--Cléris <tim.shamrock@proton.me>
// SPDX-License-Identifier: CeCILL Free Software License Agreement v2.1
// Shamrock is licensed under the CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

/**
 * @file EulerTimeDerivativeGas.cpp
 * @author Timothée David--Cléris (tim.shamrock@proton.me)
 * @brief
 *
 */

#include "shambase/string.hpp"
#include "shambackends/kernel_call_distrib.hpp"
#include "shambackends/math.hpp"
#include "shammodels/ramses/modules/EulerTimeDerivativeGas.hpp"
#include "shamrock/patch/PatchDataField.hpp"
#include "shamsys/NodeInstance.hpp"

namespace {

    template<class Tvec>
    struct KernelEulerTimeDerivativeGas {
        using Tscal = shambase::VecComponent<Tvec>;

        inline static void kernel(
            const shambase::DistributedData<shamrock::PatchDataFieldSpanPointer<Tscal>> &spans_rho,
            const shambase::DistributedData<shamrock::PatchDataFieldSpanPointer<Tvec>> &spans_vel,
            const shambase::DistributedData<shamrock::PatchDataFieldSpanPointer<Tscal>>
                &spans_press,
            const shambase::DistributedData<shamrock::PatchDataFieldSpanPointer<Tvec>>
                &spans_grad_rho,
            const shambase::DistributedData<shamrock::PatchDataFieldSpanPointer<Tvec>> &spans_dx_v,
            const shambase::DistributedData<shamrock::PatchDataFieldSpanPointer<Tvec>> &spans_dy_v,
            const shambase::DistributedData<shamrock::PatchDataFieldSpanPointer<Tvec>> &spans_dz_v,
            const shambase::DistributedData<shamrock::PatchDataFieldSpanPointer<Tvec>>
                &spans_grad_P,

            shambase::DistributedData<shamrock::PatchDataFieldSpanPointer<Tscal>> &spans_dt_rho,
            shambase::DistributedData<shamrock::PatchDataFieldSpanPointer<Tvec>> &spans_dt_vel,
            shambase::DistributedData<shamrock::PatchDataFieldSpanPointer<Tscal>> &spans_dt_press,
            const shambase::DistributedData<u32> &sizes,
            u32 block_size,
            Tscal gamma) {

            shambase::DistributedData<u32> cell_counts
                = sizes.map<u32>([&](u64 id, u32 block_count) {
                      u32 cell_count = block_count * block_size;
                      return cell_count;
                  });

            sham::distributed_data_kernel_call(
                shamsys::instance::get_compute_scheduler_ptr(),
                sham::DDMultiRef{
                    spans_rho,
                    spans_vel,
                    spans_press,
                    spans_grad_rho,
                    spans_dx_v,
                    spans_dy_v,
                    spans_dz_v,
                    spans_grad_P},
                sham::DDMultiRef{spans_dt_rho, spans_dt_vel, spans_dt_press},
                cell_counts,
                [gamma](
                    u32 i,
                    const Tscal *__restrict rho,
                    const Tvec *__restrict vel,
                    const Tscal *__restrict press,
                    const Tvec *__restrict grad_rho,
                    const Tvec *__restrict dx_v,
                    const Tvec *__restrict dy_v,
                    const Tvec *__restrict dz_v,
                    const Tvec *__restrict grad_P,
                    Tscal *__restrict dt_rho,
                    Tvec *__restrict dt_vel,
                    Tscal *__restrict dt_press) {
                    Tscal rho_i     = rho[i];
                    Tvec v_i        = vel[i];
                    Tscal P_i       = press[i];
                    Tvec grad_rho_i = grad_rho[i];
                    Tvec dx_v_i     = dx_v[i];
                    Tvec dy_v_i     = dy_v[i];
                    Tvec dz_v_i     = dz_v[i];
                    Tvec grad_P_i   = grad_P[i];

                    dt_rho[i] = -(
                        sham::dot(v_i, grad_rho_i) + rho_i * (dx_v_i[0] + dy_v_i[1] + dz_v_i[2]));

                    dt_vel[i]
                        = -(v_i[0] * dx_v_i + v_i[1] * dy_v_i + v_i[2] * dz_v_i + grad_P_i / rho_i);

                    dt_press[i]
                        = -(gamma * P_i * (dx_v_i[0] + dy_v_i[1] + dz_v_i[2])
                            + sham::dot(v_i, grad_P_i));
                });
        }
    };

} // namespace

namespace shammodels::basegodunov::modules {

    template<class Tvec>
    void NodeEulerTimeDerivativeGas<Tvec>::_impl_evaluate_internal() {
        auto edges = get_edges();

        edges.spans_rho.check_sizes(edges.sizes.indexes);
        edges.spans_vel.check_sizes(edges.sizes.indexes);
        edges.spans_press.check_sizes(edges.sizes.indexes);
        edges.spans_grad_rho.check_sizes(edges.sizes.indexes);
        edges.spans_dx_v.check_sizes(edges.sizes.indexes);
        edges.spans_dy_v.check_sizes(edges.sizes.indexes);
        edges.spans_dz_v.check_sizes(edges.sizes.indexes);
        edges.spans_grad_P.check_sizes(edges.sizes.indexes);

        edges.spans_dt_rho.ensure_sizes(edges.sizes.indexes);
        edges.spans_dt_vel.ensure_sizes(edges.sizes.indexes);
        edges.spans_dt_press.ensure_sizes(edges.sizes.indexes);

        KernelEulerTimeDerivativeGas<Tvec>::kernel(
            edges.spans_rho.get_spans(),
            edges.spans_vel.get_spans(),
            edges.spans_press.get_spans(),
            edges.spans_grad_rho.get_spans(),
            edges.spans_dx_v.get_spans(),
            edges.spans_dy_v.get_spans(),
            edges.spans_dz_v.get_spans(),
            edges.spans_grad_P.get_spans(),
            edges.spans_dt_rho.get_spans(),
            edges.spans_dt_vel.get_spans(),
            edges.spans_dt_press.get_spans(),
            edges.sizes.indexes,
            block_size,
            gamma);
    }

    template<class Tvec>
    std::string NodeEulerTimeDerivativeGas<Tvec>::_impl_get_tex() const {

        auto block_count = get_ro_edge_base(0).get_tex_symbol();
        auto rho         = get_ro_edge_base(1).get_tex_symbol();
        auto vel         = get_ro_edge_base(2).get_tex_symbol();
        auto press       = get_ro_edge_base(3).get_tex_symbol();
        auto grad_rho    = get_ro_edge_base(4).get_tex_symbol();
        auto dx_v        = get_ro_edge_base(5).get_tex_symbol();
        auto dy_v        = get_ro_edge_base(6).get_tex_symbol();
        auto dz_v        = get_ro_edge_base(7).get_tex_symbol();
        auto grad_P      = get_ro_edge_base(8).get_tex_symbol();
        auto dt_rho      = get_rw_edge_base(0).get_tex_symbol();
        auto dt_vel      = get_rw_edge_base(1).get_tex_symbol();
        auto dt_press    = get_rw_edge_base(2).get_tex_symbol();

        std::string tex = R"tex(
            Euler time derivatives of the gas primitive state

            \begin{align}
            {dt_rho}_i &= - \left( {vel}_i \cdot {grad_rho}_i
                + {rho}_i \left( {dx_v}_{i,x} + {dy_v}_{i,y} + {dz_v}_{i,z} \right) \right) \\
            {dt_vel}_i &= - \left( {vel}_{i,x} {dx_v}_i + {vel}_{i,y} {dy_v}_i
                + {vel}_{i,z} {dz_v}_i + \frac{ {grad_P}_i }{ {rho}_i } \right) \\
            {dt_press}_i &= - \left( \gamma {press}_i
                \left( {dx_v}_{i,x} + {dy_v}_{i,y} + {dz_v}_{i,z} \right)
                + {vel}_i \cdot {grad_P}_i \right) \\
            i &\in [0,{block_count} * N_{\rm cell/block}) \\
            \gamma &= {gamma} \\
            N_{\rm cell/block} & = {block_size}
            \end{align}
        )tex";

        shambase::replace_all(tex, "{dt_rho}", dt_rho);
        shambase::replace_all(tex, "{dt_vel}", dt_vel);
        shambase::replace_all(tex, "{dt_press}", dt_press);
        shambase::replace_all(tex, "{rho}", rho);
        shambase::replace_all(tex, "{vel}", vel);
        shambase::replace_all(tex, "{press}", press);
        shambase::replace_all(tex, "{grad_rho}", grad_rho);
        shambase::replace_all(tex, "{dx_v}", dx_v);
        shambase::replace_all(tex, "{dy_v}", dy_v);
        shambase::replace_all(tex, "{dz_v}", dz_v);
        shambase::replace_all(tex, "{grad_P}", grad_P);
        shambase::replace_all(tex, "{block_count}", block_count);
        shambase::replace_all(tex, "{gamma}", sham::format("{}", gamma));
        shambase::replace_all(tex, "{block_size}", sham::format("{}", block_size));

        return tex;
    }

} // namespace shammodels::basegodunov::modules

template class shammodels::basegodunov::modules::NodeEulerTimeDerivativeGas<f64_3>;
