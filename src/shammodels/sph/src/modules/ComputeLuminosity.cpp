// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright (c) 2021-2026 Timothée David--Cléris <tim.shamrock@proton.me>
// SPDX-License-Identifier: CeCILL Free Software License Agreement v2.1
// Shamrock is licensed under the CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

/**
 * @file ComputeLuminosity.cpp
 * @author Timothée David--Cléris (tim.shamrock@proton.me)
 * @author Yona Lapeyre (yona.lapeyre@ens-lyon.fr)
 * @brief
 *
 */

#include "shambase/stacktrace.hpp"
#include "shambackends/kernel_call_distrib.hpp"
#include "shammodels/sph/SPHUtilities.hpp"
#include "shammodels/sph/math/forces.hpp"
#include "shammodels/sph/modules/ComputeLuminosity.hpp"
#include "shamrock/scheduler/SchedulerUtility.hpp"
#include "shamrock/solvergraph/IFieldSpan.hpp"

template<class Tvec, template<class> class SPHKernel>
void shammodels::sph::modules::NodeComputeLuminosity<Tvec, SPHKernel>::_impl_evaluate_internal() {

    __shamrock_stack_entry();

    auto edges = get_edges();

    auto dev_sched = shamsys::instance::get_compute_scheduler_ptr();

    edges.luminosity.ensure_sizes(edges.part_counts.indexes);

    sham::distributed_data_kernel_call(
        dev_sched,
        sham::DDMultiRef{
            edges.xyz.get_spans(),
            edges.hpart.get_spans(),
            edges.omega.get_spans(),
            edges.uint.get_spans(),
            edges.pressure.get_spans(),
            edges.neigh_cache.neigh_cache},
        sham::DDMultiRef{edges.luminosity.get_spans()},
        edges.part_counts.indexes,
        [part_mass = this->part_mass, alpha_u = this->alpha_u](
            u32 id_a,
            const Tvec *r,
            const Tscal *hpart,
            const Tscal *omega,
            const Tscal *uint,
            const Tscal *pressure,
            const auto ploop_ptrs,
            Tscal *luminosity) {
            shamrock::tree::ObjectCacheIterator particle_looper(ploop_ptrs);

            using namespace shamrock::sph;

            Tscal h_a               = hpart[id_a];
            Tvec xyz_a              = r[id_a];
            const Tscal u_a         = uint[id_a];
            const Tscal omega_a     = omega[id_a];
            const Tscal rho_a       = rho_h(part_mass, h_a, SPHKernel<Tscal>::hfactd);
            const Tscal P_a         = pressure[id_a];
            Tscal omega_a_rho_a_inv = 1 / (omega_a * rho_a);
            Tscal tmp_luminosity    = 0;

            particle_looper.for_each_object(id_a, [&](u32 id_b) {
                const Tscal u_b     = uint[id_b];
                const Tscal h_b     = hpart[id_b];
                const Tscal omega_b = omega[id_b];
                const Tscal P_b     = pressure[id_b];
                const Tscal rho_b   = rho_h(part_mass, h_b, SPHKernel<Tscal>::hfactd);
                Tvec dr             = xyz_a - r[id_b];
                Tscal rab2          = sycl::dot(dr, dr);
                Tscal rab           = sycl::sqrt(rab2);

                Tscal vsigu = vsig_u(P_a, P_b, rho_a, rho_b);
                Tscal Fab_a = SPHKernel<Tscal>::dW_3d(rab, h_a);
                Tscal Fab_b = SPHKernel<Tscal>::dW_3d(rab, h_b);

                tmp_luminosity += lambda_shock_conductivity(
                    part_mass,
                    alpha_u,
                    vsigu,
                    u_a - u_b,
                    Fab_a * omega_a_rho_a_inv,
                    Fab_b / (rho_b * omega_b));
            });

            luminosity[id_a] = tmp_luminosity;
        });
}

template<class Tvec, template<class> class SPHKernel>
std::string shammodels::sph::modules::NodeComputeLuminosity<Tvec, SPHKernel>::_impl_get_tex()
    const {
    auto xyz        = get_ro_edge_base(0).get_tex_symbol();
    auto hpart      = get_ro_edge_base(1).get_tex_symbol();
    auto omega      = get_ro_edge_base(2).get_tex_symbol();
    auto uint       = get_ro_edge_base(3).get_tex_symbol();
    auto pressure   = get_ro_edge_base(4).get_tex_symbol();
    auto luminosity = get_rw_edge_base(0).get_tex_symbol();

    std::string tex = R"tex(
        Compute idealized luminosity from shocks only: energy dissipated immidiately emitted
        \begin{align}
                 {luminosity}_i &= pmass * alpha_u * vsigu * u_ab * \frac{1}{2}
               * (Fab_inv_omega_a_rho_a + Fab_inv_omega_b_rho_b)\\
               &= pmass * alpha_u * \sqrt(\frac{\abs({pressure}_a - {pressure}_b)}{2 (rho_a + rho_b)}); * ({uint}_a - {uint}_b) * \frac{1}{2}
               * (\frac{Fab_a}{{omega}_a * rho_a} + \frac{Fab_b}{{omega}_b * rho_b})
        \end{align}
        )tex";

    shambase::replace_all(tex, "{xyz}", xyz);
    shambase::replace_all(tex, "{hpart}", hpart);
    shambase::replace_all(tex, "{omega}", omega);
    shambase::replace_all(tex, "{uint}", uint);
    shambase::replace_all(tex, "{pressure}", pressure);
    shambase::replace_all(tex, "{luminosity}", luminosity);
    return tex;
}

using namespace shammath;
template class shammodels::sph::modules::NodeComputeLuminosity<f64_3, M4>;
template class shammodels::sph::modules::NodeComputeLuminosity<f64_3, M6>;
template class shammodels::sph::modules::NodeComputeLuminosity<f64_3, M8>;

template class shammodels::sph::modules::NodeComputeLuminosity<f64_3, C2>;
template class shammodels::sph::modules::NodeComputeLuminosity<f64_3, C4>;
template class shammodels::sph::modules::NodeComputeLuminosity<f64_3, C6>;
