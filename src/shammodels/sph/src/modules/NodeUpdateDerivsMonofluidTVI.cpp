// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright (c) 2021-2026 Timothée David--Cléris <tim.shamrock@proton.me>
// SPDX-License-Identifier: CeCILL Free Software License Agreement v2.1
// Shamrock is licensed under the CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

/**
 * @file NodeUpdateDerivsMonofluidTVI.cpp
 * @author Timothée David--Cléris (tim.shamrock@proton.me)
 * @brief
 *
 */

#include "shambase/string.hpp"
#include "shambackends/kernel_call_distrib.hpp"
#include "shammath/sphkernels.hpp"
#include "shammodels/sph/math/density.hpp"
#include "shammodels/sph/modules/NodeUpdateDerivsMonofluidTVI.hpp"
#include "shamrock/patch/PatchDataField.hpp" // IWYU pragma: keep

template<class Tvec, template<class> class SPHKernel>
struct KernelUpdateDerivsMonofluidTVI {
    using Tscal                   = shambase::VecComponent<Tvec>;
    using Kernel                  = SPHKernel<Tscal>;
    static constexpr Tscal hfactd = Kernel::hfactd;
    static constexpr Tscal Rkern  = Kernel::Rkern;
    static constexpr Tscal Rker2  = Rkern * Rkern;

    Tscal pmass;
    u32 ndust;

    inline void operator()(
        u32 thread_id,
        // input
        const Tvec *__restrict xyz,
        const Tscal *__restrict hpart,
        const Tvec *__restrict vxyz,
        const Tscal *__restrict omega,
        const Tscal *__restrict pressure,
        const Tscal *__restrict s_j,
        const Tscal *__restrict Ttilde_sj,
        shamrock::tree::ObjectCache::ptrs_read ploop_ptrs,
        // output
        Tscal *__restrict ds_j_dt) const {

        u32 id_a  = thread_id / ndust;
        u32 jdust = thread_id % ndust;

        Tscal h_a         = hpart[id_a];
        Tvec xyz_a        = xyz[id_a];
        Tvec vxyz_a       = vxyz[id_a];
        Tscal P_a         = pressure[id_a];
        Tscal omega_a     = omega[id_a];
        Tscal s_j_a       = s_j[thread_id];
        Tscal Ttilde_sj_a = Ttilde_sj[thread_id];

        using namespace shamrock::sph;
        Tscal rho_a             = rho_h(pmass, h_a, Kernel::hfactd);
        Tscal rho_a_sq          = rho_a * rho_a;
        Tscal rho_a_inv         = 1. / rho_a;
        Tscal omega_a_rho_a_inv = 1 / (omega_a * rho_a);

        Tscal term1 = 0;
        Tscal term2 = 0;

        shamrock::tree::ObjectCacheIterator particle_looper(ploop_ptrs);
        particle_looper.for_each_object(id_a, [&](u32 id_b) {
            Tvec dr    = xyz_a - xyz[id_b];
            Tscal rab2 = sycl::dot(dr, dr);
            Tscal h_b  = hpart[id_b];

            if (rab2 > h_a * h_a * Rker2 && rab2 > h_b * h_b * Rker2) {
                return;
            }

            Tvec vxyz_b       = vxyz[id_b];
            Tscal P_b         = pressure[id_b];
            Tscal omega_b     = omega[id_b];
            Tscal s_j_b       = s_j[id_b * ndust + jdust];
            Tscal Ttilde_sj_b = Ttilde_sj[id_b * ndust + jdust];

            Tscal rab         = sycl::sqrt(rab2);
            Tscal rab_inv_sat = sham::inv_sat_positive(rab);

            Tscal rho_b = rho_h(pmass, h_b, Kernel::hfactd);

            Tscal Fab_a = Kernel::dW_3d(rab, h_a);
            Tscal Fab_b = Kernel::dW_3d(rab, h_b);

            Tvec v_ab = vxyz_a - vxyz_b;

            Tvec r_ab_unit = dr * rab_inv_sat;

            Tscal F_ab_bar    = (Fab_a + Fab_b) / 2;
            Tscal delta_P     = P_a - P_b;
            Tscal Ts_weighted = (Ttilde_sj_a / rho_a + Ttilde_sj_b / rho_b);

            term1 += (pmass * s_j_b / rho_b) * Ts_weighted * delta_P * F_ab_bar * rab_inv_sat;
            term2 += pmass * sham::dot(v_ab, r_ab_unit * Fab_a);
        });

        // eq 51, Hutchison 2018
        Tscal ds_j_dt_a = Tscal{-0.5} * term1 + (s_j_a / (2 * rho_a * omega_a)) * term2;

        // if we dip in the negative range do not dip further
        ds_j_dt_a *= (s_j_a < 0 && ds_j_dt_a < 0) ? 0 : 1;

        // restore it slowly to 0
        ds_j_dt_a += (s_j_a < 0) ? -s_j_a / (10 * Ttilde_sj_a) : 0;

        ds_j_dt[thread_id] = ds_j_dt_a;
    }
};

template<class Tvec, template<class> class SPHKernel>
void shammodels::sph::modules::NodeUpdateDerivsMonofluidTVI<Tvec, SPHKernel>::
    _impl_evaluate_internal() {

    __shamrock_stack_entry();

    auto edges = get_edges();

    auto &part_counts_with_ghost = edges.part_counts_with_ghost.indexes;
    auto &part_counts            = edges.part_counts.indexes;

    // check that all input edges have the particles with ghosts zones
    edges.xyz.check_sizes(part_counts_with_ghost);
    edges.hpart.check_sizes(part_counts_with_ghost);
    edges.vxyz.check_sizes(part_counts_with_ghost);
    edges.omega.check_sizes(part_counts_with_ghost);
    edges.pressure.check_sizes(part_counts_with_ghost);
    edges.s_j.check_sizes(part_counts_with_ghost);
    edges.Ttilde_sj.check_sizes(part_counts_with_ghost);

    // ensure that the output edges are of size part_counts (output without ghosts zones)
    edges.ds_j_dt.ensure_sizes(part_counts);

    const Tscal pmass = edges.gpart_mass.value;

    using ComputeKernel = KernelUpdateDerivsMonofluidTVI<Tvec, SPHKernel>;

    auto total_specie_count = part_counts.template map<u32>([&](u64 id, u32 count) {
        return count * ndust;
    });

    // call the kernel for each patches with part_counts.get(id_patch) threads of patch id_patch
    sham::distributed_data_kernel_call(
        shamsys::instance::get_compute_scheduler_ptr(),
        sham::DDMultiRef{
            edges.xyz.get_spans(),
            edges.hpart.get_spans(),
            edges.vxyz.get_spans(),
            edges.omega.get_spans(),
            edges.pressure.get_spans(),
            edges.s_j.get_spans(),
            edges.Ttilde_sj.get_spans(),
            edges.neigh_cache},
        sham::DDMultiRef{edges.ds_j_dt.get_spans()},
        total_specie_count,
        ComputeKernel{pmass, ndust});
}

template<class Tvec, template<class> class SPHKernel>
std::string shammodels::sph::modules::NodeUpdateDerivsMonofluidTVI<Tvec, SPHKernel>::_impl_get_tex()
    const {
    auto gpart_mass  = get_ro_edge_base(0).get_tex_symbol();
    auto part_counts = get_ro_edge_base(1).get_tex_symbol();
    auto xyz         = get_ro_edge_base(3).get_tex_symbol();
    auto hpart       = get_ro_edge_base(4).get_tex_symbol();
    auto vxyz        = get_ro_edge_base(5).get_tex_symbol();
    auto omega       = get_ro_edge_base(6).get_tex_symbol();
    auto pressure    = get_ro_edge_base(7).get_tex_symbol();
    auto s_j         = get_ro_edge_base(8).get_tex_symbol();
    auto Ttilde_sj   = get_ro_edge_base(9).get_tex_symbol();
    auto ds_j_dt     = get_rw_edge_base(0).get_tex_symbol();

    const std::string ndust_str         = std::to_string(ndust);
    const std::string kernel_radius_str = std::to_string(kernel_radius);

    std::string tex = R"tex(
        NodeUpdateDerivsMonofluidTVI (eq. 51, Hutchison 2018)

        For gas particle $a$ and dust bin $j$:

        \begin{align}
        \rho_a &= \rho_h({gpart_mass}, {hpart}_a) \\
        \rho_b &= \rho_h({gpart_mass}, {hpart}_b) \\
        \mathbf{v}_{a,b} &= {vxyz}_a - {vxyz}_b \\
        \hat{\mathbf{r}}_{a,b} &= \frac{{xyz}_a - {xyz}_b}{\lvert {xyz}_a - {xyz}_b\rvert} \\
        F_{a,b}^a &= \mathrm{d}W_{3d}(\lvert \mathbf{r}_{a,b}\rvert, {hpart}_a) \\
        F_{a,b}^b &= \mathrm{d}W_{3d}(\lvert \mathbf{r}_{a,b}\rvert, {hpart}_b) \\
        \bar{F}_{a,b} &= \frac{F_{a,b}^a + F_{a,b}^b}{2} \\
        T^{\rm w}_{j,a,b} &= \frac{{Ttilde_sj}_{j,a}}{\rho_a} + \frac{{Ttilde_sj}_{j,b}}{\rho_b}
        \\
        {ds_j_dt}_{j,a} &=
            -\frac{1}{2}\sum_{b \in \mathcal{N}(a)}
            {gpart_mass}\;
            \frac{{s_j}_{j,b}}{\rho_b}\;
            T^{\rm w}_{j,a,b}\;
            ({pressure}_a - {pressure}_b)\;
            \bar{F}_{a,b}
        \\
        &\quad +
            \frac{{s_j}_{j,a}}{2\rho_a\,{omega}_a}
            \sum_{b \in \mathcal{N}(a)}
            {gpart_mass}\;
            \mathbf{v}_{a,b}\cdot\left(\hat{\mathbf{r}}_{a,b} F_{a,b}^a\right)
        \end{align}

        with the neighbor set $\mathcal{N}(a)$ defined by the kernel support:
        $\lvert \mathbf{r}_{a,b}\rvert \le \max({hpart}_a,{hpart}_b)\,{Rkern}$.

        $a \in [0,{part_counts})$, $j \in [0,{ndust})$.
    )tex";

    shambase::replace_all(tex, "{gpart_mass}", gpart_mass);
    shambase::replace_all(tex, "{part_counts}", part_counts);
    shambase::replace_all(tex, "{xyz}", xyz);
    shambase::replace_all(tex, "{hpart}", hpart);
    shambase::replace_all(tex, "{vxyz}", vxyz);
    shambase::replace_all(tex, "{omega}", omega);
    shambase::replace_all(tex, "{pressure}", pressure);
    shambase::replace_all(tex, "{s_j}", s_j);
    shambase::replace_all(tex, "{Ttilde_sj}", Ttilde_sj);
    shambase::replace_all(tex, "{ds_j_dt}", ds_j_dt);
    shambase::replace_all(tex, "{ndust}", ndust_str);
    shambase::replace_all(tex, "{Rkern}", kernel_radius_str);

    return tex;
}

using namespace shammath;
template class shammodels::sph::modules::NodeUpdateDerivsMonofluidTVI<f64_3, M4>;
template class shammodels::sph::modules::NodeUpdateDerivsMonofluidTVI<f64_3, M6>;
template class shammodels::sph::modules::NodeUpdateDerivsMonofluidTVI<f64_3, M8>;

template class shammodels::sph::modules::NodeUpdateDerivsMonofluidTVI<f64_3, C2>;
template class shammodels::sph::modules::NodeUpdateDerivsMonofluidTVI<f64_3, C4>;
template class shammodels::sph::modules::NodeUpdateDerivsMonofluidTVI<f64_3, C6>;
