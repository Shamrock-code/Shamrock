// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright (c) 2021-2026 Timothée David--Cléris <tim.shamrock@proton.me>
// SPDX-License-Identifier: CeCILL Free Software License Agreement v2.1
// Shamrock is licensed under the CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

/**
 * @file NodeUpdateDerivsMonofluidTVA.cpp
 * @author Timothée David--Cléris (tim.shamrock@proton.me)
 * @brief
 *
 */

#include "shambase/string.hpp"
#include "shambackends/kernel_call_distrib.hpp"
#include "shammath/sphkernels.hpp"
#include "shammodels/sph/math/density.hpp"
#include "shammodels/sph/modules/NodeUpdateDerivsMonofluidTVA.hpp"
#include "shamrock/patch/PatchDataField.hpp" // IWYU pragma: keep
#include <type_traits>
#include <cstddef>

/**
 * @brief TVA ds_j/dt kernel body (eq. 51, Hutchison 2018), shared by the plain and the pair gated
 * variants.
 *
 * The flux term of the pair (a,b) is multiplied by `pair_weight(s_j_a, s_j_b, g_ab)`, with
 * g_ab = Tw_ab (P_a - P_b) Fbar_ab / r_ab (antisymmetric, g_ba = -g_ab bitwise). Since the dust
 * mass flux of the pair is antisymmetric in (a,b), any weight symmetric in (a,b) keeps the dust
 * mass conservation exact. The compression term (s_a / 2 rho_a) drho_a/dt is not a flux (it cancels
 * in d(m s^2/rho)/dt) and is left untouched.
 */
template<class Tvec, template<class> class SPHKernel, class PairWeight>
inline shambase::VecComponent<Tvec> compute_ds_j_dt_monofluid_tva(
    shambase::VecComponent<Tvec> pmass,
    u32 ndust,
    u32 id_a,
    u32 jdust,
    const Tvec *__restrict xyz,
    const shambase::VecComponent<Tvec> *__restrict hpart,
    const Tvec *__restrict vxyz,
    const shambase::VecComponent<Tvec> *__restrict omega,
    const shambase::VecComponent<Tvec> *__restrict pressure,
    const shambase::VecComponent<Tvec> *__restrict s_j,
    const shambase::VecComponent<Tvec> *__restrict Ttilde_sj,
    shamrock::tree::ObjectCache::ptrs_read ploop_ptrs,
    PairWeight &&pair_weight) {

    using Tscal                  = shambase::VecComponent<Tvec>;
    using Kernel                 = SPHKernel<Tscal>;
    static constexpr Tscal Rkern = Kernel::Rkern;
    static constexpr Tscal Rker2 = Rkern * Rkern;

    u32 thread_id = id_a * ndust + jdust;

    Tscal h_a         = hpart[id_a];
    Tvec xyz_a        = xyz[id_a];
    Tvec vxyz_a       = vxyz[id_a];
    Tscal P_a         = pressure[id_a];
    Tscal omega_a     = omega[id_a];
    Tscal s_j_a       = s_j[thread_id];
    Tscal Ttilde_sj_a = Ttilde_sj[thread_id];

    using namespace shamrock::sph;
    Tscal rho_a = rho_h(pmass, h_a, Kernel::hfactd);

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

        Tscal g_ab = Ts_weighted * delta_P * F_ab_bar * rab_inv_sat;

        if constexpr (std::is_same_v<std::decay_t<PairWeight>, std::nullptr_t>) {
            term1 += (pmass * s_j_b / rho_b) * g_ab;
        } else {
            term1 += pair_weight(s_j_a, s_j_b, g_ab) * (pmass * s_j_b / rho_b) * g_ab;
        }
        term2 += pmass * sham::dot(v_ab, r_ab_unit * Fab_a);
    });

    // eq 51, Hutchison 2018
    return Tscal{-0.5} * term1 + (s_j_a / (2 * rho_a * omega_a)) * term2;
}

/**
 * @brief Symmetrized per-pair gate of the smooth positivity limiter.
 *
 * A particle with s_j < 0 does not accept a flux contribution that decreases s_j, and the pair is
 * switched off if either member refuses it:
 *   blocked = (s_a < 0 && f_ab > 0) || (s_b < 0 && f_ba > 0)
 * with f_ab = (m s_b / rho_b) g_ab and f_ba = -(m s_a / rho_a) g_ab (ds_j/dt = -f/2). Only signs
 * are used, so both members of the pair take bitwise the same decision (g_ba = -g_ab exactly).
 */
template<class Tscal>
inline Tscal monofluid_tva_pair_gate(Tscal s_a, Tscal s_b, Tscal g_ab) {
    bool f_ab_pos = (s_b > 0 && g_ab > 0) || (s_b < 0 && g_ab < 0);
    bool f_ba_pos = (s_a > 0 && g_ab < 0) || (s_a < 0 && g_ab > 0);
    bool blocked  = (s_a < 0 && f_ab_pos) || (s_b < 0 && f_ba_pos);
    return blocked ? Tscal{0} : Tscal{1};
}

template<class Tvec, template<class> class SPHKernel, bool pair_gate>
struct KernelUpdateDerivsMonofluidTVA {
    using Tscal = shambase::VecComponent<Tvec>;

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

        auto compute = [&](auto &&pair_weight) {
            return compute_ds_j_dt_monofluid_tva<Tvec, SPHKernel>(
                pmass,
                ndust,
                id_a,
                jdust,
                xyz,
                hpart,
                vxyz,
                omega,
                pressure,
                s_j,
                Ttilde_sj,
                ploop_ptrs,
                pair_weight);
        };

        if constexpr (pair_gate) {
            ds_j_dt[thread_id] = compute([](Tscal s_a, Tscal s_b, Tscal g_ab) {
                return monofluid_tva_pair_gate(s_a, s_b, g_ab);
            });
        } else {
            ds_j_dt[thread_id] = compute(nullptr);
        }
    }
};

template<class Tvec, template<class> class SPHKernel>
void shammodels::sph::modules::NodeUpdateDerivsMonofluidTVA<Tvec, SPHKernel>::
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

    const Tscal pmass = edges.gpart_mass.data;

    auto total_specie_count = part_counts.template map<u32>([&](u64 id, u32 count) {
        return count * ndust;
    });

    auto launch = [&](auto kernel) {
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
            kernel);
    };

    if (pair_gate) {
        launch(KernelUpdateDerivsMonofluidTVA<Tvec, SPHKernel, true>{pmass, ndust});
    } else {
        launch(KernelUpdateDerivsMonofluidTVA<Tvec, SPHKernel, false>{pmass, ndust});
    }
}

template<class Tvec, template<class> class SPHKernel>
std::string shammodels::sph::modules::NodeUpdateDerivsMonofluidTVA<Tvec, SPHKernel>::_impl_get_tex()
    const {
    const std::string ndust_str         = std::to_string(ndust);
    const std::string kernel_radius_str = std::to_string(kernel_radius);

    std::string tex = R"tex(
        NodeUpdateDerivsMonofluidTVA (eq. 51, Hutchison 2018)

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
            {gate_symbol}{gpart_mass}\;
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

        {gate_def}

        $a \in [0,{part_counts})$, $j \in [0,{ndust})$.
    )tex";

    const std::string gate_symbol = pair_gate ? R"tex(G_{j,a,b}\;)tex" : "";
    const std::string gate_def    = pair_gate ? R"tex(
        Symmetrized per-pair positivity gate, with
        $g_{j,a,b} = T^{\rm w}_{j,a,b} ({pressure}_a - {pressure}_b) \bar{F}_{a,b} / \lvert \mathbf{r}_{a,b}\rvert$:
        a particle with ${s_j} < 0$ refuses a flux contribution that decreases its ${s_j}$, and the
        pair is switched off if either member refuses it:
        \begin{align}
        G_{j,a,b} &= \begin{cases}
            0 & ({s_j}_{j,a} < 0 \land {s_j}_{j,b} g_{j,a,b} > 0)
                \lor ({s_j}_{j,b} < 0 \land {s_j}_{j,a} g_{j,a,b} < 0) \\
            1 & \text{otherwise}
        \end{cases}
        \end{align}
    )tex"
                                              : "";

    shambase::replace_all(tex, "{gate_symbol}", gate_symbol);
    shambase::replace_all(tex, "{gate_def}", gate_def);

    replace_edges_tex_symbols(tex);

    shambase::replace_all(tex, "{ndust}", ndust_str);
    shambase::replace_all(tex, "{Rkern}", kernel_radius_str);

    return tex;
}

using namespace shammath;
template class shammodels::sph::modules::NodeUpdateDerivsMonofluidTVA<f64_3, M4>;
template class shammodels::sph::modules::NodeUpdateDerivsMonofluidTVA<f64_3, M6>;
template class shammodels::sph::modules::NodeUpdateDerivsMonofluidTVA<f64_3, M8>;

template class shammodels::sph::modules::NodeUpdateDerivsMonofluidTVA<f64_3, C2>;
template class shammodels::sph::modules::NodeUpdateDerivsMonofluidTVA<f64_3, C4>;
template class shammodels::sph::modules::NodeUpdateDerivsMonofluidTVA<f64_3, C6>;
