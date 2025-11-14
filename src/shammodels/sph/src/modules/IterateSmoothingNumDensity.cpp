// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright (c) 2021-2025 Timothée David--Cléris <tim.shamrock@proton.me>
// SPDX-License-Identifier: CeCILL Free Software License Agreement v2.1
// Shamrock is licensed under the CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

/**
 * @file IterateSmoothingNumDensity.cpp
 * @author Timothée David--Cléris (tim.shamrock@proton.me)
 * @brief Implements the IterateSmoothingLengthDensity module for iterating smoothing length based
 * on the SPH density sum.
 */

#include "shambase/stacktrace.hpp"
#include "shambackends/kernel_call_distrib.hpp"
#include "shamcomm/logs.hpp"
#include "shammath/sphkernels.hpp"
#include "shammodels/sph/math/density.hpp"
#include "shammodels/sph/modules/IterateSmoothingNumDensity.hpp"
#include "shamrock/patch/PatchDataField.hpp"

using namespace shammodels::sph::modules;

template<class Tvec, class SPHKernel>
void IterateSmoothingNumDensity<Tvec, SPHKernel>::_impl_evaluate_internal() {
    StackEntry stack_loc{};

    auto edges = get_edges();

    auto &thread_counts = edges.sizes.indexes;

    edges.neigh_cache.check_sizes(thread_counts);
    edges.positions.check_sizes(thread_counts);
    edges.old_h.check_sizes(thread_counts);
    edges.new_h.ensure_sizes(thread_counts);
    edges.eps_h.ensure_sizes(thread_counts);
    edges.was_limited.ensure_sizes(thread_counts);

    auto &neigh_cache = edges.neigh_cache.neigh_cache;
    auto &positions   = edges.positions.get_spans();
    auto &old_h       = edges.old_h.get_spans();
    auto &new_h       = edges.new_h.get_spans();
    auto &eps_h       = edges.eps_h.get_spans();
    auto &was_limited = edges.was_limited.get_spans();

    auto dev_sched = shamsys::instance::get_compute_scheduler_ptr();

    static constexpr Tscal Rkern = SPHKernel::Rkern;

    sham::distributed_data_kernel_call(
        dev_sched,
        sham::DDMultiRef{neigh_cache, positions, old_h},
        sham::DDMultiRef{new_h, eps_h, was_limited},
        thread_counts,
        [gpart_mass      = this->gpart_mass,
         h_evol_max      = this->h_evol_max,
         h_evol_iter_max = this->h_evol_iter_max](
            u32 id_a,
            auto ploop_ptrs,
            const Tvec *__restrict r,
            const Tscal *__restrict h_old,
            Tscal *__restrict h_new,
            Tscal *__restrict eps,
            u32 *__restrict was_limited) {
            // attach the neighbor looper on the cache
            shamrock::tree::ObjectCacheIterator particle_looper(ploop_ptrs);

            Tscal part_mass          = gpart_mass;
            Tscal h_max_tot_max_evol = h_evol_max;
            Tscal h_max_evol_p       = h_evol_iter_max;
            Tscal h_max_evol_m       = 1 / h_evol_iter_max;

            // TODO: make this tolerance configurable
            if (eps[id_a] > 1e-6) {

                Tvec xyz_a = r[id_a]; // could be recovered from lambda

                Tscal h_a  = h_new[id_a];
                Tscal dint = h_a * h_a * Rkern * Rkern;

                Tscal rho_sum = 0;
                Tscal sumdWdh = 0;

                particle_looper.for_each_object(id_a, [&](u32 id_b) {
                    Tvec dr    = xyz_a - r[id_b];
                    Tscal rab2 = sycl::dot(dr, dr);

                    if (rab2 > dint) {
                        return; // early return if the particle is too far away
                    }

                    Tscal rab = sycl::sqrt(rab2);

                    rho_sum += part_mass * SPHKernel::W_3d(rab, h_a);
                    sumdWdh += part_mass * SPHKernel::dhW_3d(rab, h_a);
                });

                using namespace shamrock::sph;

                Tscal rho_ha = rho_h(part_mass, h_a, SPHKernel::hfactd);
                Tscal new_h  = newtown_iterate_new_h(rho_ha, rho_sum, sumdWdh, h_a);

                if (new_h < h_a * h_max_evol_m)
                    new_h = h_max_evol_m * h_a;
                if (new_h > h_a * h_max_evol_p)
                    new_h = h_max_evol_p * h_a;

                Tscal ha_0 = h_old[id_a];

                if (new_h < ha_0 * h_max_tot_max_evol) {
                    h_new[id_a] = new_h;
                    eps[id_a]   = sycl::fabs(new_h - h_a) / ha_0;
                } else {
                    h_new[id_a] = ha_0 * h_max_tot_max_evol;
                    eps[id_a]   = -1;
                }
            }
            was_limited[id_a] = 1;
        });
}

template<class Tvec, class SPHKernel>
std::string IterateSmoothingNumDensity<Tvec, SPHKernel>::_impl_get_tex() {
    auto sizes       = get_ro_edge_base(0).get_tex_symbol();
    auto neigh_cache = get_ro_edge_base(1).get_tex_symbol();
    auto positions   = get_ro_edge_base(2).get_tex_symbol();
    auto old_h       = get_ro_edge_base(3).get_tex_symbol();
    auto new_h       = get_rw_edge_base(0).get_tex_symbol();
    auto eps_h       = get_rw_edge_base(1).get_tex_symbol();

    std::string tex = R"tex(
            Iterate smoothing length and density

            \begin{align}
            \rho_i &= \sum_{j \in \mathcal{N}_i} m_j W(r_{ij}, h_i) \\
            \frac{\partial \rho_i}{\partial h_i} &= \sum_{j \in \mathcal{N}_i} m_j \frac{\partial W}{\partial h}(r_{ij}, h_i) \\
            h_i^{\rm new} &= h_i - \frac{\rho_i - \rho_h(m_i, h_i)}{\frac{\partial \rho_i}{\partial h_i} + \frac{3\rho_h(m_i, h_i)}{h_i}} \\
            \epsilon_i &= \frac{|h_i^{\rm new} - h_i|}{h_i^{\rm old}}
            \end{align}

            where:
            \begin{itemize}
            \item $\mathcal{N}_i$ is the set of neighbors of particle $i$
            \item $W(r, h)$ is the SPH kernel function
            \item $\rho_h(m, h) = m \left(\frac{h_{\rm fact}}{h}\right)^3$ is the target density
            \item $h_{\rm fact} = {hfact}$ is the kernel factor
            \item $R_{\rm kern} = {Rkern}$ is the kernel radius
            \end{itemize}

            Input: ${sizes}$, ${neigh_cache}$, ${positions}$, ${old_h}$
            Output: ${new_h}$, ${eps_h}$
        )tex";

    shambase::replace_all(tex, "{sizes}", sizes);
    shambase::replace_all(tex, "{neigh_cache}", neigh_cache);
    shambase::replace_all(tex, "{positions}", positions);
    shambase::replace_all(tex, "{old_h}", old_h);
    shambase::replace_all(tex, "{new_h}", new_h);
    shambase::replace_all(tex, "{eps_h}", eps_h);
    shambase::replace_all(tex, "{hfact}", shambase::format("{}", SPHKernel::hfactd));
    shambase::replace_all(tex, "{Rkern}", shambase::format("{}", SPHKernel::Rkern));

    return tex;
}

template class shammodels::sph::modules::IterateSmoothingNumDensity<f64_3, shammath::M4<f64>>;
template class shammodels::sph::modules::IterateSmoothingNumDensity<f64_3, shammath::M4DH<f64>>;
template class shammodels::sph::modules::IterateSmoothingNumDensity<f64_3, shammath::M4DH3<f64>>;
template class shammodels::sph::modules::IterateSmoothingNumDensity<f64_3, shammath::M4DH5<f64>>;
template class shammodels::sph::modules::IterateSmoothingNumDensity<f64_3, shammath::M4DH7<f64>>;
template class shammodels::sph::modules::IterateSmoothingNumDensity<f64_3, shammath::M4Shift2<f64>>;
template class shammodels::sph::modules::IterateSmoothingNumDensity<f64_3, shammath::M4Shift4<f64>>;
template class shammodels::sph::modules::IterateSmoothingNumDensity<f64_3, shammath::M4Shift8<f64>>;
template class shammodels::sph::modules::
    IterateSmoothingNumDensity<f64_3, shammath::M4Shift16<f64>>;
