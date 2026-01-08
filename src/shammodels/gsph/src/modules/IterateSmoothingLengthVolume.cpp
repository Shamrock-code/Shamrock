// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright (c) 2021-2025 Timothée David--Cléris <tim.shamrock@proton.me>
// SPDX-License-Identifier: CeCILL Free Software License Agreement v2.1
// Shamrock is licensed under the CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

/**
 * @file IterateSmoothingLengthVolume.cpp
 * @author Guo Yansong (guo.yansong.ngy@gmail.com)
 * @brief Volume-based smoothing length iteration (Kitajima et al. 2025)
 */

#include "shambase/stacktrace.hpp"
#include "shammodels/gsph/modules/IterateSmoothingLengthVolume.hpp"
#include "shambackends/kernel_call_distrib.hpp"
#include "shammath/sphkernels.hpp"
#include "shammodels/sph/math/density.hpp"
#include "shamrock/patch/PatchDataField.hpp"

using namespace shammodels::gsph::modules;

template<class Tvec, class SPHKernel>
void IterateSmoothingLengthVolume<Tvec, SPHKernel>::_impl_evaluate_internal() {
    StackEntry stack_loc{};

    auto edges = get_edges();

    auto &thread_counts = edges.sizes.indexes;

    edges.neigh_cache.check_sizes(thread_counts);
    edges.positions.check_sizes(thread_counts);
    edges.old_h.check_sizes(thread_counts);
    edges.new_h.check_sizes(thread_counts);
    edges.eps_h.check_sizes(thread_counts);

    auto &neigh_cache = edges.neigh_cache.neigh_cache;
    auto &positions   = edges.positions.get_spans();
    auto &old_h       = edges.old_h.get_spans();
    auto &new_h       = edges.new_h.get_spans();
    auto &eps_h       = edges.eps_h.get_spans();

    auto dev_sched = shamsys::instance::get_compute_scheduler_ptr();

    static constexpr Tscal Rkern = SPHKernel::Rkern;
    static constexpr u32 dim     = shambase::VectorProperties<Tvec>::dimension;

    sham::distributed_data_kernel_call(
        dev_sched,
        sham::DDMultiRef{neigh_cache, positions, old_h},
        sham::DDMultiRef{new_h, eps_h},
        thread_counts,
        [h_evol_max      = this->h_evol_max,
         h_evol_iter_max = this->h_evol_iter_max,
         c_smooth        = this->c_smooth](
            u32 id_a,
            auto ploop_ptrs,
            const Tvec *__restrict r,
            const Tscal *__restrict h_old,
            Tscal *__restrict h_new,
            Tscal *__restrict eps) {
            // Attach the neighbor looper on the cache
            shamrock::tree::ObjectCacheIterator particle_looper(ploop_ptrs);

            Tscal h_max_tot_max_evol = h_evol_max;
            Tscal h_max_evol_p       = h_evol_iter_max;
            Tscal h_max_evol_m       = Tscal(1) / h_evol_iter_max;

            // Volume-based h: only iterate if not converged
            if (eps[id_a] > Tscal(1e-6)) {

                Tvec xyz_a = r[id_a];
                Tscal h_a  = h_new[id_a]; // Current h being iterated
                Tscal ha_0 = h_old[id_a]; // Original h at start of subcycle

                // Kitajima Eq. 232-233: Use C_smooth × h for computing V_p*
                // V_p*(x) = [Σ_j W(x-x_j, C_smooth × h(x))]^(-1)
                // This smoothes h variation across discontinuities
                Tscal h_smooth = c_smooth * h_a;
                Tscal dint     = h_smooth * h_smooth * Rkern * Rkern;

                // Volume-based approach (Kitajima Eq. 232-233 - gather approach):
                // V_p* = [Σ_j W(r_ij, C_smooth × h_i)]^(-1)
                // h_i = η × V_p*^(1/dim)
                // This is implicitly coupled, so we iterate until convergence.
                // IMPORTANT: Σ_j includes self (j=i), so add self-contribution W(0, h)
                Tscal W_sum = SPHKernel::W_3d(Tscal(0), h_smooth); // Self-contribution

                particle_looper.for_each_object(id_a, [&](u32 id_b) {
                    // Skip self (already counted above)
                    if (id_a == id_b)
                        return;

                    Tvec dr    = xyz_a - r[id_b];
                    Tscal rab2 = sycl::dot(dr, dr);

                    // Use C_smooth × h for cutoff (Kitajima Eq. 232-233)
                    if (rab2 > dint) {
                        return;
                    }

                    Tscal rab = sycl::sqrt(rab2);

                    // Volume-based gather: use C_smooth × particle's h (Kitajima Eq. 232)
                    // W_sum = Σ_j W(r_ij, C_smooth × h_i)
                    W_sum += SPHKernel::W_3d(rab, h_smooth);
                });

                // Direct h computation from volume (Kitajima Eq. 221, 230-231):
                // V = 1/W_sum, h = η × V^(1/d) = η / W_sum^(1/d)
                // In Shamrock, η = hfact (= hfactd)
                Tscal hfact = SPHKernel::hfactd;
                Tscal new_h_val;

                // FAIL FAST: Check for invalid W_sum
                if (!sycl::isfinite(W_sum) || W_sum <= Tscal{0}) {
                    printf("H_ITER FAIL: particle %u has invalid W_sum=%.6e (h=%.6e)\\n",
                           id_a, (double)W_sum, (double)h_a);
                    new_h_val = h_a;  // Keep old value to continue and expose more errors
                } else if constexpr (dim == 3) {
                    new_h_val = hfact / sycl::cbrt(W_sum);
                } else if constexpr (dim == 2) {
                    new_h_val = hfact / sycl::sqrt(W_sum);
                } else {
                    new_h_val = hfact / W_sum;
                }

                // FAIL FAST: Check for invalid new_h
                if (!sycl::isfinite(new_h_val) || new_h_val <= Tscal(0)) {
                    printf("H_ITER FAIL: particle %u computed invalid h=%.6e (W_sum=%.6e)\\n",
                           id_a, (double)new_h_val, (double)W_sum);
                    new_h_val = h_a;  // Keep old value to continue and expose more errors
                }

                // Per-iteration clamping (like standard SPH iteration)
                if (new_h_val < h_a * h_max_evol_m)
                    new_h_val = h_max_evol_m * h_a;
                if (new_h_val > h_a * h_max_evol_p)
                    new_h_val = h_max_evol_p * h_a;

                // Total evolution clamp per subcycle
                if (new_h_val < ha_0 * h_max_tot_max_evol) {
                    h_new[id_a] = new_h_val;
                    eps[id_a]   = sycl::fabs(new_h_val - h_a) / ha_0;
                } else {
                    h_new[id_a] = ha_0 * h_max_tot_max_evol;
                    eps[id_a]   = Tscal(-1); // Signal cache rebuild needed
                }
            }
        });
}

template<class Tvec, class SPHKernel>
std::string IterateSmoothingLengthVolume<Tvec, SPHKernel>::_impl_get_tex() const {
    return R"(
\textbf{Volume-based smoothing length (Kitajima et al. 2025)}

Density with averaged $h$:
\[
\rho_i = \sum_j m_j W(|\mathbf{r}_i - \mathbf{r}_j|, \bar{h}_{ij})
\]
where $\bar{h}_{ij} = (h_i + h_j)/2$

Direct $h$ computation:
\[
h_i = \eta \left(\frac{m}{\rho_i}\right)^{1/d}
\]
)";
}

// Explicit template instantiations
template class shammodels::gsph::modules::IterateSmoothingLengthVolume<f64_3, shammath::M4<f64>>;
template class shammodels::gsph::modules::IterateSmoothingLengthVolume<f64_3, shammath::M6<f64>>;
template class shammodels::gsph::modules::IterateSmoothingLengthVolume<f64_3, shammath::M8<f64>>;

template class shammodels::gsph::modules::IterateSmoothingLengthVolume<f64_3, shammath::C2<f64>>;
template class shammodels::gsph::modules::IterateSmoothingLengthVolume<f64_3, shammath::C4<f64>>;
template class shammodels::gsph::modules::IterateSmoothingLengthVolume<f64_3, shammath::C6<f64>>;

template class shammodels::gsph::modules::
    IterateSmoothingLengthVolume<f64_3, shammath::TGauss3<f64>>;
