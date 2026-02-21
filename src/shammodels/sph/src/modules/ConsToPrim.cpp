// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright (c) 2021-2026 Timothée David--Cléris <tim.shamrock@proton.me>
// SPDX-License-Identifier: CeCILL Free Software License Agreement v2.1
// Shamrock is licensed under the CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

/**
 * @file ConsToPrim.cpp
 * @author Yona Lapeyre (yona.lapeyre@ens-lyon.fr)
 * @brief
 *
 */

#include "shammodels/sph/modules/ConsToPrim.hpp"
#include "shambackends/kernel_call_distrib.hpp"
#include "shamcomm/logs.hpp"
#include "shammath/riemann.hpp"
#include "shamrock/patch/PatchDataField.hpp"
#include "shamsys/NodeInstance.hpp"

namespace {

    template<class Tvec>
    struct KernelConsToPrim {
        using Tscal = shambase::VecComponent<Tvec>;
    };

} // namespace

namespace shammodels::basegodunov::modules {

    template<class Tvec>
    void NodeConsToPrim<Tvec>::_impl_evaluate_internal() {
        auto edges          = get_edges();
        auto &thread_counts = edges.sizes.indexes;

        edges.spans_rhostar.check_sizes(thread_counts);
        edges.spans_momentum.check_sizes(thread_counts);
        edges.spans_K.check_sizes(thread_counts);

        edges.spans_rho.check_sizes(thread_counts);
        edges.spans_vel.check_sizes(thread_counts);
        edges.spans_u.check_sizes(thread_counts);
        edges.spans_P.check_sizes(thread_counts);

        auto &rhostar  = edges.spans_rhostar.get_spans();
        auto &momentum = edges.spans_momentum.get_spans();
        auto &K        = edges.spans_K.get_spans();

        auto &rho = edges.spans_rho.get_spans();
        auto &vel = edges.spans_vel.get_spans();
        auto &u   = edges.spans_u.get_spans();
        auto &P   = edges.spans_P.get_spans();

        auto dev_sched = shamsys::instance::get_compute_scheduler_ptr();

        sham::distributed_data_kernel_call(
            dev_sched,
            sham::DDMultiRef{rhostar, momentum, K},
            sham::DDMultiRef{rho, vel, u, P},
            thread_counts,
            [gamma = this->gamma](
                u32 id_a,
                Tscal *__restrict rhostar,
                Tvec *__restrict momentum,
                Tscal *__restrict K,

                Tscal *__restrict rho,
                Tvec *__restrict vel,
                Tscal *__restrict u,
                Tscal *__restrict P) {
                // on patch, no need of neighbours

                // get metric
                Tscal sqrt_g         = get_sqrtg(gcov);
                Tscal inv_sqrt_g     = 1. / sqrt_g;
                Tscal sqrt_gamma     = get_sqrt_gamma(gcov);
                Tscal sqrt_gamma_inv = alpha * inv_sqrt_g;

                // guess enthalpy w, with adiabatic EOS and previous values
                Tscal w        = gamma / (gamma - 1) * P[id_a] / rhostar[id_a];
                bool converged = false;
                // compute u
                // iterate
                u32 Niter = 0;
                do {
                    // get values of density and pressure from alod w
                    Tscal lorentz_factor
                        = sycl::sqrt(1. + sycl::dot(momentum[id_a], momentum[id_a] / (w * w)));

                    rho[id_a]   = sqrt_gamma_inv * rhostar[id_a] / lorentz_factor;
                    Tscal polyk = 1.;
                    P[id_a]     = (gamma - 1.) * rho[id_a] * polyk;

                    Tscal new_w = 1;

                    converged = sycl::fabs(new_w - w) < 1e-6;
                    w         = new_w;
                    Niter++;
                } while (Niter < 100 or converged);

                if (converged) {
                    Tvec v3d = alpha * momentum[id_a] / (w * lorentz_factor) - betadown;
                    // Raise index from down to up
                    for (u32 i = 0; i < 4; i++) {
                        vel[id_a][i] = sycl::dot(gammaijUP[:, i], v3d);
                    }
                } else {
                    logger::err_ln(
                        "GRSPH",
                        "the enthalpy iterator is not converged after",
                        Niter,
                        "iterations");
                }
            });
    }

    template<class Tvec>
    std::string NodeConsToPrim<Tvec>::_impl_get_tex() const {

        return "TODO";
    }

} // namespace shammodels::basegodunov::modules

template class shammodels::basegodunov::modules::NodeConsToPrim<f64_3>;
