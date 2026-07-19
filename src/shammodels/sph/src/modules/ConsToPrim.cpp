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
#include "shamphys/GRUtils.hpp"
#include "shamphys/metrics.hpp"
#include "shamrock/patch/PatchDataField.hpp"
#include "shamsys/NodeInstance.hpp"

namespace {

    template<class Tvec>
    struct KernelConsToPrim {
        using Tscal = shambase::VecComponent<Tvec>;
    };

} // namespace

namespace shammodels::sph::modules {

    template<class Tvec, class SizeType, class Layout, class Accessor>
    void NodeConsToPrim<Tvec, SizeType, Layout, Accessor>::_impl_evaluate_internal() {
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

        auto &gcov = edges.gcov;
        auto &gcon = edges.gcon;

        auto dev_sched = shamsys::instance::get_compute_scheduler_ptr();

        sham::distributed_data_kernel_call(
            dev_sched,
            sham::DDMultiRef{rhostar, momentum, K},
            sham::DDMultiRef{rho, vel, u, P},
            thread_counts,
            [gamma = this->gamma, gcov = edges.gcov](
                u32 id_a,
                Tscal *__restrict rhostar,
                Tvec *__restrict momentum,
                Tscal *__restrict K,

                Tscal *__restrict rho,
                Tvec *__restrict vel,
                Tscal *__restrict u,
                Tscal *__restrict P) {
                // on patch, no need of neighbours

                // get metric quantities
                const Tscal sqrt_g         = get_sqrt_g(vel[id_a], gcov);
                const Tscal inv_sqrt_g     = 1. / sqrt_g;
                const Tscal alpha          = get_alpha(gcon);
                const Tscal sqrt_gamma_inv = alpha * inv_sqrt_g;
                const Tvec betaUP          = get_betaUP(gcon);
                const Tvec betaDOWN        = get_betaDOWN(gcon);
                const std::mdspan<Tscal, std::extents<SizeType, 3, 3>, Layout, Accessor> gammaijUP
                    = get_gammaijUP(gcov);

                Tscal rho_a        = rho[id_a];
                const Tscal gamfac = gamma / (gamma - 1.);
                Tscal w            = 1 + gamfac * P[id_a] / rho[id_a]; // initial guess
                Tscal pp           = 0;
                for (u32 i = 0; i < 3; i++) {
                    for (u32 j = 0; j < 3; j++) {
                        pp += momentum[id_a][i] * gammaijUP(i, j) * momentum[id_a][j];
                    }
                }

                Tscal lorentz_factor = get_lorentz_factor(momentum[id_a], w, gcov);

                // Tscal Kstar = K[id_a] + 0.5 * pp / rhostar[id_a];

                bool converged         = false;
                constexpr Tscal tol    = Tscal(1e-12);
                constexpr u32 Nitermax = 100;
                // compute u
                // iterate
                u32 Niter = 0;

                do {
                    const Tscal w_old = w;

                    lorentz_factor = get_lorentz_factor(momentum[id_a], w, gcov);

                    // eq 97
                    rho_a = rhostar[id_a] * sqrt_gamma_inv / lorentz_factor;

                    // eq 62
                    P[id_a] = K[id_a] * sycl::pow(rho_a, gamma);

                    // eq B4
                    const Tscal f = (1. + gamfac * P[id_a] / rho_a) - w_old;

                    // eq B5 ... maybe should use B6
                    const Tscal df = -1
                                     + gamfac
                                           * (1.
                                              - pp * P[id_a]
                                                    / (lorentz_factor * lorentz_factor * w_old
                                                       * w_old * w_old * rho_a));

                    w = w_old - f / df;

                    converged = (sycl::fabs(w - w_old) / w < tol);
                    Niter++;
                } while (Niter < Nitermax and !converged);

                if (converged) {
                    // compute P, v, u with the last expression of enthalpy
                    lorentz_factor = get_lorentz_factor(momentum[id_a], w, gcov);
                    rho_a          = rhostar[id_a] * sqrt_gamma_inv / lorentz_factor;
                    rho[id_a]      = rho_a;
                    P[id_a]        = K[id_a] * sycl::pow(rho_a, gamma);

                    Tvec v3d = alpha * momentum[id_a] / (w * lorentz_factor) - betaDOWN;

                    // Raise index from down to up
                    for (u32 i = 0; i < 3; i++) {
                        Tscal vi = 0.;
                        for (u32 j = 0; j < 3; j++) {
                            vi += gammaijUP(i, j) * v3d[j];
                        }
                        vel[id_a][i] = vi;
                    }

                    u[id_a] = (P[id_a] < Tscal(1e-30)) ? Tscal(0)
                                                       : P[id_a] / (rho_a * (gamma - Tscal(1)));
                } else {
                    logger::err_ln(
                        "GRSPH",
                        "the enthalpy iterator is not converged after",
                        Niter,
                        "iterations");
                }
            });
    }

    template<class Tvec, class SizeType, class Layout, class Accessor>
    std::string NodeConsToPrim<Tvec, SizeType, Layout, Accessor>::_impl_get_tex() const {

        return "TODO";
    }

} // namespace shammodels::sph::modules
