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
        auto edges = get_edges();
        auto &thread_counts = edges.sizes.indexes;

        edges.spans_rhostar.check_sizes(thread_counts);
        edges.spans_momentum.check_sizes(thread_counts);
        edges.spans_K.check_sizes(thread_counts);
        edges.spans_u.check_sizes(thread_counts);
        edges.spans_P.check_sizes(thread_counts);

        auto &rhostar = edges.spans_rhostar.get_spans();
        auto &momentum = edges.spans_momentum.get_spans();
        auto &K = edges.spans_K.get_spans();
        auto &u = edges.spans_u.get_spans();
        auto &P = edges.spans_P.get_spans();

        auto dev_sched = shamsys::instance::get_compute_scheduler_ptr();

        sham::distributed_data_kernel_call(
            dev_sched,
            sham::DDMultiRef{rhostar, momentum, K},
            sham::DDMultiRef{u, P}, 
            thread_counts, 
            [gamma = this->gamma](
                u32 id_a, 
                Tscal *__restrict rhostar, 
                Tvec  *__restrict momentum, 
                Tscal *__restrict K, 
                Tscal *__restrict u, 
                Tscal *__restrict P) {
                // on patch, no need of neighbours

                //get metric

                //guess enthalpy w, with adiabatic EOS and previous values
                Tscal w = gamma/(gamma-1) * P[id_a] / rhostar[id_a];
                bool converged = false;
                //compute u 
                //iterate
                u32 Niter = 0;
                do {

                    Tscal new_w = 0;

                    converged = sycl::fabs(new_w - w) < 1e-6;
                    w = new_w;
                    Niter++;
                } while (Niter < 100 or converged);
            });


    }

    template<class Tvec>
    std::string NodeConsToPrim<Tvec>::_impl_get_tex() const {

        return "TODO";
    }

} // namespace shammodels::basegodunov::modules

template class shammodels::basegodunov::modules::NodeConsToPrim<f64_3>;
