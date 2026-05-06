// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright (c) 2021-2026 Timothée David--Cléris <tim.shamrock@proton.me>
// SPDX-License-Identifier: CeCILL Free Software License Agreement v2.1
// Shamrock is licensed under the CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

/**
 * @file NodeEvolveDustCOALASourceTerm.cpp
 * @author Timothée David--Cléris (tim.shamrock@proton.me)
 * @brief
 *
 */

#include "shambase/memory.hpp"
#include "shambase/stacktrace.hpp"
#include "shambackends/DeviceBuffer.hpp"
#include "shambackends/kernel_call.hpp"
#include "shambackends/math.hpp"
#include "shambackends/vec.hpp"
#include "shammodels/sph/modules/NodeEvolveDustCOALASourceTerm.hpp"
#include "shamrock/patch/PatchDataField.hpp" // IWYU pragma: keep
#include "shamsys/NodeInstance.hpp"
#include <experimental/mdspan>
#include <vector>

namespace shammodels::sph::modules {

    template<
        class T,
        class E1,
        class E2,
        class E3,
        class E4,
        class L1,
        class L2,
        class L3,
        class L4,
        class A1,
        class A2,
        class A3,
        class A4>
    inline void compute_flux_coag_k0_kdv(
        int nbins,
        const std::mdspan<T, E1, L1, A1> &gij,
        const std::mdspan<const T, E2, L2, A2> &tensor_tabflux_coag,
        const std::mdspan<T, E3, L3, A3> &dv,
        std::mdspan<T, E4, L4, A4> &flux) {
        // --- Compile-time rank checks ---
        static_assert(E1::rank() == 1, "gij must be rank 1");
        static_assert(E2::rank() == 3, "tensor_tabflux_coag must be rank 3");
        static_assert(E3::rank() == 2, "dv must be rank 2");
        static_assert(E4::rank() == 1, "flux must be rank 1");

        // --- Runtime extent checks ---
        SHAM_ASSERT(gij.extent(0) == dv.extent(0));
        SHAM_ASSERT(gij.extent(0) == dv.extent(1));

        SHAM_ASSERT(tensor_tabflux_coag.extent(1) == dv.extent(0));
        SHAM_ASSERT(tensor_tabflux_coag.extent(2) == dv.extent(1));

        SHAM_ASSERT(tensor_tabflux_coag.extent(0) == flux.extent(0));

        /*
         * Python version:
         * flux = np.einsum("jlm,lm,l,m->j", tensor_tabflux_coag, dv, gij, gij)
         */

        for (int j = 0; j < nbins; ++j) {
            double sum = 0.0;
            for (int l = 0; l < nbins; ++l) {
                for (int m = 0; m < nbins; ++m) {
                    sum += tensor_tabflux_coag(j, l, m) * dv(l, m) * gij[l] * gij[m];
                }
            }
            flux[j] = sum;
        }
    }

    template<class Tvec>
    struct KernelGenCoala_k0 {
        using Tscal = shambase::VecComponent<Tvec>;

        u32 nbins;
        Tscal rho_eps;
        u32 corrected_len;
        u32 group_size;
        u32 true_size;

        auto operator()(
            u32 /**/,
            // common to all kernel calls
            const Tscal *__restrict massgrid_ptr,
            const Tscal *__restrict tensor_tabflux_coag,
            // field specific data
            const Tscal *__restrict hpart,
            const Tscal *__restrict s_j,
            const Tvec *__restrict delta_v_j,
            Tscal *__restrict S_coag) const {

            auto range = sycl::nd_range<1>{corrected_len, group_size};

            auto local_acc_sz_nbins  = sycl::range<1>{group_size * nbins};
            auto local_acc_sz_nbins2 = sycl::range<1>{group_size * nbins * nbins};

            return [=, nbins = this->nbins](sycl::handler &cgh) {
                auto gij_acc  = sycl::local_accessor<Tscal>{local_acc_sz_nbins, cgh};
                auto flux_acc = sycl::local_accessor<Tscal>{local_acc_sz_nbins, cgh};
                auto dv_acc   = sycl::local_accessor<Tscal>{local_acc_sz_nbins2, cgh};

                cgh.parallel_for(range, [=](sycl::nd_item<1> tid) {
                    const u64 id_a = tid.get_global_linear_id();
                    const u64 lid  = tid.get_local_linear_id();

                    if (id_a >= true_size) {
                        return;
                    }

                    using mdspan_rank_1 = std::mdspan<Tscal, std::dextents<u32, 1>>;
                    using mdspan_rank_2 = std::mdspan<Tscal, std::dextents<u32, 2>>;
                    using mdspan_rank_3 = std::mdspan<Tscal, std::dextents<u32, 3>>;

                    using const_mdspan_rank_1 = std::mdspan<const Tscal, std::dextents<u32, 1>>;
                    using const_mdspan_rank_3 = std::mdspan<const Tscal, std::dextents<u32, 3>>;

                    auto gij_loc  = &(gij_acc[nbins * lid]);
                    auto flux_loc = &(flux_acc[nbins * lid]);
                    auto dv_loc   = &(dv_acc[nbins * nbins * lid]);

                    mdspan_rank_1 gij(gij_loc, nbins);
                    mdspan_rank_1 flux(flux_loc, nbins);
                    mdspan_rank_2 dv(dv_loc, nbins, nbins);

                    const_mdspan_rank_3 tabflux_coag(tensor_tabflux_coag, nbins, nbins, nbins);
                    const_mdspan_rank_1 massgrid(massgrid_ptr, nbins);

                    auto rho_dust = [&](int j) {
                        auto tmp = s_j[j];
                        return tmp * tmp;
                    };

                    // should implement the same content as
                    // src/pylib/shamrock/external/coala/interface_coala_shamrock.py

                    for (int j = 0; j < nbins; j++) {
                        Tscal rho_d = rho_dust(j);
                        gij(j) = (rho_d > rho_eps) ? rho_d / (massgrid[j + 1] - massgrid[j]) : 0;
                    }

                    // dv_ij = v_dust_j - v_dust_i
                    for (int i = 0; i < nbins; ++i) {
                        for (int j = 0; j < nbins; ++j) {
                            dv(i, j) = sycl::length(delta_v_j[j] - delta_v_j[i]);
                        }
                    }

                    // compute flux for all dust bins
                    compute_flux_coag_k0_kdv(nbins, gij, tabflux_coag, dv, flux);

                    // compute flux diff and store
                    u32 id_a_d     = id_a * nbins;
                    S_coag[id_a_d] = -flux[0];
                    for (int j = 1; j < nbins; ++j) {
                        S_coag[id_a_d + j] = flux[j - 1] - flux[j];
                    }
                });
            };
        }
    };

    template<class Tvec>
    inline void NodeEvolveDustCOALASourceTerm<Tvec>::_impl_evaluate_internal() {

        __shamrock_stack_entry();

        auto edges = get_edges();

        auto hpart_spans     = edges.hpart.get_spans();
        auto s_j_spans       = edges.s_j.get_spans();
        auto delta_v_j_spans = edges.delta_v_j.get_spans();

        auto counts = edges.part_counts.indexes;

        edges.S_coag.ensure_sizes(counts);
        auto S_coag_spans = edges.S_coag.get_spans();

        Tscal rho_eps                                 = edges.rhodust_eps.value;
        const std::vector<Tscal> &massgrid            = edges.massgrid.value;
        const std::vector<Tscal> &tensor_tabflux_coag = edges.tensor_tabflux_coag.value;

        auto dev_sched = shamsys::instance::get_compute_scheduler_ptr();
        auto &q        = shambase::get_check_ref(dev_sched).get_queue();

        sham::DeviceBuffer<Tscal> massgrid_buf(nbins + 1, dev_sched);
        massgrid_buf.copy_from_stdvec(massgrid);

        sham::DeviceBuffer<Tscal> tensor_tabflux_coag_buf(nbins * nbins * nbins, dev_sched);
        tensor_tabflux_coag_buf.copy_from_stdvec(tensor_tabflux_coag);

        u32 group_size = 64;

        counts.for_each([&](u64 id_patch, u64 count) {
            u32 group_cnt     = shambase::group_count(count, group_size);
            u32 corrected_len = group_cnt * group_size;

            sham::kernel_call_hndl(
                q,
                sham::MultiRef{
                    massgrid_buf,
                    tensor_tabflux_coag_buf,
                    hpart_spans.get(id_patch),
                    s_j_spans.get(id_patch),
                    delta_v_j_spans.get(id_patch)},
                sham::MultiRef{S_coag_spans.get(id_patch)},
                count,
                KernelGenCoala_k0<Tvec>{nbins, rho_eps, corrected_len, group_size, u32(count)});
        });
    }
} // namespace shammodels::sph::modules

template class shammodels::sph::modules::NodeEvolveDustCOALASourceTerm<f64_3>;
