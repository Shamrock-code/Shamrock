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
#include "shambase/string.hpp"
#include "shambackends/DeviceBuffer.hpp"
#include "shambackends/kernel_call.hpp"
#include "shambackends/math.hpp"
#include "shambackends/vec.hpp"
#include "shamcomm/logs.hpp"
#include "shammodels/sph/modules/NodeEvolveDustCOALASourceTerm.hpp"
#include "shamphys/coala_interface.hpp"
#include "shamrock/patch/PatchDataField.hpp" // IWYU pragma: keep
#include "shamsys/NodeInstance.hpp"
#include <experimental/mdspan>
#include <vector>

namespace shammodels::sph::modules {

    template<class Tvec>
    struct KernelGenCoala_k0 {
        using Tscal = shambase::VecComponent<Tvec>;

        using mdspan_rank_1 = std::mdspan<Tscal, std::dextents<u32, 1>>;
        using mdspan_rank_3 = std::mdspan<Tscal, std::dextents<u32, 3>>;

        using const_mdspan_rank_1 = std::mdspan<const Tscal, std::dextents<u32, 1>>;
        using const_mdspan_rank_3 = std::mdspan<const Tscal, std::dextents<u32, 3>>;

        u32 nbins;
        Tscal rho_eps;
        Tscal dv_max;
        u32 corrected_len;
        u32 group_size;
        u32 true_size;

        auto operator()(
            u32 /**/,
            // common to all kernel calls
            const Tscal *__restrict massgrid_ptr,
            const Tscal *__restrict tensor_tabflux_coag,
            // field specific data
            const Tscal *__restrict s_j,
            const Tvec *__restrict delta_v_j,
            Tscal *__restrict S_coag) const {

            auto range = sycl::nd_range<1>{corrected_len, group_size};

            auto local_acc_sz_nbins = sycl::range<1>{group_size * nbins};

            auto true_size = this->true_size;
            auto rho_eps   = this->rho_eps;
            auto dv_max    = this->dv_max;

            return [=, nbins = this->nbins](sycl::handler &cgh) {
                auto gij_acc  = sycl::local_accessor<Tscal>{local_acc_sz_nbins, cgh};
                auto flux_acc = sycl::local_accessor<Tscal>{local_acc_sz_nbins, cgh};

                cgh.parallel_for(range, [=](sycl::nd_item<1> tid) {
                    const u64 id_a = tid.get_global_linear_id();
                    const u64 lid  = tid.get_local_linear_id();

                    if (id_a >= true_size) {
                        return;
                    }

                    u32 id_a_d = id_a * nbins;

                    /* inputs */
                    const_mdspan_rank_3 tabflux_coag(tensor_tabflux_coag, nbins, nbins, nbins);
                    const_mdspan_rank_1 massgrid(massgrid_ptr, nbins + 1);

                    /* internal */
                    auto gij_loc  = &(gij_acc[nbins * lid]);
                    auto flux_loc = &(flux_acc[nbins * lid]);

                    mdspan_rank_1 gij(gij_loc, nbins);
                    mdspan_rank_1 flux(flux_loc, nbins);

                    /* output */
                    mdspan_rank_1 S_coag_span(S_coag + id_a_d, nbins);

                    /* lambda getters */
                    auto rho_dust = [&](int j) {
                        auto tmp = s_j[id_a_d + j];
                        return tmp * tmp;
                    };

                    auto dv = [&, delta_v = delta_v_j + id_a_d](int i, int j) {
                        // dv_ij = v_dust_j - v_dust_i = delta_v_j[j] - delta_v_j[i]
                        auto tmp = sycl::length(delta_v[j] - delta_v[i]);
                        return (tmp > dv_max) ? 0 : tmp;
                    };

                    // should implement the same content as
                    // src/pylib/shamrock/external/coala/interface_coala_shamrock.py

                    if (id_a == 1408) {
                        // print coala inputs

                        std::vector<Tvec> delta_v_vec(nbins);
                        for (int i = 0; i < nbins; ++i) {
                            delta_v_vec[i] = (delta_v_j + id_a_d)[i];
                        }

                        std::vector<Tscal> rho_dust_vec(nbins);
                        for (int i = 0; i < nbins; ++i) {
                            rho_dust_vec[i] = rho_dust(i);
                        }

                        std::vector<Tscal> massgrid_vec(nbins + 1);
                        for (int i = 0; i < nbins + 1; ++i) {
                            massgrid_vec[i] = massgrid[i];
                        }

                        Tscal max_tabflux_coag = 0;
                        Tscal min_tabflux_coag = 1000000;

                        std::vector<Tscal> tabflux_coag_vec(nbins * nbins * nbins);
                        for (int i = 0; i < nbins * nbins * nbins; ++i) {
                            tabflux_coag_vec[i] = tensor_tabflux_coag[i];
                            max_tabflux_coag    = std::max(max_tabflux_coag, tabflux_coag_vec[i]);
                            if (tabflux_coag_vec[i] > 0)
                                min_tabflux_coag = std::min(min_tabflux_coag, tabflux_coag_vec[i]);
                        }

                        logger::raw_ln("delta_v = ", delta_v_vec);
                        logger::raw_ln("rho_dust = ", rho_dust_vec);
                        logger::raw_ln("rho_eps = ", rho_eps);
                        logger::raw_ln("massgrid = ", massgrid_vec);
                        // logger::raw_ln("tabflux_coag = ", tabflux_coag_vec);
                        logger::raw_ln("max_tabflux_coag = ", max_tabflux_coag);
                        logger::raw_ln("min_tabflux_coag = ", min_tabflux_coag);
                    }

                    shamphys::coala_k0_source_term(
                        nbins,
                        dv,
                        rho_dust,
                        rho_eps,
                        massgrid,
                        tabflux_coag,
                        gij,
                        flux,
                        S_coag_span);

                    Tscal S_coag_max_norm = 0;
                    Tscal S_coag_sum      = 0;
                    for (int j = 0; j < S_coag_span.extent(0); ++j) {
                        S_coag_sum += S_coag_span(j);
                        S_coag_max_norm = std::max(S_coag_max_norm, std::abs(S_coag_span(j)));
                    }

                    if (id_a == 1408) {
                        std::vector<Tscal> S_coag_vec(nbins);
                        for (int i = 0; i < nbins; ++i) {
                            S_coag_vec[i] = S_coag_span(i);
                        }
                        logger::raw_ln("S_coag = ", S_coag_vec);

                        logger::raw_ln(
                            "S_coag_sum = ", S_coag_sum, " S_coag_max_norm = ", S_coag_max_norm);
                    }
                });
            };
        }
    };

    template<class Tvec>
    inline void NodeEvolveDustCOALASourceTerm<Tvec>::_impl_evaluate_internal() {

        __shamrock_stack_entry();

        auto edges = get_edges();

        auto s_j_spans       = edges.s_j.get_spans();
        auto delta_v_j_spans = edges.delta_v_j.get_spans();

        auto counts = edges.part_counts.indexes;

        edges.S_coag.ensure_sizes(counts);
        auto S_coag_spans = edges.S_coag.get_spans();

        Tscal rho_eps                                 = edges.rhodust_eps.value;
        Tscal dv_max                                  = edges.dv_max.value;
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
                    s_j_spans.get(id_patch),
                    delta_v_j_spans.get(id_patch)},
                sham::MultiRef{S_coag_spans.get(id_patch)},
                count,
                KernelGenCoala_k0<Tvec>{
                    .nbins         = nbins,
                    .rho_eps       = rho_eps,
                    .dv_max        = dv_max,
                    .corrected_len = corrected_len,
                    .group_size    = group_size,
                    .true_size     = u32(count)});
        });
    }

    template<class Tvec>
    std::string NodeEvolveDustCOALASourceTerm<Tvec>::_impl_get_tex() const {

        auto rhodust_eps         = get_ro_edge_base(0).get_tex_symbol();
        auto massgrid            = get_ro_edge_base(1).get_tex_symbol();
        auto tensor_tabflux_coag = get_ro_edge_base(2).get_tex_symbol();
        auto part_counts         = get_ro_edge_base(3).get_tex_symbol();
        auto s_j                 = get_ro_edge_base(5).get_tex_symbol();
        auto delta_v_j           = get_ro_edge_base(6).get_tex_symbol();
        auto S_coag              = get_rw_edge_base(0).get_tex_symbol();

        std::string tex = R"tex(
            COALA dust coagulation source term, DG $k=0$ (Lombart et al., 2021)

            Per gas particle $a$ and mass bin $j$ (monofluid: $\rho_{{\rm d},j,a}} = {s_j}_{j,a}^2$):

            \begin{align}
            \rho_{{\rm d},j,a} &= {s_j}_{j,a}^2 \\
            \Delta m_j &= {massgrid}_{j+1} - {massgrid}_j \\
            g_{j,a} &= \begin{cases}
                \rho_{{\rm d},j,a} / \Delta m_j & \rho_{{\rm d},j,a} > \rho_{\rm eps} \\
                0 & \text{otherwise}
            \end{cases} \\
            \mathrm{dv}_{l,m,a} &= \left| {delta_v_j}_{m,a} - {delta_v_j}_{l,a} \right| \\
            \mathrm{flux}_{j,a} &= \sum_{l,m}
                {tensor_tabflux_coag}_{j,l,m}\,
                \mathrm{dv}_{l,m,a}\, g_{l,a}\, g_{m,a} \\
            {S_coag}_{0,a} &= -\mathrm{flux}_{0,a}, \quad
            {S_coag}_{j,a} = \mathrm{flux}_{j-1,a} - \mathrm{flux}_{j,a}
            \quad (j \ge 1) \\
            a &\in [0, {part_counts}), \quad j,l,m \in [0, N_{\rm bins}) \\
            \rho_{\rm eps} &= {rhodust_eps}, \quad N_{\rm bins} = {nbins}
            \end{align}
        )tex";

        shambase::replace_all(tex, "{rhodust_eps}", rhodust_eps);
        shambase::replace_all(tex, "{massgrid}", massgrid);
        shambase::replace_all(tex, "{tensor_tabflux_coag}", tensor_tabflux_coag);
        shambase::replace_all(tex, "{part_counts}", part_counts);
        shambase::replace_all(tex, "{s_j}", s_j);
        shambase::replace_all(tex, "{delta_v_j}", delta_v_j);
        shambase::replace_all(tex, "{S_coag}", S_coag);
        shambase::replace_all(tex, "{nbins}", shambase::format("{}", nbins));

        return tex;
    }
} // namespace shammodels::sph::modules

template class shammodels::sph::modules::NodeEvolveDustCOALASourceTerm<f64_3>;
