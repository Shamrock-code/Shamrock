// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright (c) 2021-2026 Timothée David--Cléris <tim.shamrock@proton.me>
// SPDX-License-Identifier: CeCILL Free Software License Agreement v2.1
// Shamrock is licensed under the CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

#pragma once

/**
 * @file coala_interface.hpp
 * @author Maxime Lombart (maxime.lombart@cea.fr)
 * @author Timothée David--Cléris (tim.shamrock@proton.me)
 * @brief COALA dust coagulation helpers for a DG \f$k=0\f$ (piecewise-constant) basis
 *
 * C++ counterparts of the COALA Python routines used to build dust coagulation source
 * terms in the conservative form of the Smoluchowski equation (Lombart et al., 2021). The reference
 * implementation lives in
 * `src/pylib/shamrock/external/coala/interface_coala_shamrock.py` and
 * `src/pylib/shamrock/external/coala/generate_flux_intflux.py`.
 *
 * Only the coagulation flux with \f$k=0\f$ approximation.
 */

#include "shambase/aliases_int.hpp"
#include "shambase/assert.hpp"
#include "shambase/mdspan_concepts.hpp"
#include <experimental/mdspan>
#include <concepts>
#include <vector>

namespace shamphys {

    /**
     * @brief Build \f$g_j\f$ coefficients on the piecewise-constant DG basis (\f$k=0\f$)
     *
     * For each mass bin \f$j\f$, converts the dust density to the polynomial coefficient
     * \f$g_j = \rho_{\rm d,j} / \Delta m_j\f$ when \f$\rho_{\rm d,j} > \rho_{\rm eps}\f$,
     * and sets \f$g_j = 0\f$ otherwise, with
     * \Delta m_j = massgrid[j+1] - massgrid[j] the bin width from consecutive mass-grid
     * edges.
     *
     * @tparam T  Floating-point scalar type; @p rho_dust must satisfy
     *            `rho_dust(j) -> T`, and @p massgrid / @p gij must be rank-1 `std::mdspan`
     *            with element type `T`
     * @param rho_dust  Callable invoked as `rho_dust(j)` returning dust density in bin \f$j\f$
     * @param rho_eps     Density threshold below which \f$g_j\f$ is set to zero
     * @param massgrid    Rank-1 `std::mdspan` (`shambase::is_mdspan_rank<1>`) of bin-edge masses;
     *                    extent must be `gij.extent(0) + 1`
     * @param gij         Rank-1 `std::mdspan` (`shambase::is_mdspan_rank<1>`) of DG coefficients;
     *                    one entry per bin, written in place
     */
    template<class T>
    inline void compute_gij_k0(
        auto &&rho_dust,
        T rho_eps,
        shambase::is_mdspan_rank<1> auto massgrid,
        shambase::is_mdspan_rank<1> auto gij)
        requires requires(decltype(rho_dust) rd, int j) {
            { rd(j) } -> std::same_as<T>;
        }
    {

        SHAM_ASSERT(massgrid.extent(0) == gij.extent(0) + 1);

        for (std::size_t j = 0; j < gij.extent(0); ++j) {
            T rho_d = rho_dust(j);
            gij(j)  = (rho_d > rho_eps) ? rho_d / (massgrid[j + 1] - massgrid[j]) : 0;
        }
    }

    /**
     * @brief Coagulation flux at bin right edges for a ballistic kernel (\f$k=0\f$)
     *
     * Evaluates the flux approximation at the right boundary of each mass bin,
     * \f$\mathrm{flux}[j] \approx F(m_{j+1/2})\f$, by summing over all bin pairs
     * \f$(l, m)\f$:
     *
     * \f[
     *     \mathrm{flux}[j] = \sum_{l,m}
     *         \mathrm{tensor\_tabflux\_coag}[j,l,m]\,
     *         \mathrm{dv}(l,m)\, g_l\, g_m
     * \f]
     *
     * Equivalent to the NumPy contraction
     * `einsum("jlm,lm,l,m->j", tensor_tabflux_coag, dv, gij, gij)`.
     *
     * @p gij, @p tensor_tabflux_coag and @p flux are expected to share the same scalar element
     * type.
     *
     * @tparam Func  Callable invoked as `dv(l, m)` returning the differential velocity between
     *               bins \f$l\f$ and \f$m\f$ (e.g. \f$|\mathbf{v}_m - \mathbf{v}_l|\f$)
     * @param nbins                Number of dust mass bins
     * @param gij                  Rank-1 `std::mdspan` (`shambase::is_mdspan_rank<1>`) of DG
     *                             coefficients \f$g_l\f$; extent @p nbins
     * @param tensor_tabflux_coag  Rank-3 `std::mdspan` (`shambase::is_mdspan_rank<3>`) of
     *                             precomputed coagulation flux entries; extents
     *                             @p nbins \(\times\) @p nbins \(\times\) @p nbins
     * @param dv                   Pair-wise differential-velocity callable
     * @param flux                 Rank-1 `std::mdspan` (`shambase::is_mdspan_rank<1>`) of output
     *                             fluxes; extent @p nbins, written in place
     */
    template<class Func>
        requires requires(Func f, int a, int b) {
            { f(a, b) };
        }
    inline void compute_flux_coag_k0_kdv(
        int nbins,
        shambase::is_mdspan_rank<1> auto gij,
        shambase::is_mdspan_rank<3> auto tensor_tabflux_coag,
        Func &&dv,
        shambase::is_mdspan_rank<1> auto flux) {

        SHAM_ASSERT(gij.extent(0) == nbins);
        SHAM_ASSERT(flux.extent(0) == nbins);
        SHAM_ASSERT(tensor_tabflux_coag.extent(0) == nbins);
        SHAM_ASSERT(tensor_tabflux_coag.extent(1) == nbins);
        SHAM_ASSERT(tensor_tabflux_coag.extent(2) == nbins);

        // initialize flux to 0
        for (int j = 0; j < nbins; ++j) {
            flux[j] = 0;
        }

        /*
         * Python version:
         * flux = np.einsum("jlm,lm,l,m->j", tensor_tabflux_coag, dv, gij, gij)
         */

        for (int l = 0; l < nbins; ++l) {
            for (int m = 0; m < nbins; ++m) {
                auto term = dv(l, m) * gij[l] * gij[m];
                for (int j = 0; j < nbins; ++j) {
                    flux[j] += tensor_tabflux_coag(j, l, m) * term;
                }
            }
        }
    }

    /**
     * @brief Sparse storage of `tensor_tabflux_coag` (\f$k=0\f$)
     *
     * For each ordered pair \f$p = (l,m)\f$, enumerated as `for l, for m`, only the range
     * \f$j \in [{\rm pair\_jmin}[p], {\rm pair\_jmin}[p] + {\rm pair\_offset}[p+1] -
     * {\rm pair\_offset}[p])\f$ containing all the non-zero entries
     * \f$\mathrm{tensor\_tabflux\_coag}[j,l,m]\f$ is stored, contiguously in `values` starting
     * at `pair_offset[p]`. The range is found from the tensor itself, so no assumption is made
     * on its sparsity pattern, nor on the symmetry of \f$\mathrm{dv}\f$.
     *
     * @tparam T     Floating-point scalar type
     * @tparam Tidx  Index type
     */
    template<class T, class Tidx = u32>
    struct TabfluxCoagK0Sparse {
        /// Offset in values of each pair, size nbins^2 + 1
        std::vector<Tidx> pair_offset;
        /// First bin \f$j\f$ stored for each pair, size nbins^2
        std::vector<Tidx> pair_jmin;
        /// Stored entries
        std::vector<T> values;
    };

    /**
     * @brief Device view of a TabfluxCoagK0Sparse (see its documentation for the layout)
     */
    template<class T, class Tidx = u32>
    struct TabfluxCoagK0SparseView {
        Tidx nbins;
        const Tidx *pair_offset;
        const Tidx *pair_jmin;
        const T *values;

        /// The stored entries of one pair \f$(l,m)\f$: \f$j \in [j_{\rm min}, j_{\rm min} +
        /// {\rm count})\f$
        struct PairEntries {
            const T *values;
            Tidx jmin;
            Tidx count;

            /// Whether the pair has no stored entry (it does not contribute to the flux)
            inline bool is_empty() const { return count == 0; }

            /// Call `func(j, tensor_tabflux_coag[j,l,m])` for each stored entry of the pair
            template<class Func>
            inline void for_each_sparse(Func &&func) const {
                for (Tidx k = 0; k < count; ++k) {
                    func(jmin + k, values[k]);
                }
            }
        };

        /// Stored entries of the pair \f$(l,m)\f$
        inline PairEntries get_entries(Tidx l, Tidx m) const {
            Tidx p   = l * nbins + m;
            Tidx beg = pair_offset[p];
            return PairEntries{values + beg, pair_jmin[p], pair_offset[p + 1] - beg};
        }
    };

    /**
     * @brief Build the TabfluxCoagK0Sparse form of `tensor_tabflux_coag`
     *
     * @param nbins                Number of dust mass bins
     * @param tensor_tabflux_coag  Rank-3 `std::mdspan` of the dense tensor; extents
     *                             @p nbins \(\times\) @p nbins \(\times\) @p nbins
     */
    template<class T, class Tidx = u32>
    inline TabfluxCoagK0Sparse<T, Tidx> make_tabflux_coag_k0_sparse(
        int nbins, shambase::is_mdspan_rank<3> auto tensor_tabflux_coag) {

        SHAM_ASSERT(tensor_tabflux_coag.extent(0) == nbins);
        SHAM_ASSERT(tensor_tabflux_coag.extent(1) == nbins);
        SHAM_ASSERT(tensor_tabflux_coag.extent(2) == nbins);

        TabfluxCoagK0Sparse<T, Tidx> ret;
        ret.pair_offset.push_back(0);

        for (int l = 0; l < nbins; ++l) {
            for (int m = 0; m < nbins; ++m) {
                auto tab = [&](int j) -> T {
                    return tensor_tabflux_coag(j, l, m);
                };

                int jmin = 0;
                int jend = nbins;
                while (jmin < jend && tab(jmin) == 0) {
                    ++jmin;
                }
                while (jend > jmin && tab(jend - 1) == 0) {
                    --jend;
                }

                for (int j = jmin; j < jend; ++j) {
                    ret.values.push_back(tab(j));
                }
                ret.pair_jmin.push_back(jmin);
                ret.pair_offset.push_back(ret.values.size());
            }
        }

        return ret;
    }

    /**
     * @brief Same as compute_flux_coag_k0_kdv but using the TabfluxCoagK0Sparse form
     *
     * Pairs \f$(l,m)\f$ without any non-zero entry are skipped before evaluating
     * \f$\mathrm{dv}(l,m)\f$.
     *
     * @param nbins    Number of dust mass bins
     * @param gij      Rank-1 `std::mdspan` of DG coefficients \f$g_l\f$; extent @p nbins
     * @param tabflux  View of the TabfluxCoagK0Sparse tensor
     * @param dv       Pair-wise differential-velocity callable, invoked as `dv(l, m)`
     * @param flux     Rank-1 `std::mdspan` of output fluxes; extent @p nbins, written in place
     */
    template<class T, class Tidx, class Func>
        requires requires(Func f, int a, int b) {
            { f(a, b) };
        }
    inline void compute_flux_coag_k0_kdv(
        int nbins,
        shambase::is_mdspan_rank<1> auto gij,
        TabfluxCoagK0SparseView<T, Tidx> tabflux,
        Func &&dv,
        shambase::is_mdspan_rank<1> auto flux) {

        SHAM_ASSERT(gij.extent(0) == nbins);
        SHAM_ASSERT(flux.extent(0) == nbins);

        for (int j = 0; j < nbins; ++j) {
            flux[j] = 0;
        }

        for (int l = 0; l < nbins; ++l) {
            for (int m = 0; m < nbins; ++m) {
                auto entries = tabflux.get_entries(l, m);

                // skip the pairs without entries before evaluating dv
                if (entries.is_empty()) {
                    continue;
                }

                auto term = dv(l, m) * gij[l] * gij[m];
                entries.for_each_sparse([&](Tidx j, T val) {
                    flux[j] += val * term;
                });
            }
        }
    }

    /**
     * @brief Convert interface fluxes to a mass-bin coagulation source term
     *
     * Applies the DG \f$k=0\f$ divergence operator (finite difference across bin
     * boundaries) to obtain the source term \f$S_{\rm coag}\f$ in the conservative form of the
     * Smoluchowski equation:
     *
     * \f[
     *     S_{\rm coag}[0] = -\mathrm{flux}[0], \qquad
     *     S_{\rm coag}[j] = \mathrm{flux}[j-1] - \mathrm{flux}[j]
     *     \quad (j \ge 1)
     * \f]
     *
     * @param flux    Rank-1 view of coagulation fluxes at bin right edges
     * @param S_coag  Rank-1 output view of the same length; filled in place
     */
    void coala_flux_diff(
        shambase::is_mdspan_rank<1> auto flux, shambase::is_mdspan_rank<1> auto S_coag) {

        SHAM_ASSERT(flux.extent(0) == S_coag.extent(0));

        S_coag(0) = -flux(0);
        for (int j = 1; j < flux.extent(0); ++j) {
            S_coag(j) = flux(j - 1) - flux(j);
        }
    }

    template<class T, class FuncDv, class FuncRhoDust>
    void coala_k0_source_term(
        int nbins,
        /* inputs */
        FuncDv &&dv,
        FuncRhoDust &&rho_dust,
        T rho_eps,
        shambase::is_mdspan_rank<1> auto massgrid,
        /* COALA inputs (dense rank-3 mdspan or TabfluxCoagK0SparseView) */
        auto tabflux_coag,
        /* internal */
        shambase::is_mdspan_rank<1> auto gij,
        shambase::is_mdspan_rank<1> auto flux,
        /* output */
        shambase::is_mdspan_rank<1> auto S_coag) {

        // init the gij coefficients
        shamphys::compute_gij_k0(rho_dust, rho_eps, massgrid, gij);

        // compute flux for all dust bins
        shamphys::compute_flux_coag_k0_kdv(nbins, gij, tabflux_coag, dv, flux);

        // compute flux diff and store result
        shamphys::coala_flux_diff(flux, S_coag);
    }

} // namespace shamphys
