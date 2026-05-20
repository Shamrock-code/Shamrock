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

#include "shambase/assert.hpp"
#include "shambase/mdspan_concepts.hpp"
#include <experimental/mdspan>
#include <concepts>

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

} // namespace shamphys
