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
 * Only the coagulation flux with a ballistic kernel and pair-wise differential
 * velocities is covered here (\f$k=0\f$ approximation).
 */

#include "shambase/assert.hpp"
#include <experimental/mdspan>
#include <concepts>

namespace shamphys {

    /**
     * @brief Build \f$g_j\f$ coefficients on the piecewise-constant DG basis (\f$k=0\f$)
     *
     * For each mass bin \f$j\f$, converts the dust density to the polynomial coefficient
     * \f$g_j = \rho_{\rm d,j} / \Delta m_j\f$ when \f$\rho_{\rm d,j} > \rho_{\rm eps}\f$,
     * and sets \f$g_j = 0\f$ otherwise, with
     * \f$\Delta m_j = massgrid[j+1] - massgrid[j]\f$ the bin width from consecutive mass-grid edges.
     *
     * @tparam T         Floating-point scalar type
     * @tparam RhoGetter Callable `rho_dust(j) -> T` returning dust density in bin \f$j\f$
     * @tparam E1        `std::mdspan` extents type for @p massgrid
     * @tparam L1        `std::mdspan` layout type for @p massgrid
     * @tparam A1        `std::mdspan` accessor type for @p massgrid
     * @tparam E2        `std::mdspan` extents type for @p gij
     * @tparam L2        `std::mdspan` layout type for @p gij
     * @tparam A2        `std::mdspan` accessor type for @p gij
     * @param rho_dust   Dust density accessor (one value per bin)
     * @param rho_eps    Density threshold below which rho_dust is set to zero
     * @param massgrid   Rank-1 view of mass-bin edges (length `gij.extent(0) + 1`)
     * @param gij        Rank-1 output view of DG coefficients (one per bin); filled in place
     */
    template<class T, class RhoGetter, class E1, class L1, class A1, class E2, class L2, class A2>
    inline void compute_gij_k0(
        RhoGetter &&rho_dust,
        T rho_eps,
        const std::mdspan<const T, E1, L1, A1> &massgrid,
        std::mdspan<T, E2, L2, A2> &gij) {

        static_assert(E1::rank() == 1, "massgrid must be rank 1");
        static_assert(E2::rank() == 1, "gij must be rank 1");

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
     * @tparam T      Floating-point scalar type
     * @tparam E1     `std::mdspan` extents type for @p gij
     * @tparam E2     `std::mdspan` extents type for @p tensor_tabflux_coag
     * @tparam E3     `std::mdspan` extents type for @p flux
     * @tparam L1     `std::mdspan` layout type for @p gij
     * @tparam L2     `std::mdspan` layout type for @p tensor_tabflux_coag
     * @tparam L3     `std::mdspan` layout type for @p flux
     * @tparam A1     `std::mdspan` accessor type for @p gij
     * @tparam A2     `std::mdspan` accessor type for @p tensor_tabflux_coag
     * @tparam A3     `std::mdspan` accessor type for @p flux
     * @tparam Func   Callable `dv(l, m) -> T` returning the differential velocity
     *                  between bins \f$l\f$ and \f$m\f$ (e.g. \f$|\mathbf{v}_m - \mathbf{v}_l|\f$)
     * @param nbins                 Number of dust mass bins
     * @param gij                   Rank-1 view of DG coefficients (\f$g_l\f$, length @p nbins)
     * @param tensor_tabflux_coag   Rank-3 precomputed coagulation flux tensor
     *                              (shape @p nbins \(\times\) @p nbins \(\times\) @p nbins)
     * @param dv                    Pair-wise differential velocity accessor
     * @param flux                  Rank-1 output view of length @p nbins; filled in place
     */
    template<
        class T,
        class E1,
        class E2,
        class E3,
        class L1,
        class L2,
        class L3,
        class A1,
        class A2,
        class A3,
        class Func>
        requires requires(Func f, int a, int b) {
            { f(a, b) } -> std::same_as<T>;
        }
    inline void compute_flux_coag_k0_kdv(
        int nbins,
        const std::mdspan<T, E1, L1, A1> &gij,
        const std::mdspan<const T, E2, L2, A2> &tensor_tabflux_coag,
        Func &&dv,
        std::mdspan<T, E3, L3, A3> &flux) {
        // --- Compile-time rank checks ---
        static_assert(E1::rank() == 1, "gij must be rank 1");
        static_assert(E2::rank() == 3, "tensor_tabflux_coag must be rank 3");
        static_assert(E3::rank() == 1, "flux must be rank 1");

        // --- Runtime extent checks ---
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
     * boundaries) to obtain the source term \f$S_{\rm coag}\f$ in the conservative form of the Smoluchowski equation:
     *
     * \f[
     *     S_{\rm coag}[0] = -\mathrm{flux}[0], \qquad
     *     S_{\rm coag}[j] = \mathrm{flux}[j-1] - \mathrm{flux}[j]
     *     \quad (j \ge 1)
     * \f]
     *
     * @tparam T   Floating-point scalar type
     * @tparam E1  `std::mdspan` extents type for @p flux
     * @tparam L1  `std::mdspan` layout type for @p flux
     * @tparam A1  `std::mdspan` accessor type for @p flux
     * @tparam E2  `std::mdspan` extents type for @p S_coag
     * @tparam L2  `std::mdspan` layout type for @p S_coag
     * @tparam A2  `std::mdspan` accessor type for @p S_coag
     * @param flux    Rank-1 view of coagulation fluxes at bin right edges
     * @param S_coag  Rank-1 output view of the same length; filled in place
     */
    template<class T, class E1, class L1, class A1, class E2, class L2, class A2>
    void coala_flux_diff(
        const std::mdspan<T, E1, L1, A1> &flux, const std::mdspan<T, E2, L2, A2> &S_coag) {

        // --- Compile-time rank checks ---
        static_assert(E1::rank() == 1, "flux must be rank 1");
        static_assert(E2::rank() == 1, "S_coag must be rank 1");

        // --- Runtime extent checks ---
        SHAM_ASSERT(flux.extent(0) == S_coag.extent(0));

        S_coag(0) = -flux(0);
        for (int j = 1; j < flux.extent(0); ++j) {
            S_coag(j) = flux(j - 1) - flux(j);
        }
    }

} // namespace shamphys
