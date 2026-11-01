// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright (c) 2021-2026 Timothée David--Cléris <tim.shamrock@proton.me>
// SPDX-License-Identifier: CeCILL Free Software License Agreement v2.1
// Shamrock is licensed under the CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

#include "shambase/aliases_float.hpp"
#include "shambase/aliases_int.hpp"
#include "shambackends/sycl.hpp" // before <experimental/mdspan>, which uses sycl:: under DPC++
#include "shamphys/coala_interface.hpp"
#include "shamtest/shamtest.hpp"
#include <experimental/mdspan>
#include <array>
#include <cmath>
#include <random>
#include <vector>

/// compare the sparse flux to the dense reference on random inputs
void test_coala_flux_sparse(int nbins, f64 zero_fraction) {

    std::mt19937 eng(0x1111 + nbins);
    std::uniform_real_distribution<f64> distval(0.1, 1.0);
    std::uniform_real_distribution<f64> distvel(-1.0, 1.0);
    std::uniform_real_distribution<f64> dist01(0.0, 1.0);

    // dense tensor, with some zeros (including whole leading/trailing j ranges)
    std::vector<f64> tab(nbins * nbins * nbins);
    std::mdspan<f64, std::dextents<u32, 3>> tab_span(tab.data(), nbins, nbins, nbins);
    for (int j = 0; j < nbins; j++) {
        for (int l = 0; l < nbins; l++) {
            for (int m = 0; m < nbins; m++) {
                tab_span(j, l, m) = (dist01(eng) < zero_fraction) ? 0 : distval(eng);
            }
        }
    }

    std::vector<f64> gij(nbins);
    for (auto &g : gij) {
        g = distval(eng);
    }
    if (nbins > 2) {
        gij[1] = 0; // empty bin
    }

    std::vector<std::array<f64, 3>> v(nbins);
    for (auto &vi : v) {
        vi = {distvel(eng), distvel(eng), distvel(eng)};
    }
    // not symmetric and non-zero on the diagonal on purpose, nothing should assume otherwise
    auto dv = [&](int l, int m) {
        f64 dx = v[m][0] - v[l][0], dy = v[m][1] - v[l][1], dz = v[m][2] - v[l][2];
        return std::sqrt(dx * dx + dy * dy + dz * dz) + 0.1 * (l + 1) / (m + 2);
    };

    std::mdspan<f64, std::dextents<u32, 1>> gij_span(gij.data(), nbins);

    std::vector<f64> flux_ref(nbins);
    std::mdspan<f64, std::dextents<u32, 1>> flux_ref_span(flux_ref.data(), nbins);
    shamphys::compute_flux_coag_k0_kdv(nbins, gij_span, tab_span, dv, flux_ref_span);

    auto sparse = shamphys::make_tabflux_coag_k0_sparse<f64>(nbins, tab_span);

    REQUIRE_EQUAL(sparse.pair_offset.size(), usize(nbins * nbins + 1));
    REQUIRE_EQUAL(sparse.pair_jmin.size(), usize(nbins * nbins));

    shamphys::TabfluxCoagK0SparseView<f64> view{
        u32(nbins), sparse.pair_offset.data(), sparse.pair_jmin.data(), sparse.values.data()};

    std::vector<f64> flux(nbins);
    std::mdspan<f64, std::dextents<u32, 1>> flux_span(flux.data(), nbins);
    shamphys::compute_flux_coag_k0_kdv(nbins, gij_span, view, dv, flux_span);

    for (int j = 0; j < nbins; j++) {
        REQUIRE_FLOAT_EQUAL(flux[j], flux_ref[j], 1e-12 * (1 + std::abs(flux_ref[j])));
    }
}

NEW_TEST(Unittest, "shamphys/coala_interface/flux_sparse", 1) {
    for (int nbins : {1, 2, 3, 7, 20}) {
        for (f64 zero_fraction : {0.0, 0.5, 0.9}) {
            test_coala_flux_sparse(nbins, zero_fraction);
        }
    }
}
