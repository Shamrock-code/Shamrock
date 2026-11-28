// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright (c) 2021-2026 Timothée David--Cléris <tim.shamrock@proton.me>
// SPDX-License-Identifier: CeCILL Free Software License Agreement v2.1
// Shamrock is licensed under the CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

#include "shambase/DistributedData.hpp"
#include "shambase/constants.hpp"
#include "shambase/string.hpp"
#include "shamalgs/primitives/mock_vector.hpp"
#include "shamcomm/logs.hpp"
#include "shammodels/common/amr/NeighGraph.hpp"
#include "shammodels/ramses/modules/NodeSpMVPoisson3D.hpp"
#include "shammodels/ramses/solvegraph/OrientedAMRGraphEdge.hpp"
#include "shamrock/solvergraph/Field.hpp"
#include "shamrock/solvergraph/Indexes.hpp"
#include "shamtest/shamtest.hpp"
#include "tests/shammodels/ramses/modules/amr_grid_test_utils.hpp"
#include <array>
#include <cmath>
#include <functional>
#include <memory>
#include <vector>

using namespace shammodels::basegodunov::test_utils;

namespace {

    Tscal dot(const std::vector<Tscal> &a, const std::vector<Tscal> &b) {
        Tscal ret = 0;
        for (u32 i = 0; i < a.size(); i++) {
            ret += a[i] * b[i];
        }
        return ret;
    }

    Tscal dot_abs(const std::vector<Tscal> &a, const std::vector<Tscal> &b) {
        Tscal ret = 0;
        for (u32 i = 0; i < a.size(); i++) {
            ret += std::abs(a[i] * b[i]);
        }
        return ret;
    }

    constexpr Tscal two_pi = 2 * shambase::constants::pi<Tscal>;

    /// f = sin(2 pi x) sin(2 pi y) sin(2 pi z), -Laplacian(f) = 12 pi^2 f
    Tscal sines_3d(Tvec r) {
        return sycl::sin(two_pi * r.x()) * sycl::sin(two_pi * r.y()) * sycl::sin(two_pi * r.z());
    }

    /// f = sin(2 pi x), -Laplacian(f) = 4 pi^2 f, extremal on the level jumps of the test grid
    Tscal sine_x(Tvec r) { return sycl::sin(two_pi * r.x()); }

    /// f = cos(2 pi x), -Laplacian(f) = 4 pi^2 f, extremal in the middle of each level
    Tscal cosine_x(Tvec r) { return sycl::cos(two_pi * r.x()); }

} // namespace

NEW_TEST(Unittest, "shammodels/ramses/modules/NodeSpMVPoisson3D/uniform_convergence", 1) {

    // the operator is -dV Laplacian, so out / dV should converge to 12 pi^2 f
    std::vector<u32> resolutions = {16, 32, 64};
    std::vector<Tscal> errors;

    for (u32 n : resolutions) {
        TestGrid grid = make_uniform_grid(n);
        auto graph    = build_graph(grid);

        std::vector<Tscal> f  = eval_on_grid(grid, sines_3d);
        std::vector<Tscal> Af = apply_spmv(grid, graph, f);

        Tscal err = 0;
        for (u32 c = 0; c < grid.cell_count(); c++) {
            Tscal expected = 3 * two_pi * two_pi * f[c];
            err            = std::max(err, std::abs(Af[c] / grid.volume(c) - expected));
        }
        errors.push_back(err / (3 * two_pi * two_pi));
    }

    for (u32 i = 0; i + 1 < resolutions.size(); i++) {
        Tscal order = std::log2(errors[i] / errors[i + 1]);
        logger::raw_ln(
            shambase::format(
                "uniform grid: n = {} -> {}, relative max error {:.3e} -> {:.3e}, order {:.3f}",
                resolutions[i],
                resolutions[i + 1],
                errors[i],
                errors[i + 1],
                order));
        REQUIRE(order > 1.9);
    }
    REQUIRE(errors.back() < 1e-2);
}

NEW_TEST(Unittest, "shammodels/ramses/modules/NodeSpMVPoisson3D/level_jump_convergence", 1) {

    // The test functions only vary normally to the level jumps.
    // Away from the jumps the stencil is the uniform one and converges at second order (checked
    // with cos(2 pi x), extremal away from the jumps). At the jumps the truncation error of the
    // two-point flux is O(1) but does not grow: with sin(2 pi x), extremal at the jumps, the
    // expected relative errors are 1/4 on the fine side and 1/8 on the coarse side.
    std::vector<u32> resolutions = {32, 64, 128};
    std::vector<Tscal> errors_away;
    std::vector<Tscal> errors_jump;

    for (u32 n : resolutions) {
        TestGrid grid = make_level_jump_grid(n);
        auto graph    = build_graph(grid);

        std::vector<bool> away = get_cells_away_from_jumps(grid, *graph);

        auto get_max_error = [&](const std::function<Tscal(Tvec)> &func, bool away_from_jumps) {
            std::vector<Tscal> f  = eval_on_grid(grid, func);
            std::vector<Tscal> Af = apply_spmv(grid, graph, f);

            Tscal err = 0;
            for (u32 c = 0; c < grid.cell_count(); c++) {
                if (away[c] == away_from_jumps) {
                    Tscal expected = two_pi * two_pi * f[c];
                    err            = std::max(err, std::abs(Af[c] / grid.volume(c) - expected));
                }
            }
            return err;
        };

        Tscal err_away = get_max_error(cosine_x, true);
        Tscal err_jump = get_max_error(sine_x, false);
        errors_away.push_back(err_away / (two_pi * two_pi));
        errors_jump.push_back(err_jump / (two_pi * two_pi));
    }

    for (u32 i = 0; i + 1 < resolutions.size(); i++) {
        Tscal order = std::log2(errors_away[i] / errors_away[i + 1]);
        logger::raw_ln(
            shambase::format(
                "level jump grid: n = {} -> {}, relative max error away from the jumps {:.3e} -> "
                "{:.3e}, order {:.3f}, at the jumps {:.3e} -> {:.3e}",
                resolutions[i],
                resolutions[i + 1],
                errors_away[i],
                errors_away[i + 1],
                order,
                errors_jump[i],
                errors_jump[i + 1]));
        REQUIRE(order > 1.9);
        REQUIRE(errors_jump[i + 1] < 0.26);
    }
}

NEW_TEST(Unittest, "shammodels/ramses/modules/NodeSpMVPoisson3D/weak_form_convergence", 1) {

    // <f, A f> -> integral of -f Laplacian(f) = 12 pi^2 integral of f^2 = 3 pi^2 / 2
    //
    // Second order on a uniform grid. Across the level jumps it is only first order: the two-point
    // flux between a fine and a coarse cell ignores the lateral offset between their centers, which
    // adds about h/6 (df/dy^2 + df/dz^2) of spurious energy per unit area of the jump (h the fine
    // cell size), i.e. a relative error of ~0.44 h here, the O(h^2) part being of the opposite
    // sign (measured: -5.6e-3, 5.7e-3, 4.9e-3, 3.0e-3 for n = 16, 32, 64, 128).
    Tscal expected = 3 * shambase::constants::pi<Tscal> * shambase::constants::pi<Tscal> / 2;

    for (bool level_jump : {false, true}) {
        std::vector<u32> resolutions
            = level_jump ? std::vector<u32>{32, 64, 128} : std::vector<u32>{16, 32, 64};
        std::vector<Tscal> errors;
        for (u32 n : resolutions) {
            TestGrid grid = level_jump ? make_level_jump_grid(n) : make_uniform_grid(n);
            auto graph    = build_graph(grid);

            std::vector<Tscal> f  = eval_on_grid(grid, sines_3d);
            std::vector<Tscal> Af = apply_spmv(grid, graph, f);

            errors.push_back(std::abs(dot(f, Af) - expected) / expected);
        }

        for (u32 i = 0; i + 1 < resolutions.size(); i++) {
            Tscal order = std::log2(errors[i] / errors[i + 1]);
            logger::raw_ln(
                shambase::format(
                    "weak form, {} grid: n = {} -> {}, relative error {:.3e} -> {:.3e}, order "
                    "{:.3f}",
                    level_jump ? "level jump" : "uniform",
                    resolutions[i],
                    resolutions[i + 1],
                    errors[i],
                    errors[i + 1],
                    order));
            if (level_jump) {
                // the first and second order errors only separate on the finest pair
                REQUIRE(errors[i + 1] < 6e-3);
                if (i + 2 == resolutions.size()) {
                    REQUIRE(order > 0.6);
                }
            } else {
                REQUIRE(order > 1.9);
            }
        }
    }
}

NEW_TEST(Unittest, "shammodels/ramses/modules/NodeSpMVPoisson3D/symmetry", 1) {

    // CG requires <x, A y> == <A x, y>
    for (const TestGrid &grid : make_all_test_grids()) {
        auto graph = build_graph(grid);

        std::vector<Tscal> x = shamalgs::primitives::mock_vector<Tscal>(
            0x111, grid.cell_count(), Tscal{-1}, Tscal{1});
        std::vector<Tscal> y = shamalgs::primitives::mock_vector<Tscal>(
            0x222, grid.cell_count(), Tscal{-1}, Tscal{1});

        std::vector<Tscal> Ax = apply_spmv(grid, graph, x);
        std::vector<Tscal> Ay = apply_spmv(grid, graph, y);

        Tscal xAy = dot(x, Ay);
        Tscal Axy = dot(Ax, y);

        logger::raw_ln(
            shambase::format(
                "symmetry, {} cells: <x, A y> = {:.17e}, <A x, y> = {:.17e}",
                grid.cell_count(),
                xAy,
                Axy));

        REQUIRE(std::abs(xAy - Axy) < 1e-12 * dot_abs(x, Ay));

        // A is also positive
        REQUIRE(dot(x, Ax) > 0);
    }
}

NEW_TEST(Unittest, "shammodels/ramses/modules/NodeSpMVPoisson3D/constant_and_nan", 1) {

    for (const TestGrid &grid : make_all_test_grids()) {
        auto graph = build_graph(grid);

        // A applied to a constant field is exactly zero
        std::vector<Tscal> cst(grid.cell_count(), 3.14);
        std::vector<Tscal> Acst = apply_spmv(grid, graph, cst);
        REQUIRE_EQUAL(Acst, std::vector<Tscal>(grid.cell_count(), 0));

        // NaN input values are treated as zeros
        std::vector<Tscal> x = shamalgs::primitives::mock_vector<Tscal>(
            0x333, grid.cell_count(), Tscal{-1}, Tscal{1});
        std::vector<Tscal> x_nan = x;
        for (u32 c = 0; c < grid.cell_count(); c += 7) {
            x[c]     = 0;
            x_nan[c] = std::numeric_limits<Tscal>::quiet_NaN();
        }
        REQUIRE_EQUAL(apply_spmv(grid, graph, x_nan), apply_spmv(grid, graph, x));
    }
}
