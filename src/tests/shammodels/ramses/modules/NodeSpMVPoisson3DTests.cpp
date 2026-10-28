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
#include "shamsys/NodeInstance.hpp"
#include "shamtest/shamtest.hpp"
#include <array>
#include <cmath>
#include <functional>
#include <memory>
#include <vector>

namespace {

    using Tvec     = f64_3;
    using Tscal    = f64;
    using TgridVec = i64_3;

    using AMRGraph         = shammodels::basegodunov::modules::AMRGraph;
    using OrientedAMRGraph = shammodels::basegodunov::modules::OrientedAMRGraph<Tvec, TgridVec>;
    using CellGraphEdge
        = shammodels::basegodunov::solvergraph::OrientedAMRGraphEdge<Tvec, TgridVec>;

    constexpr u32 block_size = 8; // 2x2x2 cells per block

    /**
     * @brief Periodic AMR grid of the unit box built on the host.
     *
     * Cell positions and sizes are stored in units of a lattice of `lattice` cells per dimension
     * (the finest level), each block holds 2x2x2 cells of the same size.
     */
    struct TestGrid {
        u32 lattice;
        std::vector<std::array<u32, 3>> cell_lo;
        std::vector<u32> cell_size;
        std::vector<Tscal> block_cell_sizes;

        void add_block(std::array<u32, 3> lo, u32 s) {
            for (u32 i = 0; i < 2; i++) {
                for (u32 j = 0; j < 2; j++) {
                    for (u32 k = 0; k < 2; k++) {
                        cell_lo.push_back({lo[0] + i * s, lo[1] + j * s, lo[2] + k * s});
                        cell_size.push_back(s);
                    }
                }
            }
            block_cell_sizes.push_back(Tscal(s) / lattice);
        }

        u32 cell_count() const { return cell_lo.size(); }
        u32 block_count() const { return block_cell_sizes.size(); }

        Tvec center(u32 c) const {
            Tscal hs = Tscal(cell_size[c]) / 2;
            return Tvec{cell_lo[c][0] + hs, cell_lo[c][1] + hs, cell_lo[c][2] + hs} / lattice;
        }

        Tscal volume(u32 c) const {
            Tscal dx = Tscal(cell_size[c]) / lattice;
            return dx * dx * dx;
        }
    };

    /// Uniform grid of n^3 cells (n even)
    TestGrid make_uniform_grid(u32 n) {
        TestGrid grid{n, {}, {}, {}};
        for (u32 i = 0; i < n; i += 2) {
            for (u32 j = 0; j < n; j += 2) {
                for (u32 k = 0; k < n; k += 2) {
                    grid.add_block({i, j, k}, 1);
                }
            }
        }
        return grid;
    }

    /**
     * @brief Grid of (n/2)^3 coarse cells refined once (2:1) in the slab 1/4 <= x < 3/4
     *
     * The level jumps are the planes x = 1/4 and x = 3/4 (n must be a multiple of 8).
     */
    TestGrid make_level_jump_grid(u32 n) {
        TestGrid grid{n, {}, {}, {}};
        auto is_fine = [&](u32 x) {
            return x >= n / 4 && x < 3 * n / 4;
        };
        for (u32 i = 0; i < n;) {
            u32 s = is_fine(i) ? 1 : 2;
            for (u32 j = 0; j < n; j += 2 * s) {
                for (u32 k = 0; k < n; k += 2 * s) {
                    grid.add_block({i, j, k}, s);
                }
            }
            i += 2 * s;
        }
        return grid;
    }

    /// Build the face neighbour graph of the grid (periodic boundaries)
    std::shared_ptr<CellGraphEdge> build_graph(const TestGrid &grid) {
        u32 L = grid.lattice;

        std::vector<u32> owner(L * L * L);
        for (u32 c = 0; c < grid.cell_count(); c++) {
            u32 s = grid.cell_size[c];
            for (u32 i = 0; i < s; i++) {
                for (u32 j = 0; j < s; j++) {
                    for (u32 k = 0; k < s; k++) {
                        u32 x                      = grid.cell_lo[c][0] + i;
                        u32 y                      = grid.cell_lo[c][1] + j;
                        u32 z                      = grid.cell_lo[c][2] + k;
                        owner[(x * L + y) * L + z] = c;
                    }
                }
            }
        }

        auto dev_sched = shamsys::instance::get_compute_scheduler_ptr();

        OrientedAMRGraph oriented_graph;
        for (u32 dir = 0; dir < 6; dir++) {
            u32 axis      = dir / 2;
            bool positive = (dir % 2) == 0;
            u32 t1        = (axis + 1) % 3;
            u32 t2        = (axis + 2) % 3;

            std::vector<u32> offsets{0};
            std::vector<u32> links;
            for (u32 c = 0; c < grid.cell_count(); c++) {
                u32 s = grid.cell_size[c];
                std::array<u32, 3> pos{};
                pos[axis]
                    = (positive ? grid.cell_lo[c][axis] + s : grid.cell_lo[c][axis] + L - 1) % L;

                u32 start = links.size();
                for (u32 a = 0; a < s; a++) {
                    for (u32 b = 0; b < s; b++) {
                        pos[t1]    = grid.cell_lo[c][t1] + a;
                        pos[t2]    = grid.cell_lo[c][t2] + b;
                        u32 o      = owner[(pos[0] * L + pos[1]) * L + pos[2]];
                        bool found = false;
                        for (u32 l = start; l < links.size(); l++) {
                            found = found || (links[l] == o);
                        }
                        if (!found) {
                            links.push_back(o);
                        }
                    }
                }
                offsets.push_back(links.size());
            }

            sham::DeviceBuffer<u32> buf_offsets(offsets.size(), dev_sched);
            sham::DeviceBuffer<u32> buf_links(links.size(), dev_sched);
            buf_offsets.copy_from_stdvec(offsets);
            buf_links.copy_from_stdvec(links);

            u32 link_count                  = links.size();
            oriented_graph.graph_links[dir] = std::make_unique<AMRGraph>(AMRGraph{
                std::move(buf_offsets), std::move(buf_links), link_count, grid.cell_count()});
        }

        auto graph = std::make_shared<CellGraphEdge>("", "");
        graph->graph.add_obj(0, std::move(oriented_graph));
        return graph;
    }

    /// Tells whether all the neighbours of cell c have the same size as c
    std::vector<bool> get_cells_away_from_jumps(const TestGrid &grid, CellGraphEdge &graph) {
        std::vector<bool> ret(grid.cell_count(), true);
        for (u32 dir = 0; dir < 6; dir++) {
            AMRGraph &g              = shambase::get_check_ref(graph.graph.get(0).graph_links[dir]);
            std::vector<u32> offsets = g.node_link_offset.copy_to_stdvec();
            std::vector<u32> links   = g.node_links.copy_to_stdvec();
            for (u32 c = 0; c < grid.cell_count(); c++) {
                for (u32 l = offsets[c]; l < offsets[c + 1]; l++) {
                    if (grid.cell_size[links[l]] != grid.cell_size[c]) {
                        ret[c] = false;
                    }
                }
            }
        }
        return ret;
    }

    /// Apply NodeSpMVPoisson3D to x
    std::vector<Tscal> apply_spmv(
        const TestGrid &grid,
        const std::shared_ptr<CellGraphEdge> &graph,
        const std::vector<Tscal> &x) {

        auto sizes = std::make_shared<shamrock::solvergraph::Indexes<u32>>("", "");
        sizes->indexes.add_obj(0, grid.block_count());

        auto cell_sizes = std::make_shared<shamrock::solvergraph::Field<Tscal>>(1, "", "");
        auto in         = std::make_shared<shamrock::solvergraph::Field<Tscal>>(block_size, "", "");
        auto out        = std::make_shared<shamrock::solvergraph::Field<Tscal>>(block_size, "", "");

        cell_sizes->ensure_sizes(sizes->indexes);
        in->ensure_sizes(sizes->indexes);
        cell_sizes->get_buf(0).copy_from_stdvec(grid.block_cell_sizes);
        in->get_buf(0).copy_from_stdvec(x);

        shammodels::basegodunov::modules::NodeSpMVPoisson3D<Tvec, TgridVec> node(block_size);
        node.set_edges(sizes, graph, cell_sizes, in, out);
        node.evaluate();

        return out->get_buf(0).copy_to_stdvec();
    }

    std::vector<Tscal> eval_on_grid(const TestGrid &grid, const std::function<Tscal(Tvec)> &f) {
        std::vector<Tscal> ret(grid.cell_count());
        for (u32 c = 0; c < grid.cell_count(); c++) {
            ret[c] = f(grid.center(c));
        }
        return ret;
    }

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

    std::vector<TestGrid> make_all_test_grids() {
        return {make_uniform_grid(16), make_level_jump_grid(16)};
    }

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
