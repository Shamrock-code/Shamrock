// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright (c) 2021-2026 Timothée David--Cléris <tim.shamrock@proton.me>
// SPDX-License-Identifier: CeCILL Free Software License Agreement v2.1
// Shamrock is licensed under the CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

#include "shambase/constants.hpp"
#include "shambase/string.hpp"
#include "shamcomm/logs.hpp"
#include "shammodels/ramses/modules/NodeSelfGravityAcceleration.hpp"
#include "shamrock/solvergraph/Field.hpp"
#include "shamrock/solvergraph/Indexes.hpp"
#include "shamtest/shamtest.hpp"
#include "tests/shammodels/ramses/modules/amr_grid_test_utils.hpp"
#include <algorithm>
#include <cmath>
#include <functional>
#include <memory>
#include <vector>

using namespace shammodels::basegodunov::test_utils;

namespace {

    const Tscal pi = shambase::constants::pi<Tscal>;

    std::vector<Tvec> compute_g(
        const TestGrid &grid,
        const std::shared_ptr<CellGraphEdge> &graph,
        const std::vector<Tscal> &phi) {

        auto sizes = std::make_shared<shamrock::solvergraph::Indexes<u32>>("", "");
        sizes->indexes.add_obj(0, grid.block_count());

        auto cell_sizes = std::make_shared<shamrock::solvergraph::Field<Tscal>>(1, "", "");
        auto field_phi  = std::make_shared<shamrock::solvergraph::Field<Tscal>>(block_size, "", "");
        auto field_g    = std::make_shared<shamrock::solvergraph::Field<Tvec>>(block_size, "", "");

        cell_sizes->ensure_sizes(sizes->indexes);
        field_phi->ensure_sizes(sizes->indexes);
        cell_sizes->get_buf(0).copy_from_stdvec(grid.block_cell_sizes);
        field_phi->get_buf(0).copy_from_stdvec(phi);

        shammodels::basegodunov::modules::NodeSelfGravityAcceleration<Tvec, TgridVec> node(
            block_size);
        node.set_edges(sizes, graph, cell_sizes, field_phi, field_g);
        node.evaluate();

        return field_g->get_buf(0).copy_to_stdvec();
    }

    struct Errors {
        Tscal away_from_jumps = 0; ///< max error on the cells away from the level jumps
        Tscal at_jumps        = 0; ///< max error on the cells at a level jump
    };

    Errors get_errors(
        const TestGrid &grid,
        const std::function<Tscal(Tvec)> &phi,
        const std::function<Tvec(Tvec)> &minus_grad_phi) {
        auto graph             = build_graph(grid);
        std::vector<Tvec> g    = compute_g(grid, graph, eval_on_grid(grid, phi));
        std::vector<bool> away = get_cells_away_from_jumps(grid, *graph);
        Errors ret{};
        for (u32 c = 0; c < grid.cell_count(); c++) {
            Tvec d  = g[c] - minus_grad_phi(grid.center(c));
            Tscal e = std::max({std::abs(d[0]), std::abs(d[1]), std::abs(d[2])});
            if (away[c]) {
                ret.away_from_jumps = std::max(ret.away_from_jumps, e);
            } else {
                ret.at_jumps = std::max(ret.at_jumps, e);
            }
        }
        return ret;
    }

    Tscal phi_3d(Tvec r) {
        return std::sin(2 * pi * r[0]) * std::cos(2 * pi * r[1]) * std::sin(2 * pi * r[2]);
    }

    Tvec minus_grad_phi_3d(Tvec r) {
        Tscal sx = std::sin(2 * pi * r[0]), cx = std::cos(2 * pi * r[0]);
        Tscal sy = std::sin(2 * pi * r[1]), cy = std::cos(2 * pi * r[1]);
        Tscal sz = std::sin(2 * pi * r[2]), cz = std::cos(2 * pi * r[2]);
        return -Tvec{2 * pi * cx * cy * sz, -2 * pi * sx * sy * sz, 2 * pi * sx * cy * cz};
    }

    /// depends only on x, the direction normal to the level jumps of make_level_jump_grid
    Tscal phi_x(Tvec r) { return std::sin(2 * pi * r[0]); }

    Tvec minus_grad_phi_x(Tvec r) { return -Tvec{2 * pi * std::cos(2 * pi * r[0]), 0, 0}; }

    Tscal order(Tscal e_coarse, Tscal e_fine) { return std::log2(e_coarse / e_fine); }

} // namespace

NEW_TEST(Unittest, "shammodels/ramses/modules/NodeSelfGravityAcceleration", 1) {

    // uniform grid: centered difference, second order
    {
        std::vector<Tscal> err;
        for (u32 n : {16, 32, 64}) {
            err.push_back(
                get_errors(make_uniform_grid(n), phi_3d, minus_grad_phi_3d).away_from_jumps);
        }
        for (u32 i = 0; i + 1 < err.size(); i++) {
            Tscal o = order(err[i], err[i + 1]);
            logger::raw_ln(
                sham::format("g uniform grid : max error {:.3e}, order {:.3f}", err[i + 1], o));
            REQUIRE_NAMED("g uniform grid second order", o > 1.9 && o < 2.1);
        }
    }

    // 2:1 level jumps normal to x, phi(x): second order away from the jumps, first order at the
    // jumps, where the distances between the cell centers (3/2 dx and 3/4 dx) differ on the two
    // sides of the cell
    {
        std::vector<Errors> err;
        for (u32 n : {16, 32, 64}) {
            err.push_back(get_errors(make_level_jump_grid(n), phi_x, minus_grad_phi_x));
        }
        for (u32 i = 0; i + 1 < err.size(); i++) {
            Tscal o_away = order(err[i].away_from_jumps, err[i + 1].away_from_jumps);
            Tscal o_jump = order(err[i].at_jumps, err[i + 1].at_jumps);
            logger::raw_ln(
                sham::format(
                    "g level jump, phi(x) : away {:.3e} (order {:.3f}), at jumps {:.3e} (order "
                    "{:.3f})",
                    err[i + 1].away_from_jumps,
                    o_away,
                    err[i + 1].at_jumps,
                    o_jump));
            REQUIRE_NAMED("g level jump phi(x) second order away", o_away > 1.85 && o_away < 2.15);
            REQUIRE_NAMED("g level jump phi(x) first order at jumps", o_jump > 0.9 && o_jump < 1.1);
        }
    }

    // 2:1 level jumps with a 3D phi: the coarse neighbour of a fine cell is laterally offset by
    // dx/2, so the gradient at the jumps does not converge (as for laplacian_7pt, see
    // NodeSpMVPoisson3DTests.cpp); it stays bounded
    {
        std::vector<Errors> err;
        for (u32 n : {16, 32, 64}) {
            err.push_back(get_errors(make_level_jump_grid(n), phi_3d, minus_grad_phi_3d));
        }
        for (u32 i = 0; i + 1 < err.size(); i++) {
            Tscal o_away = order(err[i].away_from_jumps, err[i + 1].away_from_jumps);
            Tscal o_jump = order(err[i].at_jumps, err[i + 1].at_jumps);
            logger::raw_ln(
                sham::format(
                    "g level jump, 3D phi : away {:.3e} (order {:.3f}), at jumps {:.3e} (order "
                    "{:.3f})",
                    err[i + 1].away_from_jumps,
                    o_away,
                    err[i + 1].at_jumps,
                    o_jump));
            // 16 -> 32 is not yet in the asymptotic regime (order ~1.7)
            if (i + 2 == err.size()) {
                REQUIRE_NAMED(
                    "g level jump 3D phi second order away", o_away > 1.85 && o_away < 2.15);
            }
            REQUIRE_NAMED("g level jump 3D phi bounded at jumps", o_jump > -0.1);
        }
    }
}
