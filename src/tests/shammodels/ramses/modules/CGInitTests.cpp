// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright (c) 2021-2026 Timothée David--Cléris <tim.shamrock@proton.me>
// SPDX-License-Identifier: CeCILL Free Software License Agreement v2.1
// Shamrock is licensed under the CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

#include "shambase/constants.hpp"
#include "shambase/memory.hpp"
#include "shamalgs/primitives/mock_vector.hpp"
#include "shammodels/ramses/modules/CGInit.hpp"
#include "shamrock/solvergraph/Field.hpp"
#include "shamrock/solvergraph/Indexes.hpp"
#include "shamrock/solvergraph/ScalarEdge.hpp"
#include "shamtest/shamtest.hpp"
#include "tests/shammodels/ramses/modules/amr_grid_test_utils.hpp"
#include <algorithm>
#include <cmath>
#include <memory>
#include <vector>

using namespace shammodels::basegodunov::test_utils;

namespace {

    /// Diagonal coefficient of the Poisson matrix, as laplacian_7pt_with_Aii computes it
    std::vector<Tscal> get_expected_Aii(const TestGrid &grid, CellGraphEdge &graph) {
        std::vector<Tscal> Aii(grid.cell_count(), 0);
        for (u32 dir = 0; dir < 6; dir++) {
            AMRGraph &g              = shambase::get_check_ref(graph.graph.get(0).graph_links[dir]);
            std::vector<u32> offsets = g.node_link_offset.copy_to_stdvec();
            std::vector<u32> links   = g.node_links.copy_to_stdvec();
            for (u32 c = 0; c < grid.cell_count(); c++) {
                Tscal fac = 1;
                for (u32 l = offsets[c]; l < offsets[c + 1]; l++) {
                    if (grid.cell_size[links[l]] > grid.cell_size[c]) {
                        fac = 3. / 2.;
                    }
                    if (grid.cell_size[links[l]] < grid.cell_size[c]) {
                        fac = 3. / 4.;
                    }
                }
                Tscal dx = Tscal(grid.cell_size[c]) / grid.lattice;
                Aii[c] += 1. / (dx * fac);
            }
        }
        for (u32 c = 0; c < grid.cell_count(); c++) {
            Tscal dx = Tscal(grid.cell_size[c]) / grid.lattice;
            Aii[c] *= dx * dx;
        }
        return Aii;
    }

    Tscal max_abs(const std::vector<Tscal> &v) {
        Tscal ret = 0;
        for (Tscal x : v) {
            ret = std::max(ret, std::abs(x));
        }
        return ret;
    }

} // namespace

NEW_TEST(Unittest, "shammodels/ramses/modules/CGInit", 1) {

    const Tscal fourPiG  = 4 * shambase::constants::pi<Tscal>;
    const Tscal mean_rho = 0.7;

    for (const TestGrid &grid : make_all_test_grids()) {
        auto graph = build_graph(grid);

        std::vector<Tscal> phi = shamalgs::primitives::mock_vector<Tscal>(
            0x123, grid.cell_count(), Tscal{-1}, Tscal{1});
        std::vector<Tscal> rho = shamalgs::primitives::mock_vector<Tscal>(
            0x456, grid.cell_count(), Tscal{0}, Tscal{2});

        auto sizes = std::make_shared<shamrock::solvergraph::Indexes<u32>>("", "");
        sizes->indexes.add_obj(0, grid.block_count());

        auto make_field = [&](u32 nvar) {
            auto f = std::make_shared<shamrock::solvergraph::Field<Tscal>>(nvar, "", "");
            f->ensure_sizes(sizes->indexes);
            return f;
        };

        auto cell_sizes = make_field(1);
        auto field_phi  = make_field(block_size);
        auto field_rho  = make_field(block_size);
        auto phi_res    = make_field(block_size);
        auto phi_p      = make_field(block_size);
        auto rhs        = make_field(block_size);
        auto phi_z      = make_field(block_size);

        cell_sizes->get_buf(0).copy_from_stdvec(grid.block_cell_sizes);
        field_phi->get_buf(0).copy_from_stdvec(phi);
        field_rho->get_buf(0).copy_from_stdvec(rho);

        auto mean_rho_edge   = std::make_shared<shamrock::solvergraph::ScalarEdge<Tscal>>("", "");
        mean_rho_edge->value = mean_rho;

        shammodels::basegodunov::modules::CGInit<Tvec, TgridVec> node(block_size, fourPiG);
        node.set_edges(
            sizes,
            graph,
            cell_sizes,
            field_phi,
            field_rho,
            mean_rho_edge,
            phi_res,
            phi_p,
            rhs,
            phi_z);
        node.evaluate();

        std::vector<Tscal> res_got   = phi_res->get_buf(0).copy_to_stdvec();
        std::vector<Tscal> p_got     = phi_p->get_buf(0).copy_to_stdvec();
        std::vector<Tscal> rhs_got   = rhs->get_buf(0).copy_to_stdvec();
        std::vector<Tscal> phi_z_got = phi_z->get_buf(0).copy_to_stdvec();

        // reference: b = -4 pi G (rho - mean_rho) dV, res = b - A phi, z = res / diag(A)
        std::vector<Tscal> Aphi = apply_spmv(grid, graph, phi);
        std::vector<Tscal> Aii  = get_expected_Aii(grid, *graph);
        std::vector<Tscal> b(grid.cell_count());
        for (u32 c = 0; c < grid.cell_count(); c++) {
            b[c] = -fourPiG * (rho[c] - mean_rho) * grid.volume(c);
        }

        Tscal scale   = max_abs(b) + max_abs(Aphi);
        Tscal err_rhs = 0, err_res = 0, err_z = 0;
        for (u32 c = 0; c < grid.cell_count(); c++) {
            err_rhs = std::max(err_rhs, std::abs(rhs_got[c] - b[c]));
            err_res = std::max(err_res, std::abs(res_got[c] - (b[c] - Aphi[c])));
            err_z   = std::max(err_z, std::abs(phi_z_got[c] * Aii[c] - res_got[c]));
        }

        REQUIRE(err_rhs < 1e-14 * scale);
        REQUIRE(err_res < 1e-14 * scale);
        REQUIRE(err_z < 1e-14 * scale);
        REQUIRE_EQUAL(p_got, res_got);
    }
}
