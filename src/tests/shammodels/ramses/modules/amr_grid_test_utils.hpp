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
 * @file amr_grid_test_utils.hpp
 * @author Timothée David--Cléris (tim.shamrock@proton.me)
 * @brief Helpers building periodic AMR grids and their neighbour graph by hand for the Poisson
 * solver node tests.
 */

#include "shambase/DistributedData.hpp"
#include "shambase/memory.hpp"
#include "shambackends/DeviceBuffer.hpp"
#include "shambackends/vec.hpp"
#include "shammodels/common/amr/NeighGraph.hpp"
#include "shammodels/ramses/modules/NodeSpMVPoisson3D.hpp"
#include "shammodels/ramses/solvegraph/OrientedAMRGraphEdge.hpp"
#include "shamrock/solvergraph/Field.hpp"
#include "shamrock/solvergraph/Indexes.hpp"
#include "shamsys/NodeInstance.hpp"
#include <array>
#include <functional>
#include <memory>
#include <vector>

namespace shammodels::basegodunov::test_utils {

    using Tvec     = f64_3;
    using Tscal    = f64;
    using TgridVec = i64_3;

    using AMRGraph         = shammodels::basegodunov::modules::AMRGraph;
    using OrientedAMRGraph = shammodels::basegodunov::modules::OrientedAMRGraph<Tvec, TgridVec>;
    using CellGraphEdge
        = shammodels::basegodunov::solvergraph::OrientedAMRGraphEdge<Tvec, TgridVec>;

    inline constexpr u32 block_size = 8; // 2x2x2 cells per block

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
    inline TestGrid make_uniform_grid(u32 n) {
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
    inline TestGrid make_level_jump_grid(u32 n) {
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
    inline std::shared_ptr<CellGraphEdge> build_graph(const TestGrid &grid) {
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
    inline std::vector<bool> get_cells_away_from_jumps(const TestGrid &grid, CellGraphEdge &graph) {
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
    inline std::vector<Tscal> apply_spmv(
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

    inline std::vector<Tscal> eval_on_grid(
        const TestGrid &grid, const std::function<Tscal(Tvec)> &f) {
        std::vector<Tscal> ret(grid.cell_count());
        for (u32 c = 0; c < grid.cell_count(); c++) {
            ret[c] = f(grid.center(c));
        }
        return ret;
    }

    /// Uniform and level jump test grids of 16^3 lattice cells
    inline std::vector<TestGrid> make_all_test_grids() {
        return {make_uniform_grid(16), make_level_jump_grid(16)};
    }

} // namespace shammodels::basegodunov::test_utils
