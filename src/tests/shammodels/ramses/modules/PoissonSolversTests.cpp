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
#include "shambase/string.hpp"
#include "shamalgs/primitives/mock_vector.hpp"
#include "shambackends/DeviceBuffer.hpp"
#include "shamcomm/logs.hpp"
#include "shammodels/ramses/SolverConfig.hpp"
#include "shammodels/ramses/modules/NodeBiCGSTABLoop.hpp"
#include "shammodels/ramses/modules/NodeCGLoop.hpp"
#include "shamrock/solvergraph/DDSharedBuffers.hpp"
#include "shamrock/solvergraph/Field.hpp"
#include "shamrock/solvergraph/Indexes.hpp"
#include "shamrock/solvergraph/RankGetter.hpp"
#include "shamrock/solvergraph/ScalarEdge.hpp"
#include "shamsolvergraph/edge/IDataEdge.hpp"
#include "shamsys/NodeInstance.hpp"
#include "shamtest/shamtest.hpp"
#include "tests/shammodels/ramses/modules/amr_grid_test_utils.hpp"
#include <cmath>
#include <map>
#include <memory>
#include <string>
#include <vector>

using namespace shammodels::basegodunov::test_utils;

namespace {

    enum class SolverKind { CG, BiCGSTAB };

    const Tscal pi      = shambase::constants::pi<Tscal>;
    const Tscal G       = 1;
    const Tscal fourPiG = 4 * pi * G;

    /// rho - mean(rho)
    Tscal rho_pert(Tvec r) {
        return 0.5 * std::sin(2 * pi * r[0]) * std::sin(2 * pi * r[1]) * std::sin(2 * pi * r[2]);
    }

    /// Solution of Delta phi = 4 pi G rho_pert, defined up to a constant
    Tscal phi_exact(Tvec r) { return -fourPiG * rho_pert(r) / (12 * pi * pi); }

    /**
     * @brief Periodic grid stored as a single patch holding the patch blocks then ghost blocks,
     * the ghost blocks being periodic copies of patch blocks
     *
     * The patch blocks are those of `periodic_grid` (same order), the graph is not periodic
     * (patch cells find their neighbours in the ghost blocks).
     */
    struct GhostedGrid {
        TestGrid periodic_grid;
        TestGrid grid;
        u32 offset;
        std::vector<u32> ghost_src_block;
        std::shared_ptr<CellGraphEdge> graph;

        u32 patch_block_count() const { return periodic_grid.block_count(); }

        /// center of cell c, ghost cells are outside of the unit box
        Tvec center(u32 c) const {
            return grid.center(c) - Tvec{1, 1, 1} * (Tscal(offset) / grid.lattice);
        }
    };

    /// Uniform grid of n^3 cells with a layer of one block (2 cells) of ghosts around it
    GhostedGrid make_uniform_ghosted_grid(u32 n) {
        u32 offset = 2;
        u32 extent = n + 2 * offset;

        GhostedGrid ret{make_uniform_grid(n), TestGrid{n, {}, {}, {}}, offset, {}, {}};

        auto is_patch = [&](u32 x) {
            return x >= offset && x < n + offset;
        };

        // patch blocks first, in the order of make_uniform_grid
        std::map<std::array<u32, 3>, u32> patch_block_id;
        for (u32 i = 0; i < n; i += 2) {
            for (u32 j = 0; j < n; j += 2) {
                for (u32 k = 0; k < n; k += 2) {
                    patch_block_id[{i, j, k}] = ret.grid.block_count();
                    ret.grid.add_block({i + offset, j + offset, k + offset}, 1);
                }
            }
        }

        for (u32 i = 0; i < extent; i += 2) {
            for (u32 j = 0; j < extent; j += 2) {
                for (u32 k = 0; k < extent; k += 2) {
                    if (is_patch(i) && is_patch(j) && is_patch(k)) {
                        continue;
                    }
                    ret.grid.add_block({i, j, k}, 1);
                    auto wrap = [&](u32 x) {
                        return (x + n - offset) % n;
                    };
                    ret.ghost_src_block.push_back(patch_block_id.at({wrap(i), wrap(j), wrap(k)}));
                }
            }
        }

        ret.graph = build_graph(ret.grid, extent, false);
        return ret;
    }

    /// Poisson problem on a single patch, with its ghost blocks
    struct PoissonProblem {
        std::shared_ptr<shamrock::solvergraph::Indexes<u32>> sizes;
        std::shared_ptr<shamrock::solvergraph::Indexes<u32>> sizes_no_gz;
        std::shared_ptr<CellGraphEdge> graph;
        std::shared_ptr<shamrock::solvergraph::Field<Tscal>> cell_sizes;
        std::shared_ptr<shamrock::solvergraph::Field<Tscal>> rho;
        std::shared_ptr<shamrock::solvergraph::ScalarEdge<Tscal>> mean_rho;
        std::shared_ptr<shamrock::solvergraph::DDSharedBuffers<u32>> idx_in_ghost;
        std::shared_ptr<shamrock::solvergraph::RankGetter> rank_owner;
        std::shared_ptr<shamrock::solvergraph::Field<Tscal>> phi;
    };

    /**
     * @brief Build the problem
     *
     * @param grid all the blocks (patch blocks first)
     * @param graph neighbour graph of grid
     * @param patch_block_count number of patch blocks
     * @param ghost_src_block for each ghost block, the patch block it copies
     * @param rho density of every cell
     * @param phi0 initial guess of every cell
     */
    PoissonProblem make_problem(
        const TestGrid &grid,
        const std::shared_ptr<CellGraphEdge> &graph,
        u32 patch_block_count,
        const std::vector<u32> &ghost_src_block,
        const std::vector<Tscal> &rho,
        const std::vector<Tscal> &phi0) {

        PoissonProblem pb;

        pb.sizes = std::make_shared<shamrock::solvergraph::Indexes<u32>>("", "");
        pb.sizes->indexes.add_obj(0, grid.block_count());
        pb.sizes_no_gz = std::make_shared<shamrock::solvergraph::Indexes<u32>>("", "");
        pb.sizes_no_gz->indexes.add_obj(0, u32{patch_block_count});

        pb.graph = graph;

        auto make_field = [&](u32 nvar) {
            auto f = std::make_shared<shamrock::solvergraph::Field<Tscal>>(nvar, "", "");
            f->ensure_sizes(pb.sizes->indexes);
            return f;
        };

        pb.cell_sizes = make_field(1);
        pb.cell_sizes->get_buf(0).copy_from_stdvec(grid.block_cell_sizes);
        pb.rho = make_field(block_size);
        pb.rho->get_buf(0).copy_from_stdvec(rho);
        pb.phi = make_field(block_size);
        pb.phi->get_buf(0).copy_from_stdvec(phi0);

        // mean over the patch cells
        Tscal sum_m = 0, sum_v = 0;
        for (u32 c = 0; c < patch_block_count * block_size; c++) {
            sum_m += rho[c] * grid.volume(c);
            sum_v += grid.volume(c);
        }
        pb.mean_rho        = std::make_shared<shamrock::solvergraph::ScalarEdge<Tscal>>("", "");
        pb.mean_rho->value = sum_m / sum_v;

        // the patch is its own ghost provider (periodic copies)
        pb.idx_in_ghost = std::make_shared<shamrock::solvergraph::DDSharedBuffers<u32>>("", "");
        sham::DeviceBuffer<u32> idx(
            ghost_src_block.size(), shamsys::instance::get_compute_scheduler_ptr());
        idx.copy_from_stdvec(ghost_src_block);
        pb.idx_in_ghost->buffers.add_obj(0, 0, std::move(idx));

        pb.rank_owner = std::make_shared<shamrock::solvergraph::RankGetter>(
            [](u64) -> u32 {
                return 0;
            },
            "",
            "");

        return pb;
    }

    /// set the ghost cells of v to the value of the patch cells they copy
    void fill_ghosts(const GhostedGrid &g, std::vector<Tscal> &v) {
        u32 patch_block_count = g.patch_block_count();
        for (u32 gb = 0; gb < g.ghost_src_block.size(); gb++) {
            for (u32 i = 0; i < block_size; i++) {
                v[(patch_block_count + gb) * block_size + i]
                    = v[g.ghost_src_block[gb] * block_size + i];
            }
        }
    }

    PoissonProblem make_problem(const GhostedGrid &g, const std::vector<Tscal> &phi0) {
        std::vector<Tscal> rho(g.grid.cell_count());
        for (u32 c = 0; c < g.grid.cell_count(); c++) {
            rho[c] = 1 + rho_pert(g.center(c));
        }
        return make_problem(g.grid, g.graph, g.patch_block_count(), g.ghost_src_block, rho, phi0);
    }

    struct SolveResult {
        std::vector<Tscal> phi;
        u32 iterations;
    };

    SolveResult solve(SolverKind kind, PoissonProblem &pb, u32 Niter_max, Tscal tol) {
        using namespace shammodels::basegodunov::modules;

        auto field = [&]() {
            return std::make_shared<shamrock::solvergraph::Field<Tscal>>(block_size, "", "");
        };
        auto scal = [&]() {
            return shamrock::solvergraph::IDataEdge<Tscal>::make_shared("", "");
        };

        SolveResult ret{};

        if (kind == SolverKind::CG) {
            NodeCGLoop<Tvec, TgridVec> node(block_size, fourPiG, Niter_max, tol);
            node.set_edges(
                pb.sizes,
                pb.sizes_no_gz,
                pb.graph,
                pb.cell_sizes,
                pb.rho,
                pb.mean_rho,
                pb.idx_in_ghost,
                pb.rank_owner,
                pb.phi,
                field(),
                field(),
                field(),
                field(),
                field(),
                field(),
                scal(),
                scal(),
                scal(),
                scal(),
                scal(),
                scal(),
                scal(),
                scal(),
                field());
            node.evaluate();
            ret.iterations = node.get_last_iteration_count();
        } else {
            // default lucky breakdown tolerance of the config
            Tscal tol_happy_bk = shammodels::basegodunov::GravityConfig<Tvec>{}.tol_hp_bk;
            NodeBiCGSTABLoop<Tvec, TgridVec> node(
                block_size, fourPiG, Niter_max, tol, tol_happy_bk);
            node.set_edges(
                pb.sizes,
                pb.sizes_no_gz,
                pb.graph,
                pb.cell_sizes,
                pb.rho,
                pb.mean_rho,
                pb.idx_in_ghost,
                pb.rank_owner,
                pb.phi,
                field(),
                field(),
                field(),
                field(),
                field(),
                field(),
                field(),
                scal(),
                scal(),
                scal(),
                scal(),
                scal(),
                scal(),
                scal(),
                field(),
                scal());
            node.evaluate();
            ret.iterations = node.get_last_iteration_count();
        }

        ret.phi = pb.phi->get_buf(0).copy_to_stdvec();
        return ret;
    }

    Tscal l2(const std::vector<Tscal> &v) {
        Tscal ret = 0;
        for (Tscal x : v) {
            ret += x * x;
        }
        return std::sqrt(ret);
    }

    /// b - A phi on the periodic grid (patch cells only)
    std::vector<Tscal> true_residual(
        const TestGrid &periodic_grid,
        const std::vector<Tscal> &rho,
        Tscal mean_rho,
        const std::vector<Tscal> &phi,
        std::vector<Tscal> &b) {
        auto graph              = build_graph(periodic_grid);
        u32 ncell               = periodic_grid.cell_count();
        std::vector<Tscal> phi_ = std::vector<Tscal>(phi.begin(), phi.begin() + ncell);
        std::vector<Tscal> Aphi = apply_spmv(periodic_grid, graph, phi_);
        std::vector<Tscal> r(ncell);
        b.resize(ncell);
        for (u32 c = 0; c < ncell; c++) {
            b[c] = -fourPiG * (rho[c] - mean_rho) * periodic_grid.volume(c);
            r[c] = b[c] - Aphi[c];
        }
        return r;
    }

    std::string solver_name(SolverKind kind) {
        return (kind == SolverKind::CG) ? "CG" : "BiCGSTAB";
    }

    void test_solver_uniform(SolverKind kind) {
        const Tscal tol     = 1e-8;
        const u32 Niter_max = 1000;

        std::vector<Tscal> errors;
        for (u32 n : {8, 16, 32}) {
            GhostedGrid g = make_uniform_ghosted_grid(n);

            // random initial guess, so that the solver has to iterate
            std::vector<Tscal> phi0 = shamalgs::primitives::mock_vector<Tscal>(
                0x111 + n, g.grid.cell_count(), Tscal{-1e-2}, Tscal{1e-2});
            // the ghosts of phi are exchanged before the solve in the solver
            fill_ghosts(g, phi0);

            PoissonProblem pb = make_problem(g, phi0);
            SolveResult res   = solve(kind, pb, Niter_max, tol);

            // determinism: a second identical solve gives the same bits
            PoissonProblem pb2 = make_problem(g, phi0);
            SolveResult res2   = solve(kind, pb2, Niter_max, tol);
            REQUIRE_EQUAL_NAMED(solver_name(kind) + " deterministic phi", res.phi, res2.phi);
            REQUIRE_EQUAL_NAMED(
                solver_name(kind) + " deterministic iteration count",
                res.iterations,
                res2.iterations);

            REQUIRE_NAMED(
                solver_name(kind) + " converged before Niter_max", res.iterations < Niter_max);

            // the residual recomputed independently reaches the tolerance
            std::vector<Tscal> rho = pb.rho->get_buf(0).copy_to_stdvec();
            std::vector<Tscal> b;
            std::vector<Tscal> r
                = true_residual(g.periodic_grid, rho, pb.mean_rho->value, res.phi, b);
            Tscal res_norm = l2(r) / l2(b);
            REQUIRE_NAMED(solver_name(kind) + " residual below tolerance", res_norm < 10 * tol);

            // error vs the analytic solution, up to the additive constant
            u32 ncell        = g.periodic_grid.cell_count();
            Tscal mean_shift = 0;
            for (u32 c = 0; c < ncell; c++) {
                mean_shift += res.phi[c] - phi_exact(g.center(c));
            }
            mean_shift /= ncell;

            Tscal err = 0, ref = 0;
            for (u32 c = 0; c < ncell; c++) {
                Tscal e = res.phi[c] - mean_shift - phi_exact(g.center(c));
                err += e * e;
                ref += phi_exact(g.center(c)) * phi_exact(g.center(c));
            }
            err = std::sqrt(err / ref);
            errors.push_back(err);

            logger::raw_ln(
                sham::format(
                    "{} n = {:3} : {:4} iterations, residual = {:.3e}, relative L2 error = {:.3e}",
                    solver_name(kind),
                    n,
                    res.iterations,
                    res_norm,
                    err));
        }

        for (u32 i = 0; i + 1 < errors.size(); i++) {
            Tscal order = std::log2(errors[i] / errors[i + 1]);
            logger::raw_ln(sham::format("{} convergence order = {:.3f}", solver_name(kind), order));
            REQUIRE_NAMED(solver_name(kind) + " second order", order > 1.9 && order < 2.1);
        }
    }

    /**
     * Level jump grid: check that the solver reaches the tolerance and that the independently
     * recomputed residual does as well.
     */
    void test_solver_level_jump(SolverKind kind) {
        const Tscal tol     = 1e-8;
        const u32 Niter_max = 1000;

        TestGrid grid = make_level_jump_grid(16);
        auto graph    = build_graph(grid);

        std::vector<Tscal> rho(grid.cell_count());
        for (u32 c = 0; c < grid.cell_count(); c++) {
            rho[c] = 1 + rho_pert(grid.center(c));
        }
        std::vector<Tscal> phi0(grid.cell_count(), 0);

        // periodic graph, no ghost block
        PoissonProblem pb = make_problem(grid, graph, grid.block_count(), {}, rho, phi0);
        SolveResult res   = solve(kind, pb, Niter_max, tol);

        REQUIRE_NAMED(
            solver_name(kind) + " level jump converged before Niter_max",
            res.iterations < Niter_max);

        std::vector<Tscal> b;
        std::vector<Tscal> r = true_residual(grid, rho, pb.mean_rho->value, res.phi, b);
        Tscal res_norm       = l2(r) / l2(b);

        logger::raw_ln(
            sham::format(
                "{} level jump grid : {:4} iterations, residual = {:.3e}",
                solver_name(kind),
                res.iterations,
                res_norm));

        REQUIRE_NAMED(solver_name(kind) + " level jump residual", res_norm < 10 * tol);
    }

} // namespace

NEW_TEST(Unittest, "shammodels/ramses/modules/NodeCGLoop", 1) {
    test_solver_uniform(SolverKind::CG);
    test_solver_level_jump(SolverKind::CG);
}

NEW_TEST(Unittest, "shammodels/ramses/modules/NodeBiCGSTABLoop", 1) {
    test_solver_uniform(SolverKind::BiCGSTAB);
    test_solver_level_jump(SolverKind::BiCGSTAB);
}
