// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright (c) 2021-2026 Timothée David--Cléris <tim.shamrock@proton.me>
// SPDX-License-Identifier: CeCILL Free Software License Agreement v2.1
// Shamrock is licensed under the CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

/**
 * @file NodeCGLoop.cpp
 * @author Léodasce Sewanou (leodasce.sewanou@ens-lyon.fr)
 * @author Timothée David--Cléris (tim.shamrock@proton.me)
 * @brief Jacobi preconditioned conjugate gradient solver of the Poisson equation for the
 * self-gravity.
 *
 * Adapted from the PhD work of Léodasce Sewanou.
 */

#include "shambase/stacktrace.hpp"
#include "shambase/string.hpp"
#include "shamcomm/logs.hpp"
#include "shamcomm/worldInfo.hpp"
#include "shammodels/ramses/modules/NodeCGLoop.hpp"
#include <cmath>

namespace shammodels::basegodunov::modules {

    template<class Tvec, class TgridVec>
    void NodeCGLoop<Tvec, TgridVec>::_impl_evaluate_internal() {
        StackEntry stack_loc{};
        auto edges = get_edges();

        edges.spans_block_cell_sizes.check_sizes(edges.sizes.indexes);
        edges.spans_phi.check_sizes(edges.sizes.indexes);
        edges.spans_rho.check_sizes(edges.sizes.indexes);
        edges.spans_phi_res.ensure_sizes(edges.sizes.indexes);
        edges.spans_phi_p.ensure_sizes(edges.sizes.indexes);
        edges.spans_phi_Ap.ensure_sizes(edges.sizes.indexes);
        edges.spans_phi_hadamard_prod.ensure_sizes(edges.sizes.indexes);
        edges.spans_phi_rhs.ensure_sizes(edges.sizes.indexes);
        edges.spans_phi_z.ensure_sizes(edges.sizes.indexes);
        edges.spans_rz_hadamard_prod.ensure_sizes(edges.sizes.indexes);

        // r0 = b - A phi0, with b = -4 pi G (rho - mean_rho) dV, and z0 = p0 = r0 / A_ii
        cg_init_node.evaluate();

        // <r0, r0>
        res_ddot_node.evaluate();

        // <b, b>
        rhs_node.evaluate();

        // <r0, z0>
        rz_hadamard_prod_node.evaluate();
        rz_reduction_node.evaluate();

        shamlog_debug_ln(
            "Ramses::CG",
            sham::format(
                "k = 0, <r,r> = {}, <r,z> = {}, <b,b> = {}",
                edges.old_values.data,
                edges.rz_old_values.data,
                edges.rhs_norm_values.data));

        // TODO: (kept from #1942) if <p, A p> == 0 (e.g. r0 == 0 when phi0 is already the
        // solution, or a uniform density with a uniform phi0), alpha = 0 / 0 is NaN and phi becomes
        // NaN. Likewise, <b, b> == 0 (uniform density) makes the convergence test always false.

        u32 k          = 0;
        bool converged = false;
        while (k < Niter_max) {
            k = k + 1;

            // exchange the ghost blocks of p_k (on them the stencil is incomplete, so the update of
            // p_k at the previous iteration did not give the right values)
            node_gz_p.evaluate();
            node_exch_gz_p.evaluate();
            node_replace_gz_p.evaluate();

            // A p_k
            spmv_node.evaluate();

            // <p_k, A p_k>
            hadamard_prod_node.evaluate();
            a_norm_node.evaluate();

            // alpha_k = <r_k, z_k> / <p_k, A p_k>
            edges.alpha.data = edges.rz_old_values.data / edges.e_norm.data;

            // phi_{k+1} = phi_k + alpha_k p_k
            new_potential_node.evaluate();

            // r_{k+1} = r_k - alpha_k A p_k
            edges.alpha.data = -edges.alpha.data;
            new_residual_node.evaluate();

            // z_{k+1} = M^-1 r_{k+1}
            res_precond_node.evaluate();

            // <r_{k+1}, z_{k+1}>
            rz_hadamard_prod_node.evaluate();
            rz_new_reduction_node.evaluate();

            // <r_{k+1}, r_{k+1}>
            res_ddot_new_node.evaluate();

            // beta_k = <r_{k+1}, z_{k+1}> / <r_k, z_k>
            edges.beta.data          = edges.rz_new_values.data / edges.rz_old_values.data;
            edges.old_values.data    = edges.new_values.data;
            edges.rz_old_values.data = edges.rz_new_values.data;

            shamlog_debug_ln(
                "Ramses::CG",
                sham::format(
                    "k = {}, <r,r> = {}, <r,z> = {}",
                    k,
                    edges.old_values.data,
                    edges.rz_old_values.data));

            // p_{k+1} = z_{k+1} + beta_k p_k
            new_p_node.evaluate();

            if ((edges.old_values.data / edges.rhs_norm_values.data) < tol * tol) {
                converged = true;
                break;
            }
        }

        // exchange the ghost blocks of phi, needed to compute the gravitational acceleration
        node_gz_phi.evaluate();
        node_exch_gz_phi.evaluate();
        node_replace_gz_phi.evaluate();

        last_iteration_count = k;

        if (shamcomm::world_rank() == 0) {
            Tscal rel_res = std::sqrt(edges.old_values.data / edges.rhs_norm_values.data);
            if (converged) {
                logger::info_ln(
                    "Ramses::CG",
                    sham::format("converged after {} iterations, |r|/|b| = {}", k, rel_res));
            } else {
                logger::warn_ln(
                    "Ramses::CG",
                    sham::format(
                        "not converged after {} iterations, |r|/|b| = {} (tol = {})",
                        k,
                        rel_res,
                        tol));
            }
        }
    }

    template<class Tvec, class TgridVec>
    std::string NodeCGLoop<Tvec, TgridVec>::_impl_get_tex() const {
        std::string tex = R"tex(
            Jacobi preconditioned conjugate gradient solve of the Poisson equation
            \begin{equation}
            -\Delta x^3 \Delta {spans_phi} = -4 \pi G ({spans_rho} - {mean_rho}) \Delta x^3
            \end{equation}
        )tex";

        replace_edges_tex_symbols(tex);

        return tex;
    }

} // namespace shammodels::basegodunov::modules

template class shammodels::basegodunov::modules::NodeCGLoop<f64_3, i64_3>;
