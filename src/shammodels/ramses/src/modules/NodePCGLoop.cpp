// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright (c) 2021-2026 Timothée David--Cléris <tim.shamrock@proton.me>
// SPDX-License-Identifier: CeCILL Free Software License Agreement v2.1
// Shamrock is licensed under the CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

/**
 * @file NodePCGLoop.cpp
 * @author Léodasce Sewanou (leodasce.sewanou@ens-lyon.fr)
 * @author Timothée David--Cléris (tim.shamrock@proton.me) --no git blame--
 * @brief
 *
 */

#include "shambase/aliases_int.hpp"
#include "shamalgs/collective/reduction.hpp"
#include "shamcomm/logs.hpp"
#include "shamcomm/worldInfo.hpp"
#include "shammodels/ramses/modules/NodeCGLoop.hpp"
#include "shamrock/solvergraph/Indexes.hpp"
#include "shamrock/solvergraph/ScalarEdge.hpp"
#include <shambackends/sycl.hpp>

namespace shammodels::basegodunov::modules {
    template<class Tvec, class TgridVec>
    void NodeCGLoop<Tvec, TgridVec>::_impl_evaluate_internal() {
        StackEntry stack_loc{};
        auto edges = get_edges();

        auto r_0 = 1.;

        {
            edges.spans_block_cell_sizes.check_sizes(edges.sizes.indexes);
            edges.spans_phi.check_sizes(edges.sizes.indexes);
            edges.spans_rho.check_sizes(edges.sizes.indexes);
            edges.spans_phi_res.ensure_sizes(edges.sizes.indexes);
            edges.spans_phi_p.ensure_sizes(edges.sizes.indexes);
            edges.spans_phi_rhs.ensure_sizes(edges.sizes.indexes);
            edges.spans_phi_Ap.ensure_sizes(edges.sizes.indexes);
            edges.spans_phi_hadamard_prod.ensure_sizes(edges.sizes.indexes);
            edges.spans_rz_hadamard_prod.ensure_sizes(edges.sizes.indexes);
            edges.spans_phi_z.ensure_sizes(edges.sizes.indexes);

            /* compute r0 = p0 = 4*\pi*G* \left( \rho - \bar{\rho} \right) - A \phi_{0}*/
            cg_init_node.evaluate();

            /* compute <r0,r0> and assign its value to  edges.old_values.value */
            res_ddot_node.evaluate();

            /* compute norm of rhs <b_rhs,b_rsh>*/
            rhs_node.evaluate();

            /* compute hadamard r*z*/
            rz_hadamard_prod_node.evaluate();

            /* compute old <r,z>*/
            rz_reduction_node.evaluate();

            u32 k = 0;
            if (shamcomm::world_rank() == 0) {
                logger::raw_ln(" k = ", k);
                logger::raw_ln("RES (L2-squared) = ", edges.old_values.value);
                r_0 = edges.old_values.value;
                logger::raw_ln("rhs value = \t", edges.rhs_norm_values.value, "\n\n");
            }

            /*** Main loop */
            // auto dev_sched = shamsys::instance::get_compute_scheduler_ptr();

            while ((k < Niter_max)) {
                // increment iteration
                k = k + 1;

                /** Prevent NaN to propagate. This can happen for the second cell in the ghost-zone.
                 */
                if (true) {
                    // //exchange p vector
                    node_gz_p.evaluate();
                    node_exch_gz_p.evaluate();
                    node_replace_gz_p.evaluate();
                }

                /* compute Ap_{k} */
                spmv_node.evaluate();

                /** compute Hadamard product p X Ap such that \left( p_{k} X Ap_{k} \right)_{i} =
                left(
                 * p_{i} * (Ap)_{i} \right) */
                hadamard_prod_node.evaluate();

                /** compute the A-norm of p_{k} , <p_{k}, Ap_{k}> and assign its value to
                 * edges.e_norm.value */
                a_norm_node.evaluate();

                /** compute \alpha_{k} = \frac{ <r_{k},r_{k}> }{ <p_{k},Ap_{k}> }*/
                //---------------
                // edges.alpha.value = edges.old_values.value / edges.e_norm.value;
                //------------
                edges.alpha.value = edges.rz_old_values.value / edges.e_norm.value;

                /** compute new phi : \phi_{k+1} = \phi_{k} + \alpha_{k} p_{k}  */
                new_potential_node.evaluate();

                /** compute new residual : r_{k+1} = r_{k} - \alpha_{k} (Ap_{k}) */
                edges.alpha.value = -edges.alpha.value;
                new_residual_node.evaluate();

                /** compute preconditioned residual: z_{k+1} = M^{-1}*r_{k+1} */
                res_precond_node.evaluate();

                /** compute <r_{k+1},r_{k+1}> and assign its value to edges.new_values.value */
                res_ddot_new_node.evaluate();

                /** r_{k+1}*z_{k+1} **/
                rz_hadamard_prod_node.evaluate();

                /** compute new <r,z> : <r_{k+1}, z_{k+1}>*/
                rz_new_reduction_node.evaluate();

                /** compute \beta_{k} = \frac{<r_{k+1},r_{k+1}>}{<r_{k},r_{k}>}*/
                //-------------------
                // edges.beta.value = edges.new_values.value / edges.old_values.value;
                //------------------
                edges.beta.value = edges.rz_new_values.value / edges.rz_old_values.value;

                /** set <r_{k},r_{k}> = <r_{k+1},r_{k+1}>*/
                edges.old_values.value = edges.new_values.value;

                /** set <r_{k}, z_{k}> = <r_{k+1}, z_{k+1}>*/
                edges.rz_old_values = edges.rz_new_values;
                if (shamcomm::world_rank() == 0) {
                    logger::raw_ln("New-RES (L2-squared)  = \t ", (edges.old_values.value), "\n\n");
                }

                /** compute p_{k+1} = r_{k+1} + \beta_{k} p_{k} */
                new_p_node.evaluate();

                if ((edges.old_values.value / edges.rhs_norm_values.value) < tol * tol) {
                    if (shamcomm::world_rank() == 0) {
                        logger::raw_ln("The solution converged after ", k, "iterations");
                    }

                    break;
                }
            }
        }
    }

} // namespace shammodels::basegodunov::modules

template class shammodels::basegodunov::modules::NodeCGLoop<f64_3, i64_3>;
