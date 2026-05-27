// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright (c) 2021-2025 Timothée David--Cléris <tim.shamrock@proton.me>
// SPDX-License-Identifier: CeCILL Free Software License Agreement v2.1
// Shamrock is licensed under the CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

/**
 * @file BICGSTABLoop.cpp
 * @author Léodasce Sewanou (leodasce.sewanou@ens-lyon.fr)
 * @author Timothée David--Cléris (tim.shamrock@proton.me) --no git blame--
 * @brief
 *
 */

#include "shambase/aliases_int.hpp"
#include "shambase/memory.hpp"
#include "shambackends/vec.hpp"
#include "shammodels/ramses/SolverConfig.hpp"
#include "shammodels/ramses/modules/BICGSTABLoop.hpp"
#include "shammodels/ramses/solvegraph/OrientedAMRGraphEdge.hpp"
#include "shamrock/solvergraph/FieldRefs.hpp"
#include "shamrock/solvergraph/IFieldSpan.hpp"
#include "shamrock/solvergraph/INode.hpp"
#include "shamrock/solvergraph/Indexes.hpp"
#include "shamrock/solvergraph/ScalarEdge.hpp"
#include <shambackends/sycl.hpp>
#include <memory>
#include <utility>

namespace shammodels::basegodunov::modules {
    template<class Tvec, class TgridVec>
    void NodeBICGSTABLoop<Tvec, TgridVec>::_impl_evaluate_internal() {
        StackEntry stack_loc{};
        auto edges = get_edges();
        edges.spans_block_cell_sizes.check_sizes(edges.sizes.indexes);
        edges.spans_phi.check_sizes(edges.sizes.indexes);
        edges.spans_rho.check_sizes(edges.sizes.indexes);
        edges.spans_phi_res.ensure_sizes(edges.sizes.indexes);
        edges.spans_phi_res_bis.ensure_sizes(edges.sizes.indexes);
        edges.spans_phi_p.ensure_sizes(edges.sizes.indexes);
        edges.spans_phi_Ap.ensure_sizes(edges.sizes.indexes);
        edges.spans_phi_s.ensure_sizes(edges.sizes.indexes);
        edges.spans_phi_As.ensure_sizes(edges.sizes.indexes);
        edges.spans_phi_hadamard_prod.ensure_sizes(edges.sizes.indexes);

        /* compute r0 =  4*\pi*G* \left( \rho - \bar{\rho} \right) - A \phi_{0}
         *          r'0 = 0.5*r0
         *          p0 =  r0
         */
        node_init.evaluate();
        u32 k = 0;
        /* Main loop */
        while ((k < Niter_max)) {

            /** compute compute Hadamard product r_0 x r'_0 */
            node_had_prod_rj_rp0.evaluate();

            /** get the dot product <r_0, r'_0> and assign its value to  edges.old_values.value */
            node_ddot_rj_rp0.evaluate();

            // if (shamcomm::world_rank() == 0) {
            //     logger::raw_ln("k= \t ", k, " \t res = \t", edges.old_values.value);
            // }

            // increment iteration
            k = k + 1;

            //--------------------------------
            /* comm of p vector*/
            //----------------------------

            if (true) {
                // //exchange p vector
                node_gz_p.evaluate();
                node_exch_gz_p.evaluate();
                node_replace_gz_p.evaluate();
            }

            /** compute Ap_{k} */
            node_Apj.evaluate();

            /** compute Hadamard product Ap_{k} x r'_{0}*/
            node_had_Apj_rp0.evaluate();

            /** compute <Ap_{k}, r'_{0}> and assign its value to edges.e_norm.value*/
            node_ddot_Apj_rp0.evaluate();

            /** compute \alpha_{k} = \frac{ <r_{k},r'_{0}> }{ <r'_{0},Ap_{k}> }*/
            edges.alpha.value = edges.old_values.value / edges.e_norm.value;

            // if (shamcomm::world_rank() == 0) {
            //     logger::raw_ln("edges.alpha.value = \t ", k, " \t ", edges.alpha.value);
            // }

            auto alp_saved = edges.alpha.value;

            /** compute s_{k} = r_{k} - alpha_{k}Ap_{k} */
            edges.e_norm.value = 0;
            edges.beta.value   = 1;
            edges.alpha.value  = -alp_saved;
            node_sj_vec.evaluate();

            //     /** compute <s_{k}, s_{k}> and set its value to edges.e_norm.value*/
            node_ddot_sj_sj.evaluate();

            // if (shamcomm::world_rank() == 0) {
            //     logger::raw_ln("<s_k,s_k> \t ", edges.e_norm.value, "\n");
            // }

            /** perform cvg test*/
            if (edges.e_norm.value < tol_cvg * tol_cvg) {
                edges.alpha.value = alp_saved;
                node_new_phi_happy_break.evaluate();
                if (shamcomm::world_rank() == 0) {
                    logger::raw_ln("Converge on s-residual \n");
                }

                break;
            }

            //--------------------------------
            /* comm of s vector*/
            //----------------------------
            if (true) {
                // //exchange s vector
                node_gz_s.evaluate();
                node_exch_gz_s.evaluate();
                node_replace_gz_s.evaluate();
            }

            /** compute  As_{k}*/
            node_Asj.evaluate();
            /** compute As_{k} x s_{k}*/
            node_had_Asj_sj.evaluate();
            /** compute <As_{k},s_{k}> and set its value to edges.e_norm.value*/
            node_ddot_Asj_sj.evaluate();

            // if (shamcomm::world_rank() == 0) {
            //     logger::raw_ln("<As_k,s_k> \t ", edges.e_norm.value, "\n");
            // }

            /** compute <As_{k},As_{k} and set its value to edges.new_values.value*/
            node_ddot_Asj_Asj.evaluate();

            // if (shamcomm::world_rank() == 0) {
            //     logger::raw_ln("<As_k,As_k> \t ", edges.new_values.value, "\n");
            // }

            /** compute w_{k} and set its value to edges.w_stab.value*/
            edges.w_stab.value = (edges.e_norm.value / edges.new_values.value);

            auto w_saved = edges.w_stab.value;

            /** compute new-phi*/
            node_new_phi.evaluate();

            /** compute new-residual*/
            edges.w_stab.value     = -w_saved;
            edges.e_norm.value     = 0;
            edges.new_values.value = 1;
            node_new_res.evaluate();

            /** compute <r_{k+1}, r_{k+1}> and assign its value to edges.e_norm.value*/
            node_ddot_rj_rj.evaluate();
            if (shamcomm::world_rank() == 0) {
                logger::raw_ln("k= \t ", k, "<r_k+,r_k+> \t ", edges.e_norm.value, "\n");
            }

            /** perform cvg test*/
            if (edges.e_norm.value < tol_cvg * tol_cvg) {
                if (shamcomm::world_rank() == 0) {
                    logger::raw_ln("Converge on residual \n");
                }
                break;
            }

            /** compute r_{k+1} x r'{0} */
            node_had_rjnew_rp0.evaluate();
            /**compute <r_{k+1}, r'{0}> and assign its value to edges.new_values.value */
            node_ddot_rjnew_rp0.evaluate();
            /**compute beta_{k} = (alpha_{k} / w_{k}) x (<r_{k+1}, r'{0}> / <r_{k}, r'{0}>) */
            edges.beta.value
                = (alp_saved / w_saved) * (edges.new_values.value / edges.old_values.value);

            /** compute p_{k+1} = r_{k+1} + \beta_{k}(p_{k} - w_{k} Ap_{k}) */
            edges.e_norm.value = 1;
            edges.alpha.value  = (-w_saved) * (edges.beta.value);
            node_new_p_vec.evaluate();

            /** perform lucky breakdown test for restarting */
            if ((edges.new_values.value * edges.new_values.value) < (tol_happy_bk * tol_happy_bk)) {
                /** set r'_{0} = r_{k+1} and p_{k+1} = r_{k+1}*/
                if (shamcomm::world_rank() == 0) {
                    logger::raw_ln("Lucky Break \t =======> \t restarting \n");
                }

                node_overwrite_rp0.evaluate();
                node_overwrite_p.evaluate();
            }
        }
    }

    template<class Tvec, class TgridVec>
    std::string NodeBICGSTABLoop<Tvec, TgridVec>::_impl_get_tex() const {

        std::string tex = R"tex(
             BICGSTAB Main Loop
        )tex";

        return tex;
    }

} // namespace shammodels::basegodunov::modules

template class shammodels::basegodunov::modules::NodeBICGSTABLoop<f64_3, i64_3>;
