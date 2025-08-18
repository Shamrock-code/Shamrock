// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright (c) 2021-2025 Timothée David--Cléris <tim.shamrock@proton.me>
// SPDX-License-Identifier: CeCILL Free Software License Agreement v2.1
// Shamrock is licensed under the CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

/**
 * @file NodeBICGSTABLoop.cpp
 * @author Léodasce Sewanou (leodasce.sewanou@ens-lyon.fr)
 * @author Timothée David--Cléris (tim.shamrock@proton.me) --no git blame--
 * @brief
 *
 */

#include "shambase/aliases_int.hpp"
#include "shambase/memory.hpp"
#include "shambackends/vec.hpp"
#include "shammodels/ramses/SolverConfig.hpp"
#include "shammodels/ramses/modules/NodeBICGSTABLoop.hpp"
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
         *          r'0 = -r0
         *          p0 =  r0
         */
        node0_1.evaluate();

        /** compute compute Hadamard product r_0 x r'_0 */
        node0_2.evaluate();

        /** get the dot product <r_0, r'_0> and assign its value to  edges.old_values.value */
        node0_3.evaluate();

        u32 k = 0;
        logger::raw_ln(" k = ", k);
        logger::raw_ln(" RES = ", edges.old_values.value);

        /* Main loop */
        while ((k < Niter_max)) {
            // increment iteration
            k = k + 1;

            /** compute Ap_{k} */
            node1_1.evaluate();

            /** compute Hadamard product Ap_{k} x r'_{0}*/
            node1_2.evaluate();

            /** compute <Ap_{k}, r'_{0}> and assign its value to edges.e_norm.value*/
            node1_3.evaluate();

            /** compute \alpha_{k} = \frac{ <r_{k},r'_{0}> }{ <r'_{0},Ap_{k}> }*/
            edges.alpha.value = edges.old_values.value / edges.e_norm.value;
            logger::raw_ln(" alpha = ", edges.alpha.value, "\n");

            Tscal alp_saved = edges.alpha.value;

            /** compute s_{k} = r_{k} - alpha_{k}Ap_{k} */
            edges.e_norm.value = 0;
            edges.beta.value   = 1;
            edges.alpha.value  = -alp_saved;
            node2_1.evaluate();

            /** compute <s_{k}, s_{k}> and set its value to edges.e_norm.value*/
            node2_2.evaluate();
            /** perform cvg test*/
            if (edges.e_norm.value < tol_cvg) {
                /** compute new phi and break*/
                edges.alpha.value = alp_saved;
                node2_3.evaluate();
                break;
            }

            /** compute  As_{k}*/
            node3_1.evaluate();
            /** compoute As_{k} x s_{k}*/
            node3_2.evaluate();
            /** compute <As_{k},s_{k}> and set its value to edges.e_norm.value*/
            node3_3.evaluate();
            /** compute <As_{k},As_{k} and set its value to edges.new_values.value*/
            node3_4.evaluate();
            /** compute w_{k} and set its value to edges.w_stab.value*/
            edges.w_stab.value = (edges.e_norm.value / edges.new_values.value);

            Tscal w_saved = edges.w_stab.value;

            /** compute new-phi*/
            node4.evaluate();

            /** compute new-residual*/
            edges.w_stab.value     = -w_saved;
            edges.e_norm.value     = 0;
            edges.new_values.value = 1;
            node5_1.evaluate();

            /** compute <r_{k+1}, r_{k+1}> and assign its value to edges.e_norm.value*/
            node5_2.evaluate();
            logger::raw_ln(" k = ", k);
            logger::raw_ln(" RES = ", edges.e_norm.value);
            /** perform cvg test*/
            if (edges.e_norm.value < tol_cvg) {
                break;
            }

            /** compute r_{k+1} x r'{0} */
            node6_1.evaluate();
            /**compute <r_{k+1}, r'{0}> and assign its value to edges.new_values.value */
            node6_2.evaluate();
            /**compute beta_{k} = (alpha_{k} / w_{k}) x (<r_{k+1}, r'{0}> / <r_{k}, r'{0}>) */
            edges.beta.value
                = (alp_saved / w_saved) * (edges.new_values.value / edges.old_values.value);

            /** compute p_{k+1} = r_{k+1} + \beta_{k}(p_{k} - w_{k} Ap_{k}) */
            edges.e_norm.value = 1;
            edges.alpha.value  = (-w_saved) * (edges.beta.value);
            node7.evaluate();

            /** set old_value to new_value */
            edges.old_values.value = edges.new_values.value;
            // logger::raw_ln(" k = ", k);
            // logger::raw_ln(" RES = ", edges.old_values.value);

            /** perform lucky breakdown test for restarting */
            if ((edges.new_values.value * edges.new_values.value) < (tol_happy_bk * tol_happy_bk)) {
                /** set r'_{0} = r_{k+1} and p_{k+1} = r_{k+1}*/
                node8_1.evaluate();
                node8_2.evaluate();
            }
        }
    }

    template<class Tvec, class TgridVec>
    std::string NodeBICGSTABLoop<Tvec, TgridVec>::_impl_get_tex() {

        std::string tex = R"tex(
             BICGSTAB Main Loop
        )tex";

        return tex;
    }

} // namespace shammodels::basegodunov::modules

template class shammodels::basegodunov::modules::NodeBICGSTABLoop<f64_3, i64_3>;
