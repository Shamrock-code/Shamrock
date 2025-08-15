// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright (c) 2021-2025 Timothée David--Cléris <tim.shamrock@proton.me>
// SPDX-License-Identifier: CeCILL Free Software License Agreement v2.1
// Shamrock is licensed under the CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

/**
 * @file NodePCGLoop.cpp
 * @author Léodasce Sewanou (leodasce.sewanou@ens-lyon.fr)
 * @author Timothée David--Cléris (tim.shamrock@proton.me)
 * @brief
 *
 */

#include "shambase/aliases_int.hpp"
#include "shambase/memory.hpp"
#include "shambackends/vec.hpp"
#include "shammodels/ramses/SolverConfig.hpp"
#include "shammodels/ramses/modules/NodePCGLoop.hpp"
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
    void NodePCGLoop<Tvec, TgridVec>::_impl_evaluate_internal() {
        StackEntry stack_loc{};
        auto edges = get_edges();
        edges.spans_block_cell_sizes.check_sizes(edges.sizes.indexes);
        edges.spans_phi.check_sizes(edges.sizes.indexes);
        edges.spans_rho.check_sizes(edges.sizes.indexes);
        edges.spans_phi_res.ensure_sizes(edges.sizes.indexes);
        edges.spans_phi_pres.ensure_sizes(edges.sizes.indexes);
        edges.spans_phi_p.ensure_sizes(edges.sizes.indexes);

        /* compute r0 =  4*\pi*G* \left( \rho - \bar{\rho} \right) - A \phi_{0}
         *          z0 =  M^{-1} r0
         *          p0 =  z0
         */
        node0.evaluate();

        /** compute compute Hadamard product r_0 X z_0 */
        node1.evaluate();

        /** get the dot product <r_0, z_0> and assign its value to  edges.old_values.value */
        node2.evaluate();

        u32 k = 0;
        logger::raw_ln(" k = ", k);
        logger::raw_ln(" RES = ", edges.old_values.value);

        /* Main loop */
        while ((k < Niter_max)) {
            // increment iteration
            k = k + 1;

            /* compute Ap_{k} */
            node3_1.evaluate();

            // /** compute Hadamard product p X Ap such that \left( p_{k} X Ap_{k} \right)_{i} =
            // left( p_{i} * (Ap)_{i} \right) */
            node3_2.evaluate();

            /** compute the A-norm of p_{k} , <p_{k}, Ap_{k}> and assign its value to
             * edges.e_norm.value */
            node4.evaluate();

            /** compute \alpha_{k} = \frac{ <r_{k},r_{k}> }{ <p_{k},Ap_{k}> }*/
            edges.alpha.value = edges.old_values.value / edges.e_norm.value;
            logger::raw_ln(" alpha = ", edges.alpha.value, "\n");

            /** compute new phi : \phi_{k+1} = \phi_{k} + \alpha_{k} p_{k}  */
            node5.evaluate();

            edges.alpha.value = -edges.alpha.value;

            /** compute new residual : r_{k+1} = r_{k} - \alpha_{k} (Ap_{k}) */
            node6.evaluate();

            /** compute new preconditioned residual z_{k+1}= M^{-1} r_{k+1} */
            node7.evaluate();

            /** compute Hadamard product r_{k+1} X z_{k+1} */
            node8.evaluate();

            /** compute the dot product <r_{k+1}, z_{k+1}> and assign its value to
             * edges.new_values.value*/
            node9.evaluate();

            // /** compute \beta_{k} = \frac{<r_{k+1},z_{k+1}>}{<r_{k},z_{k}>}*/
            edges.beta.value = edges.new_values.value / edges.old_values.value;
            logger::raw_ln(" beta = ", edges.beta.value, "\n");

            /** set <r_{k},r_{k}> = <r_{k+1},r_{k+1}>*/
            edges.old_values.value = edges.new_values.value;

            logger::raw_ln(" k = ", k);
            logger::raw_ln(" RES = ", edges.old_values.value);

            /** compute p_{k+1} = r_{k+1} + \beta_{k} p_{k} */
            node10.evaluate();

            if (sycl::sqrt(edges.old_values.value) < tol)
                break;
        }
    }

    template<class Tvec, class TgridVec>
    std::string NodePCGLoop<Tvec, TgridVec>::_impl_get_tex() {

        std::string tex = R"tex(
             PCG Main Loop
        )tex";

        return tex;
    }

} // namespace shammodels::basegodunov::modules

template class shammodels::basegodunov::modules::NodePCGLoop<f64_3, i64_3>;
