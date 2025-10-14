// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright (c) 2021-2025 Timothée David--Cléris <tim.shamrock@proton.me>
// SPDX-License-Identifier: CeCILL Free Software License Agreement v2.1
// Shamrock is licensed under the CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

/**
 * @file NodeCGLoop.cpp
 * @author Léodasce Sewanou (leodasce.sewanou@ens-lyon.fr)
 * @author Timothée David--Cléris (tim.shamrock@proton.me) --no git blame--
 * @brief
 *
 */

#include "shambase/aliases_int.hpp"
#include "shambase/memory.hpp"
#include "shambackends/vec.hpp"
#include "shamcomm/logs.hpp"
#include "shamcomm/worldInfo.hpp"
#include "shammodels/ramses/SolverConfig.hpp"
#include "shammodels/ramses/modules/NodeCGLoop.hpp"
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
    void NodeCGLoop<Tvec, TgridVec>::_impl_evaluate_internal() {
        StackEntry stack_loc{};
        auto edges = get_edges();
        edges.spans_block_cell_sizes.check_sizes(edges.sizes.indexes);
        edges.spans_phi.check_sizes(edges.sizes.indexes);
        edges.spans_rho.check_sizes(edges.sizes.indexes);
        edges.spans_phi_res.ensure_sizes(edges.sizes.indexes);
        edges.spans_phi_p.ensure_sizes(edges.sizes.indexes);

        /* compute r0 = p0 = 4*\pi*G* \left( \rho - \bar{\rho} \right) - A \phi_{0}*/
        node0.evaluate();
        if (true) {
            for (auto id = 0; id < 1; id++) {
                auto &buf = edges.spans_phi_res.get_buf(id);
                auto vec  = buf.copy_to_stdvec();
                logger::raw_ln(id, "buf res 0 =", "--", buf.get_size());
                for (int i = 0; i < buf.get_size(); i++) {
                    logger::raw_ln(i, vec[i]);
                }
            }
        }

        /* compute <r0,r0> and assign its value to  edges.old_values.value */
        node_copy_phi.evaluate();
        edges.spans_phi_cpy.ensure_sizes(edges.sizes_no_gz.indexes);
        if (false) {
            for (auto id = 0; id < 1; id++) {
                auto &buf = edges.spans_phi_cpy.get_buf(id);
                auto vec  = buf.copy_to_stdvec();
                logger::raw_ln(id, "buf cpy_phi bf=", "--", buf.get_size());
                // for (int i = 0; i < buf.get_size(); i++) {
                //     logger::raw_ln(i, vec[i]);
                // }
            }
        }

        node1.evaluate();

        u32 k = 0;
        logger::raw_ln(" k = ", k);
        logger::raw_ln(" RES = ", edges.old_values.value);

        /*** Main loop */
        while ((k < Niter_max)) {
            // increment iteration
            k = k + 1;

            logger::raw_ln(" ================== k = ", k, "=======================\n");
            /* compute Ap_{k} */

            // if (true) {
            //     // //exchange p vector
            //     node_gz_p.evaluate();
            //     node_exch_gz_p.evaluate();
            //     node_replace_gz_p.evaluate();
            // }

            node2.evaluate();

            // if (true) {
            //     // //exchange Ap vector

            //     node_Ap_gz.evaluate();
            //     node_Ap_exch_gz.evaluate();
            //     node_Ap_replace_gz.evaluate();
            // }

            /** compute Hadamard product p X Ap such that \left( p_{k} X Ap_{k} \right)_{i} =
            left(
             * p_{i} * (Ap)_{i} \right) */
            node3.evaluate();
            auto gg = 0.;

            if (true) {
                for (auto id = 0; id < 1; id++) {
                    auto &buf = edges.spans_phi_res.get_buf(id);
                    auto vec  = buf.copy_to_stdvec();

                    auto &buf_p = edges.spans_phi_p.get_buf(id);
                    auto vec_p  = buf_p.copy_to_stdvec();

                    auto &buf_Ap = edges.spans_phi_Ap.get_buf(id);
                    auto vec_Ap  = buf_Ap.copy_to_stdvec();

                    logger::raw_ln(id, "buf res 0 =", "--", buf.get_size());
                    logger::raw_ln(
                        " no-gz  : ", edges.sizes_no_gz.indexes.get(id) * block_size, "\n");
                    for (int i = 0; i < buf.get_size(); i++) {
                        if (i < edges.sizes_no_gz.indexes.get(id) * block_size) {
                            gg += vec_p[i] * vec_Ap[i];
                            logger::raw_ln(i, vec[i], vec_p[i], vec_Ap[i], vec_p[i] * vec_Ap[i]);
                        }
                    }
                    logger::raw_ln("global e-nrm : ", gg, "\n");
                }
            }

            /** compute the A-norm of p_{k} , <p_{k}, Ap_{k}> and assign its value to
             * edges.e_norm.value */
            edges.spans_phi_hadamard_prod_cpy.ensure_sizes(edges.sizes.indexes);
            node_copy_had_prod.evaluate();
            edges.spans_phi_hadamard_prod_cpy.ensure_sizes(edges.sizes_no_gz.indexes);

            node4.evaluate();
            logger::raw_ln(" e-norm = ", edges.e_norm.value);

            /** compute \alpha_{k} = \frac{ <r_{k},r_{k}> }{ <p_{k},Ap_{k}> }*/
            edges.alpha.value = edges.old_values.value / edges.e_norm.value;
            logger::raw_ln(" alpha bf = ", edges.alpha.value, "\n");

            //     /** compute new phi : \phi_{k+1} = \phi_{k} + \alpha_{k} p_{k}  */
            node5.evaluate();

            // edges.alpha.value = -edges.alpha.value;
            logger::raw_ln(" alpha af = ", edges.alpha.value, "\n");
            /** compute new residual : r_{k+1} = r_{k} - \alpha_{k} (Ap_{k}) */

            node6.evaluate();

            edges.spans_phi_cpy.ensure_sizes(edges.sizes.indexes);
            node_copy_phi.evaluate();
            edges.spans_phi_cpy.ensure_sizes(edges.sizes_no_gz.indexes);
            /** compute <r_{k+1},r_{k+1}> and assign its value to edges.new_values.value */

            node7.evaluate();

            logger::raw_ln(" new-norm = ", edges.new_values.value);
            /** compute \beta_{k} = \frac{<r_{k+1},r_{k+1}>}{<r_{k},r_{k}>}*/
            edges.beta.value = edges.new_values.value / edges.old_values.value;
            logger::raw_ln(" beta = ", edges.beta.value, "\n");
            /** set <r_{k},r_{k}> = <r_{k+1},r_{k+1}>*/
            edges.old_values.value = edges.new_values.value;
            logger::raw_ln(" new = ", edges.old_values.value, "\n");
            logger::raw_ln(" RES = ", edges.old_values.value);
            /** compute p_{k+1} = r_{k+1} + \beta_{k} p_{k} */

            node8.evaluate();

            if (false) {
                // exchange p vector
                node_gz_p.evaluate();
                node_exch_gz_p.evaluate();
                node_replace_gz_p.evaluate();
            }

            if (sycl::sqrt(edges.old_values.value) < tol)
                break;
        }

        // while ((k < Niter_max)) {
        //     // increment iteration
        //     k = k + 1;
        //         edges.spans_phi_p.ensure_sizes(edges.sizes_no_gz.indexes);

        //         if (true) {
        //             // exchange p vector
        //             node_gz.evaluate();
        //             node_exch_gz.evaluate();
        //             node_replace_gz.evaluate();
        //         }

        //         // if (true) {
        //         //     for (auto id = 0; id < 1; id++) {
        //         //         auto &buf = edges.spans_phi_p.get_buf(id);
        //         //         auto vec  = buf.copy_to_stdvec();
        //         //         logger::raw_ln(id, "buf p =", "--", buf.get_size());
        //         //         // for (int i = 0; i < buf.get_size(); i++) {
        //         //         //     logger::raw_ln(i, vec[i]);
        //         //         // }
        //         //     }
        //         // }

        // node2.evaluate();

        //         edges.spans_phi_p.ensure_sizes(edges.sizes_no_gz.indexes);
        //         edges.spans_phi_Ap.ensure_sizes(edges.sizes_no_gz.indexes);

        //         if (false) {
        //         for (auto id = 0; id < 1; id++) {
        //             auto &buf = edges.spans_phi_p.get_buf(id);
        //             auto vec  = buf.copy_to_stdvec();
        //             logger::raw_ln(id, "buf p =", "--", buf.get_size());
        //             for (int i = 0; i < buf.get_size(); i++) {
        //                 logger::raw_ln(i, vec[i]);
        //             }
        //         }
        //     }

        // if (true) {
        //     for (auto id = 0; id < 1; id++) {
        //         auto &buf = edges.spans_phi_Ap.get_buf(id);
        //         auto vec  = buf.copy_to_stdvec();
        //         logger::raw_ln(id, "buf Ap =", "--", buf.get_size());
        //         for (int i = 0; i < buf.get_size(); i++) {
        //             logger::raw_ln(i, vec[i]);
        //         }
        //     }
        // }

        // node3.evaluate();

        // if (true) {
        //     for (auto id = 0; id < 1; id++) {
        //         auto &buf = edges.spans_phi_hadamard_prod.get_buf(id);
        //         auto vec  = buf.copy_to_stdvec();
        //         logger::raw_ln(id, "buf truc had =", "--", buf.get_size());
        //         for (int i = 0; i < buf.get_size(); i++) {
        //             logger::raw_ln(i, vec[i]);
        //         }
        //     }
        // }

        //         node4.evaluate();

        //         logger::raw_ln(" e-norm  = ", edges.e_norm.value, "\n");
        //         edges.alpha.value = edges.old_values.value / edges.e_norm.value;
        //         logger::raw_ln(" alpha  = ", edges.alpha.value, "\n");

        // //         edges.spans_phi_cpy.ensure_sizes(edges.sizes_no_gz.indexes);
        // //           if (false) {
        // //             for (auto id = 0; id < 1; id++) {
        // //                 auto &buf = edges.spans_phi_cpy.get_buf(id);
        // //                 auto vec  = buf.copy_to_stdvec();
        // //                 logger::raw_ln(id, "buf phi =", "--", buf.get_size());
        // //                 for (int i = 0; i < buf.get_size(); i++) {
        // //                     logger::raw_ln(i, vec[i]);
        // //                 }
        // //                 }
        // //           }

        // //         if (false) {
        // //         for (auto id = 0; id < 1; id++) {
        // //             auto &buf_ap = edges.spans_phi_Ap.get_buf(id);
        // //             auto &buf_p = edges.spans_phi_p.get_buf(id);
        // //             auto &buf_res = edges.spans_phi_res.get_buf(id);
        // //             auto &buf_phi = edges.spans_phi_cpy.get_buf(id);
        // //             auto &buf_had = edges.spans_phi_hadamard_prod.get_buf(id);
        // //             // auto vec_ap  = buf_ap.copy_to_stdvec();
        // //             // auto vec_p  = buf_p.copy_to_stdvec();
        // //             // auto vec_res  = buf_res.copy_to_stdvec();
        // //             // auto vec_  = buf_phi.copy_to_stdvec();
        // //             logger::raw_ln(id, "buf truc had =", "--", buf_had.get_size());
        // //             logger::raw_ln(id, "buf truc ap =", "--", buf_ap.get_size());
        // //             logger::raw_ln(id, "buf truc p =", "--", buf_p.get_size());
        // //             logger::raw_ln(id, "buf truc res =", "--", buf_res.get_size());
        // //             logger::raw_ln(id, "buf truc af =", "--", buf_phi.get_size());

        // //         }
        // //     }

        //         node5.evaluate();

        //         edges.alpha.value = -edges.alpha.value;
        //         logger::raw_ln(" alpha af = ", edges.alpha.value, "\n");

        // //   if (false) {
        // //         for (auto id = 0; id < 1; id++) {
        // //             auto &buf_ap = edges.spans_phi_Ap.get_buf(id);
        // //             auto &buf_p = edges.spans_phi_p.get_buf(id);
        // //             auto &buf_res = edges.spans_phi_res.get_buf(id);
        // //             auto &buf_phi = edges.spans_phi_cpy.get_buf(id);
        // //             auto &buf_had = edges.spans_phi_hadamard_prod.get_buf(id);
        // //             // auto vec_ap  = buf_ap.copy_to_stdvec();
        // //             // auto vec_p  = buf_p.copy_to_stdvec();
        // //             // auto vec_res  = buf_res.copy_to_stdvec();
        // //             // auto vec_  = buf_phi.copy_to_stdvec();
        // //             logger::raw_ln(id, "buf truc had =", "--", buf_had.get_size());
        // //             logger::raw_ln(id, "buf truc ap =", "--", buf_ap.get_size());
        // //             logger::raw_ln(id, "buf truc p =", "--", buf_p.get_size());
        // //             logger::raw_ln(id, "buf truc res =", "--", buf_res.get_size());
        // //             logger::raw_ln(id, "buf truc phi =", "--", buf_phi.get_size());

        // //         }
        // //     }

        // //         if (false) {
        // //         for (auto id = 0; id < 1; id++) {
        // //             auto &buf = edges.spans_phi_res.get_buf(id);
        // //             auto vec  = buf.copy_to_stdvec();
        // //             logger::raw_ln(id, "bphi-res =", "--", buf.get_size());
        // //             for (int i = 0; i < buf.get_size(); i++) {
        // //                 logger::raw_ln(i, vec[i]);
        // //             }
        // //         }
        // //     }

        // // if (false) {
        // //     for (auto id = 0; id < 1; id++) {
        // //         auto &buf = edges.spans_phi_res.get_buf(id);
        // //         auto &buf_ap = edges.spans_phi_Ap.get_buf(id);
        // //         auto vec  = buf.copy_to_stdvec();
        // //         auto vec_ap  = buf_ap.copy_to_stdvec();
        // //         logger::raw_ln(id, "bphi-res =", "--", buf.get_size());
        // //         for (int i = 0; i < buf.get_size(); i++) {
        // //             logger::raw_ln(i, vec[i] , edges.alpha.value*vec_ap[i]);
        // //         }
        // //     }
        // // }
        // // edges.spans_phi_res.ensure_sizes(edges.sizes_no_gz.indexes);

        //         node6.evaluate();

        // // //     if (true) {
        // // //     for (auto id = 0; id < 1; id++) {
        // // //         auto &buf = edges.spans_phi_res.get_buf(id);
        // // //         auto &buf_ap = edges.spans_phi_Ap.get_buf(id);
        // // //         auto vec  = buf.copy_to_stdvec();
        // // //         auto vec_ap  = buf_ap.copy_to_stdvec();
        // // //         logger::raw_ln(id, "bphi-res =", "--", buf.get_size());
        // // //         for (int i = 0; i < buf.get_size(); i++) {
        // // //             logger::raw_ln(i, vec[i] , vec_ap[i]);
        // // //         }
        // // //     }
        // // // }

        //         node7.evaluate();
        //         logger::raw_ln(" new-val = ", edges.new_values.value, "\n");

        // //         edges.beta.value = edges.new_values.value / edges.old_values.value;
        // //         logger::raw_ln(" beta = ", edges.beta.value, "\n");

        // //         edges.old_values.value = edges.new_values.value;

        // //         logger::raw_ln(" new = ", edges.old_values.value, "\n");

        // //         logger::raw_ln(" RES = ", edges.old_values.value);

        // //         node8.evaluate();

        // }
    }

    template<class Tvec, class TgridVec>
    std::string NodeCGLoop<Tvec, TgridVec>::_impl_get_tex() {

        std::string tex = R"tex(
             CG Main Loop
        )tex";

        return tex;
    }

} // namespace shammodels::basegodunov::modules

template class shammodels::basegodunov::modules::NodeCGLoop<f64_3, i64_3>;
