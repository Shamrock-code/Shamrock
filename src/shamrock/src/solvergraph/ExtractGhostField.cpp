// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright (c) 2021-2025 Timothée David--Cléris <tim.shamrock@proton.me>
// SPDX-License-Identifier: CeCILL Free Software License Agreement v2.1
// Shamrock is licensed under the CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

/**
 * @file ExtractGhostField.cpp
 * @author Léodasce Sewanou (leodasce.sewanou@ens-lyon.fr)
 * @author Timothée David--Cléris (tim.shamrock@proton.me) --no git blame--
 * @brief
 *
 */

#include "shambase/stacktrace.hpp"
#include "shambackends/kernel_call.hpp"
#include "shamrock/patch/PatchDataField.hpp"
#include "shamrock/solvergraph/ExtractGhostField.hpp"
#include "shamsys/NodeInstance.hpp"

namespace shamrock::solvergraph {
    template<class T>
    void shamrock::solvergraph::ExtractGhostField<T>::_impl_evaluate_internal() {
        StackEntry stack_loc{};

        auto edges = get_edges();
        // logger::raw_ln("Extract:[p/cpy-p, cpy-ghost, idx] \t", &edges.original_fields, "-",
        // &edges.ghost_fields,"-", &edges.idx_in_ghots,"\n");

        auto &orig_data_fields = edges.original_fields;
        auto &idx_in_ghost     = edges.idx_in_ghots;
        auto &ghost_fields     = edges.ghost_fields;

        // node_cpy_field.evaluate();

        ghost_fields.patchdata_fields.reset(); // To make sure there is no residual of last timestep

        for (const auto &[key, sender_idx_in_ghost] : idx_in_ghost.buffers) {
            auto [sender, receiver] = key;
            // logger::raw_ln("[Extr-Ghs-Field]", sender, "--", receiver,
            // "--",(u32)sender_idx_in_ghost.get_size() ,"\n"); logger::raw_ln("\n\n");
            //  logger::raw_ln(sender, "--", receiver, "\n");

            auto field_name = orig_data_fields.get_field(sender).get_name();
            // auto field_name         =
            // shambase::get_check_ref(copy_field).get_field(sender).get_name(); auto nvar =
            // shambase::get_check_ref(copy_field).get_field(sender).get_nvar();
            auto nvar = orig_data_fields.get_field(sender).get_nvar();
            PatchDataField<T> gz_field(field_name, nvar);

            // shambase::get_check_ref(copy_field).get_field(sender).append_subset_to(
            //     sender_idx_in_ghost, (u32)sender_idx_in_ghost.get_size(), gz_field);

            // ghost_fields.patchdata_fields.add_obj(sender, receiver, std::move(gz_field));

            // sham::kernel_call(shamsys::instance::get_compute_scheduler_ptr()->get_queue(),
            // sham::MultiRef{sender_idx_in_ghost}, sham::MultiRef{},
            // (u32)sender_idx_in_ghost.get_size(), [](u32 idd, const u32 *__restrict acc_idxs){
            //     logger::raw_ln("-- idd = [", idd ,"]: ", acc_idxs[idd],"\n");
            // });
            // logger::raw_ln(" sizes ==", (u32)sender_idx_in_ghost.get_size(),"\n");

            orig_data_fields.get_field(sender).append_subset_to(
                sender_idx_in_ghost, (u32) sender_idx_in_ghost.get_size(), gz_field);

            ghost_fields.patchdata_fields.add_obj(sender, receiver, std::move(gz_field));
        }

        // u32 cnt_obj = 0, cnt_cells=0;
        // for(const auto &[key, sender_layer]: ghost_fields.patchdata_fields){
        //     auto [sender, receiver] = key;
        //     cnt_obj += 1;
        //     cnt_cells += sender_layer.get_obj_cnt();

        //     logger::raw_ln("[Extr-Ghs-Field]",sender, "--", receiver, "--",
        //     sender_layer.get_obj_cnt() ,"\n"); logger::raw_ln("[Extr-Ghs-Field]-","-obj= ",
        //     cnt_obj, "-- cells=",cnt_cells ,"\n");
        // }
    }

    template class ExtractGhostField<f64>;
    template class ExtractGhostField<f64_3>;

} // namespace shamrock::solvergraph
