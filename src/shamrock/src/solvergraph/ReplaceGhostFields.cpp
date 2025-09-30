// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright (c) 2021-2025 Timothée David--Cléris <tim.shamrock@proton.me>
// SPDX-License-Identifier: CeCILL Free Software License Agreement v2.1
// Shamrock is licensed under the CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

/**
 * @file ReplaceGhostFields.cpp
 * @author Léodasce Sewanou (leodasce.sewanou@ens-lyon.fr)
 * @author Timothée David--Cléris (tim.shamrock@proton.me) --no git blame--
 * @brief
 *
 */

#include "shamrock/solvergraph/ReplaceGhostFields.hpp"
#include "shambackends/kernel_call.hpp"
#include "shambackends/kernel_call_distrib.hpp"
#include "shamcomm/logs.hpp"
#include "shamrock/patch/PatchDataField.hpp"
#include "shamsys/NodeInstance.hpp"

namespace shamrock::solvergraph {
    template<class T>
    void shamrock::solvergraph::ReplaceGhostFields<T>::_impl_evaluate_internal() {

        StackEntry stack_loc{};
        auto edges = get_edges();
        // logger::raw_ln("Rplce:[cpy-ghost, cpy-p] \t", &edges.ghost_fields, "-",
        // &edges.fields,"\n");

        auto &ghost_fields = edges.ghost_fields;
        auto &fields       = edges.fields;

        std::map<u32, u32> gz_map;
        ghost_fields.patchdata_fields.for_each([&](u32 s, u32 r, PatchDataField<T> &pdat_field) {
            gz_map[r] += pdat_field.get_obj_cnt();
        });

        shambase::DistributedData<u32> cell_counts = {};
        fields.get_refs().for_each([&](u32 id_patch, PatchDataField<T> &field) {
            cell_counts.add_obj(id_patch, u32(block_size * field.get_obj_cnt()));
        });

        /*********************************************************/
        // logger::raw_ln("\n\n ================= Before ghost replacement =============== \n\n");
        // logger::raw_ln("block size = ", block_size);

        // sham::distributed_data_kernel_call(
        //     shamsys::instance::get_compute_scheduler_ptr(),
        //     sham::DDMultiRef{fields.get_spans()},
        //     sham::DDMultiRef{},
        //     cell_counts,
        //     [=](u32 id, const T *__restrict x) {
        //         logger::raw_ln("id_b= [ ", id, " ] : ", x[id], "\n");
        //     });

        /********************************************************/

        /** Is there any other way to do this ? get_spans ?
        **  std::map<u32, PatchDataField<T>> recv_map;
        **  Add more methods to patchdata-field to do on place
        **
        */

        // std::map<u32, PatchDataField<T>> recv_map;
        // ghost_fields.patchdata_fields.for_each(
        //     [&](u32 s, u32 r, PatchDataField<T> &pdat_field) {
        //         recv_map.at(r).insert(pdat_field);
        //     });

        fields.get_refs().for_each([&](u32 id_patch, PatchDataField<T> &field) {
            PatchDataField<T> temp_pdat(field.get_name(), field.get_nvar());
            field.shrink(gz_map.at(id_patch));

            // logger::raw_ln("\n\n", field.get_obj_cnt(), "**aa*", gz_map.at(id_patch) ,"\n\n");
        });

        ghost_fields.patchdata_fields.for_each([&](u32 s, u32 r, PatchDataField<T> &pdat_field) {
            fields.get_field(r).insert(pdat_field);
        });

        // fields.get_refs().for_each([&](u32 id_patch, PatchDataField<T> &field) {
        //     PatchDataField<T> temp_pdat(field.get_name(), field.get_nvar());
        //     ghost_fields.patchdata_fields.for_each(
        //         [&](u32 s, u32 r, PatchDataField<T> &pdat_field) {
        //             if (r == id_patch) {
        //                 temp_pdat.insert(pdat_field);
        //             }
        //         });

        //     logger::raw_ln("\n\n", field.get_obj_cnt(), "**bb*", gz_map.at(id_patch) ,"\n\n");
        //     field.resize(field.get_obj_cnt() - gz_map.at(id_patch));
        //     field.insert(temp_pdat);
        // });

        // logger::raw_ln("\n\n ================= After ghost replacement =============== \n\n");

        // sham::distributed_data_kernel_call(
        // shamsys::instance::get_compute_scheduler_ptr(),
        // sham::DDMultiRef{fields.get_spans()},
        // sham::DDMultiRef{},
        // cell_counts,
        // [=](u32 id, const T *__restrict x) {
        //     logger::raw_ln("id_a = [ ", id, " ] : ", x[id], "\n");
        // });
    }

    template class shamrock::solvergraph::ReplaceGhostFields<f64>;
    template class shamrock::solvergraph::ReplaceGhostFields<f64_3>;

} // namespace shamrock::solvergraph
