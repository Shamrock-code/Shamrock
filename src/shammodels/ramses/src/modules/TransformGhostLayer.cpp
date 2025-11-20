// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright (c) 2021-2025 Timothée David--Cléris <tim.shamrock@proton.me>
// SPDX-License-Identifier: CeCILL Free Software License Agreement v2.1
// Shamrock is licensed under the CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

/**
 * @file TransformGhostLayer.cpp
 * @author Timothée David--Cléris (tim.shamrock@proton.me)
 * @brief
 */

#include "shambase/assert.hpp"
#include "shambase/exception.hpp"
#include "shambackends/DeviceBuffer.hpp"
#include "shambackends/DeviceScheduler.hpp"
#include "shambackends/kernel_call.hpp"
#include "shammath/AABB.hpp"
#include "shammath/paving_function.hpp"
#include "shammodels/common/amr/AMRBlock.hpp"
#include "shammodels/ramses/modules/FindGhostLayerCandidates.hpp"
#include "shammodels/ramses/modules/TransformGhostLayer.hpp"
#include "shamrock/patch/PatchDataField.hpp"
#include "shamrock/patch/PatchDataLayer.hpp"
#include "shamsys/NodeInstance.hpp"
#include <stdexcept>
#include <vector>

template<class Tvec, class TgridVec>
void shammodels::basegodunov::modules::TransformGhostLayer<Tvec, TgridVec>::
    _impl_evaluate_internal() {
    auto edges = get_edges();

    // inputs
    auto &sim_box                 = edges.sim_box.value;
    auto &ghost_layers_candidates = edges.ghost_layers_candidates;

    // outputs
    auto &ghost_layer = edges.ghost_layer;

    auto dev_sched       = shamsys::instance::get_compute_scheduler_ptr();
    sham::DeviceQueue &q = shambase::get_check_ref(dev_sched).get_queue();

    auto paving_function = get_paving(mode, sim_box);

    // get the block min and max field
    u32 iblock_min = ghost_layer_layout->get_field_idx<TgridVec>("cell_min");
    u32 iblock_max = ghost_layer_layout->get_field_idx<TgridVec>("cell_max");

    // extract the ghost layers
    auto ghost_layer_it      = ghost_layer.patchdatas.begin();
    auto ghost_layer_info_it = ghost_layers_candidates.values.begin();

    if (ghost_layer.patchdatas.get_element_count()
        != ghost_layers_candidates.values.get_element_count()) {
        shambase::throw_with_loc<std::runtime_error>(shambase::format(
            "ghost_layer.patchdatas.get_element_count() != "
            "ghost_layers_candidates.values.get_element_count()\n "
            "ghost_layer.patchdatas.get_element_count(): {}\n"
            "ghost_layers_candidates.values.get_element_count(): {}",
            ghost_layer.patchdatas.get_element_count(),
            ghost_layers_candidates.values.get_element_count()));
    }

    // iterate on both DDShared containers
    for (; ghost_layer_it != ghost_layer.patchdatas.end();
         ++ghost_layer_it, ++ghost_layer_info_it) {

        auto [sender, receiver] = ghost_layer_it->first;

        shamrock::patch::PatchDataLayer &ghost_layer_element = ghost_layer_it->second;
        auto &sender_ghost_layer_info                        = ghost_layer_info_it->second;

        auto &block_min_buf = ghost_layer_element.get_field<TgridVec>(iblock_min).get_buf();
        auto &block_max_buf = ghost_layer_element.get_field<TgridVec>(iblock_max).get_buf();
        auto xoff           = sender_ghost_layer_info.xoff;
        auto yoff           = sender_ghost_layer_info.yoff;
        auto zoff           = sender_ghost_layer_info.zoff;
        // transform the block min and max
        sham::kernel_call(
            q,
            sham::MultiRef{},
            sham::MultiRef{block_min_buf, block_max_buf},
            ghost_layer_element.get_obj_cnt(),
            [paving_function, xoff, yoff, zoff](
                u32 i, TgridVec *__restrict block_min, TgridVec *__restrict block_max) {
                shammath::AABB<TgridVec> block_box = {block_min[i], block_max[i]};

                block_box = paving_function.f_aabb(block_box, xoff, yoff, zoff);

                block_min[i] = block_box.lower;
                block_max[i] = block_box.upper;
            });

        // do not forget that while we have transformed the ghost layer block bound we did not
        // transform the ghost layer data Especially if the paving is reflexive a permutation needs
        // to be applied to the ghost layer data

        using AMRBlock                     = shammodels::amr::AMRBlock<Tvec, TgridVec, 1>;
        static constexpr u32 block_size    = AMRBlock::block_size;
        static constexpr u32 expected_nvar = AMRBlock::block_size;
        static constexpr u32 dim           = shambase::VectorProperties<TgridVec>::dimension;

        auto compute_field_var_permut = [&paving_function, xoff, yoff, zoff]() -> std::vector<u32> {
            // get coord list per cell in the block
            std::array<std::array<u32, dim>, block_size> coord_list = {};
            for (u32 i = 0; i < block_size; i++) {
                coord_list[i] = AMRBlock::get_coord(i);
            }

            // apply the paving function to the coord list
            std::array<TgridVec, block_size> coord_list_transformed = {};
            for (u32 i = 0; i < block_size; i++) {
                TgridVec coord            = {coord_list[i][0], coord_list[i][1], coord_list[i][2]};
                coord_list_transformed[i] = paving_function.f(coord, xoff, yoff, zoff);
            }

            // get the min and max coord in the block
            TgridVec min_coord = coord_list_transformed[0];
            TgridVec max_coord = coord_list_transformed[0];
            for (u32 i = 1; i < block_size; i++) {
                min_coord = sham::min(min_coord, coord_list_transformed[i]);
                max_coord = sham::max(max_coord, coord_list_transformed[i]);
            }

            // compute the permut
            std::vector<u32> permut(block_size);
            for (u32 i = 0; i < block_size; i++) {
                TgridVec new_coord = coord_list_transformed[i] - min_coord;
                std::array<u32, dim> new_coord_arr
                    = {static_cast<u32>(new_coord[0]),
                       static_cast<u32>(new_coord[1]),
                       static_cast<u32>(new_coord[2])};
                permut[i] = AMRBlock::get_index(new_coord_arr);
            }

            return permut;
        };

        std::vector<u32> permut = compute_field_var_permut();

        // apply the permut to the field
        auto apply_permut = [&](auto &field) {
            u32 nvar = field.get_nvar();

            if (nvar == expected_nvar) {
                field.permut_vars(permut);
            } else if (nvar % expected_nvar == 0) {

                u32 fact_expand = nvar / expected_nvar;

                std::vector<u32> new_permut(nvar);
                for (u32 i = 0; i < expected_nvar; i++) {
                    for (u32 j = 0; j < fact_expand; j++) {
                        new_permut[i * fact_expand + j] = permut[i] * fact_expand + j;
                    }
                }
                field.permut_vars(new_permut);
            } else if (nvar == 1) {
                // do nothing
            } else {
                throw shambase::make_except_with_loc<std::runtime_error>(shambase::format(
                    "the number of variables is not equal to the expected number of variables: {} "
                    "!= {}, field name: {}, layout: {}",
                    nvar,
                    expected_nvar,
                    field.get_name(),
                    ghost_layer_element.get_layout_ptr()->get_description_str()));
            }
        };

        ghost_layer_element.for_each_field_any(apply_permut);

        // Ok this is wierd but it works
        // The idea is that the paving does not deform, it only translates and invert. As such
        // if we supply two points whose difference is 1 in every compoenent, the diff after paving
        // is the termwise multiplication to be applied to vector quantities.
        // Periodic x:
        //   (0,0,0) -> (1,1,1) becomes (1,1,1) -> (2,2,2), delta = (1,1,1)
        // Reflective x:
        //   (0,0,0) -> (1,1,1) becomes (1,1,1) -> (0,2,2), delta = (-1,1,1)
        //   so a vector get its x component inverted
        // TODO: add that to the doc on paving functions
        auto get_termwise_mul_vec = [&]() {
            TgridVec p0 = {0, 0, 0};
            TgridVec p1 = {1, 1, 1};

            TgridVec p0_transformed = paving_function.f(p0, xoff, yoff, zoff);
            TgridVec p1_transformed = paving_function.f(p1, xoff, yoff, zoff);

            TgridVec mul_compo_vec = p1_transformed - p0_transformed;

            return mul_compo_vec;
        };

        using Tscal = typename shambase::VectorProperties<Tvec>::component_type;

        auto mul_compo_vec = get_termwise_mul_vec().template convert<Tscal>();

        auto transform_vecs = [&](auto &field) {
            using T = typename std::decay_t<decltype(field)>::Field_type;
            if constexpr (std::is_same_v<T, Tvec>) {
                auto &buf = field.get_buf();
                sham::kernel_call(
                    q,
                    sham::MultiRef{},
                    sham::MultiRef{buf},
                    field.get_obj_cnt() * field.get_nvar(),
                    [mul_compo_vec](u32 i, Tvec *__restrict vec) {
                        vec[i] = vec[i] * mul_compo_vec;
                    });
            } else if constexpr (std::is_same_v<T, TgridVec>) {
            } else if constexpr (std::is_same_v<T, Tscal>) {
            } else {
                shambase::throw_unimplemented();
            }
        };

        if ((xoff != 0 && transform_vec_x) || (yoff != 0 && transform_vec_y)
            || (zoff != 0 && transform_vec_z)) {
            ghost_layer_element.for_each_field_any(transform_vecs);
        }
    }
}

template<class Tvec, class TgridVec>
std::string shammodels::basegodunov::modules::TransformGhostLayer<Tvec, TgridVec>::_impl_get_tex()
    const {
    return "TODO";
}

template class shammodels::basegodunov::modules::TransformGhostLayer<f64_3, i64_3>;
