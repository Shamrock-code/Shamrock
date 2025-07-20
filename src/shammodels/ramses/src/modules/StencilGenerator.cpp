// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright (c) 2021-2024 Timothée David--Cléris <tim.shamrock@proton.me>
// SPDX-License-Identifier: CeCILL Free Software License Agreement v2.1
// Shamrock is licensed under the CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

/**
 * @file StencilGenerator.cpp
 * @author Timothée David--Cléris (tim.shamrock@proton.me)
 * @brief
 *
 */

#include "shambase/aliases_int.hpp"
#include "shambackends/sycl.hpp"
#include "shammodels/common/amr/AMRBlockStencil.hpp"
#include "shammodels/common/amr/AMRCellStencil.hpp"
#include "shammodels/ramses/modules/StencilGenerator.hpp"

template<class Tvec, class TgridVec, class Objiter>
void _kernel(
    u32 id_a,
    TgridVec relative_pos,
    Objiter cell_looper,
    TgridVec const *cell_min,
    TgridVec const *cell_max,
    shammodels::amr::block::StencilElement *stencil_info) {

    namespace block = shammodels::amr::block;

    block::StencilElement ret = block::StencilElement::make_none();

    shammath::AABB<TgridVec> cell_aabb{cell_min[id_a], cell_max[id_a]};
    cell_aabb.lower += relative_pos;
    cell_aabb.upper += relative_pos;

    std::array<u32, 8> found_cells = {u32_max};
    u32 cell_found_count           = 0;

    cell_looper.rtree_for(
        [&](u32 node_id, TgridVec bmin, TgridVec bmax) -> bool {
            return shammath::AABB<TgridVec>{bmin, bmax}
                .get_intersect(cell_aabb)
                .is_volume_not_null();
        },
        [&](u32 id_b) {
            bool interact = shammath::AABB<TgridVec>{cell_min[id_b], cell_max[id_b]}
                                .get_intersect(cell_aabb)
                                .is_volume_not_null();

            if (interact) {
                if (cell_found_count < 8) {
                    found_cells[cell_found_count] = id_b;
                }
                cell_found_count++;
            }
        });

    auto cell_found_0_aabb
        = shammath::AABB<TgridVec>{cell_min[found_cells[0]], cell_max[found_cells[0]]};

    // delt is the linear delta, so it's a factor 2 instead of 8
    bool check_levelp1
        = sham::equals(cell_found_0_aabb.delt() * 2, cell_aabb.delt()) && (cell_found_count == 8);

    bool check_levelm1
        = sham::equals(cell_found_0_aabb.delt(), cell_aabb.delt() * 2) && (cell_found_count == 1);

    bool check_levelsame
        = sham::equals(cell_found_0_aabb.delt(), cell_aabb.delt()) && (cell_found_count == 1);

    i32 state = i32(check_levelsame) + i32(check_levelm1) * 2 + i32(check_levelp1) * 4;

    switch (state) {
    case 1: ret = block::StencilElement::make_same_level(block::SameLevel{found_cells[0]}); break;
    case 2: ret = block::StencilElement::make_level_m1(block::Levelm1{found_cells[0]}); break;

    // for case 4 (level p1) sort elements in corect order
    // cf in block cell lowering
    // u32 child_select = mod_coord[0]*4 + mod_coord[1]*2 + mod_coord[2];
    case 4: ret = block::StencilElement::make_level_p1(block::Levelp1{found_cells}); break;
    }

    stencil_info[id_a] = ret;
}

template<class Tvec, class TgridVec>
auto shammodels::basegodunov::modules::StencilGenerator<Tvec, TgridVec>::compute_block_stencil_slot(
    i64_3 relative_pos, StencilOffsets result_offset) -> dd_block_stencil_el_buf {

    StackEntry stack_loc{};

    using MergedPDat = shamrock::MergedPatchData;
    using RTree      = typename Storage::RTree;

    shambase::DistributedData<std::unique_ptr<sycl::buffer<amr::block::StencilElement>>>
        block_stencil_element;

    shambase::get_check_ref(storage.trees).trees.for_each([&](u64 id, RTree &tree) {
        u32 leaf_count          = tree.tree_reduced_morton_codes.tree_leaf_count;
        u32 internal_cell_count = tree.tree_struct.internal_cell_count;
        u32 tot_count           = leaf_count + internal_cell_count;

        MergedPDat &mpdat = storage.merged_patchdata_ghost.get().get(id);

        sycl::buffer<TgridVec> &tree_bmin
            = shambase::get_check_ref(tree.tree_cell_ranges.buf_pos_min_cell_flt);
        sycl::buffer<TgridVec> &tree_bmax
            = shambase::get_check_ref(tree.tree_cell_ranges.buf_pos_max_cell_flt);

        sycl::buffer<amr::block::StencilElement> stencil_block_info(mpdat.total_elements);

        sham::DeviceBuffer<TgridVec> &buf_cell_min = mpdat.pdat.get_field_buf_ref<TgridVec>(0);
        sham::DeviceBuffer<TgridVec> &buf_cell_max = mpdat.pdat.get_field_buf_ref<TgridVec>(1);

        sham::DeviceQueue &q = shamsys::instance::get_compute_scheduler().get_queue();
        sham::EventList depends_list;

        auto acc_cell_min = buf_cell_min.get_read_access(depends_list);
        auto acc_cell_max = buf_cell_max.get_read_access(depends_list);

        auto e = q.submit(depends_list, [&](sycl::handler &cgh) {
            shamrock::tree::ObjectIterator cell_looper(tree, cgh);

            sycl::accessor acc_stencil_info{
                stencil_block_info, cgh, sycl::write_only, sycl::no_init};

            shambase::parralel_for(
                cgh, mpdat.total_elements, "compute neigh cache 1", [=](u64 gid) {
                    u32 id_a = (u32) gid;

                    _kernel<Tvec, TgridVec>(
                        id_a,
                        relative_pos,
                        cell_looper,
                        acc_cell_min,
                        acc_cell_max,
                        acc_stencil_info.get_pointer());
                });
        });

        buf_cell_min.complete_event_state(e);
        buf_cell_max.complete_event_state(e);

        block_stencil_element.add_obj(
            id, std::make_unique<sycl::buffer<amr::block::StencilElement>>(stencil_block_info));
    });

    return block_stencil_element;
}

template<class Tvec, class TgridVec>
auto shammodels::basegodunov::modules::StencilGenerator<Tvec, TgridVec>::lower_block_slot_to_cell(
    i64_3 relative_pos, StencilOffsets result_offset, dd_block_stencil_el_buf &block_stencil_el)
    -> dd_cell_stencil_el_buf {

    return block_stencil_el.map<std::unique_ptr<cell_stencil_el_buf>>(
        [&](u64 id, std::unique_ptr<block_stencil_el_buf> &block_stencil_el_b) {
            return std::make_unique<cell_stencil_el_buf>(lower_block_slot_to_cell(
                id, relative_pos, result_offset, shambase::get_check_ref(block_stencil_el_b)));
        });
}

////////////////////////////////////////////////////////////////////////////////////////////////////
// Lowering kernel
////////////////////////////////////////////////////////////////////////////////////////////////////

template<class Tvec, class TgridVec>
auto shammodels::basegodunov::modules::StencilGenerator<Tvec, TgridVec>::lower_block_slot_to_cell(
    u64 patch_id,
    i64_3 relative_pos,
    StencilOffsets result_offset,
    block_stencil_el_buf &block_stencil_el) -> cell_stencil_el_buf {

    u32 block_count = block_stencil_el.size();
    u32 cell_count  = block_count * AMRBlock::block_size;

    sycl::buffer<amr::cell::StencilElement> ret(cell_count);

    using MergedPDat                           = shamrock::MergedPatchData;
    MergedPDat &mpdat                          = storage.merged_patchdata_ghost.get().get(patch_id);
    sham::DeviceBuffer<TgridVec> &buf_cell_min = mpdat.pdat.get_field_buf_ref<TgridVec>(0);
    sham::DeviceBuffer<TgridVec> &buf_cell_max = mpdat.pdat.get_field_buf_ref<TgridVec>(1);

    // shambase::check_buffer_size(buf_cell_min, block_count);
    // shambase::check_buffer_size(buf_cell_max, block_count);

    sham::DeviceQueue &q = shamsys::instance::get_compute_scheduler().get_queue();
    sham::EventList depends_list;

    auto acc_block_min = buf_cell_min.get_read_access(depends_list);
    auto acc_block_max = buf_cell_max.get_read_access(depends_list);

    auto e = q.submit(depends_list, [&](sycl::handler &cgh) {
        sycl::accessor acc_stencil_block{block_stencil_el, cgh, sycl::read_only};
        sycl::accessor acc_stencil_cell{ret, cgh, sycl::write_only, sycl::no_init};

        shambase::parralel_for(cgh, cell_count, "block stencil to cell lowering", [=](u64 gid) {
            u32 cell_global_id = (u32) gid;

            u32 block_id    = cell_global_id / AMRBlock::block_size;
            u32 cell_loc_id = cell_global_id % AMRBlock::block_size;

            amr::block::StencilElement block_stencil = acc_stencil_block[block_id];

            // fetch block info
            TgridVec cblock_min = acc_block_min[block_id];
            TgridVec cblock_max = acc_block_max[block_id];
            TgridVec delta_cell = (cblock_max - cblock_min) / AMRBlock::Nside;

            // Compute wanted neighbourg cell bounds
            std::array<u32, 3> lcoord_arr = AMRBlock::get_coord(cell_loc_id);
            TgridVec lcoord               = {lcoord_arr[0], lcoord_arr[1], lcoord_arr[2]};

            TgridVec ccell_neigh_min = cblock_min + relative_pos + lcoord * delta_cell;
            TgridVec ccell_neigh_max
                = cblock_min + relative_pos + (lcoord + TgridVec{1, 1, 1}) * delta_cell;

            shammath::AABB<TgridVec> ccell_neigh = {ccell_neigh_min, ccell_neigh_max};

            // clang-format off
            // This block fetch the neigh block id and bounds
            u64 block_id_neigh = u64_max;
            TgridVec cblock_neigh_min = {0,0,0};
            TgridVec cblock_neigh_max = {AMRBlock::Nside,AMRBlock::Nside,AMRBlock::Nside};
            block_stencil.visitor(
                [&](amr::block::SameLevel st) {
                    cblock_neigh_min = acc_block_min[st.block_idx];
                    cblock_neigh_max = acc_block_max[st.block_idx];
                    block_id_neigh   = st.block_idx;
                },
                [&](amr::block::Levelm1 st) {
                    cblock_neigh_min = acc_block_min[st.block_idx];
                    cblock_neigh_max = acc_block_max[st.block_idx];
                    block_id_neigh   = st.block_idx;
                },
                [&](amr::block::Levelp1 st) {

                    #pragma unroll
                    for (u32 i = 0; i < split_count; i++) {
                        TgridVec tmp_cblock_neigh_min = acc_block_min[st.block_child_idxs[i]];
                        TgridVec tmp_cblock_neigh_max = acc_block_max[st.block_child_idxs[i]];
                        if (shammath::AABB<TgridVec>(tmp_cblock_neigh_min, tmp_cblock_neigh_max)
                                .get_intersect(ccell_neigh)
                                .is_volume_not_null()) {
                            cblock_neigh_min = tmp_cblock_neigh_min;
                            cblock_neigh_max = tmp_cblock_neigh_max;
                            block_id_neigh   = st.block_child_idxs[i];
                        }
                    }


                    // if none are found (which should be impossible anyway) block_id_neigh =
                    // u64_max so it will give none
                },
                [&](amr::block::None st) {
                    // do nothing as by default block_id_neigh = u64_max
                });
            // clang-format on

            // Now find the local index of the neighbouring block
            TgridVec delta_cell_neigh = (cblock_neigh_max - cblock_neigh_min) / AMRBlock::Nside;
            TgridVec cell_neigh_relat = ccell_neigh_min - cblock_neigh_min;
            TgridVec lcoord_neigh     = cell_neigh_relat / delta_cell_neigh;

            u64 cell_idx_global = block_id_neigh * AMRBlock::block_size
                                  + AMRBlock::get_index(
                                      {static_cast<unsigned int>(lcoord_neigh.x()),
                                       static_cast<unsigned int>(lcoord_neigh.y()),
                                       static_cast<unsigned int>(lcoord_neigh.z())});

            amr::cell::StencilElement ret = amr::cell::StencilElement::make_none();
            if (block_id_neigh != u64_max) {
                block_stencil.visitor(
                    [&](amr::block::SameLevel st) {
                        ret = amr::cell::StencilElement::make_same_level({cell_idx_global});
                    },
                    [&](amr::block::Levelm1 st) {
                        ret = amr::cell::StencilElement::make_level_m1({cell_idx_global});
                    },
                    [&](amr::block::Levelp1 st) {
                        // Enumerate all blocks using linearity of the indexing, all
                        // possibilities of 0 and 1 on each coord (8 cells)
                        // see amr::cell::Levelp1::for_all_indexes
                        ret = amr::cell::StencilElement::make_level_p1({cell_idx_global});
                    },
                    [&](amr::block::None st) {
                        // do nothing as by default block_id_neigh = u64_max
                    });
            }

            acc_stencil_cell[cell_global_id] = ret;
        });
    });

    buf_cell_min.complete_event_state(e);
    buf_cell_max.complete_event_state(e);

    return ret;
}

////////////////////////////////////////////////////////////////////////////////////////////////////
// Call to generate all stencil elements
////////////////////////////////////////////////////////////////////////////////////////////////////

template<class Tvec, class TgridVec>
void shammodels::basegodunov::modules::StencilGenerator<Tvec, TgridVec>::make_stencil() {
    auto block_stencil_xp1 = compute_block_stencil_slot(i64_3{+1, 0, 0}, xp1);
    auto block_stencil_xm1 = compute_block_stencil_slot(i64_3{-1, 0, 0}, xm1);
    auto block_stencil_yp1 = compute_block_stencil_slot(i64_3{0, +1, 0}, yp1);
    auto block_stencil_ym1 = compute_block_stencil_slot(i64_3{0, -1, 0}, ym1);
    auto block_stencil_zp1 = compute_block_stencil_slot(i64_3{0, 0, +1}, zp1);
    auto block_stencil_zm1 = compute_block_stencil_slot(i64_3{0, 0, -1}, zm1);

    /*
    storage.stencil.set(AMRStencilCache{});

    storage.stencil.get().insert_data(
        xp1, lower_block_slot_to_cell(i64_3{+1, 0, 0}, xp1, block_stencil_xp1));
    storage.stencil.get().insert_data(
        xm1, lower_block_slot_to_cell(i64_3{-1, 0, 0}, xm1, block_stencil_xm1));
    storage.stencil.get().insert_data(
        yp1, lower_block_slot_to_cell(i64_3{0, +1, 0}, yp1, block_stencil_yp1));
    storage.stencil.get().insert_data(
        ym1, lower_block_slot_to_cell(i64_3{0, -1, 0}, ym1, block_stencil_ym1));
    storage.stencil.get().insert_data(
        zp1, lower_block_slot_to_cell(i64_3{0, 0, +1}, zp1, block_stencil_zp1));
    storage.stencil.get().insert_data(
        zm1, lower_block_slot_to_cell(i64_3{0, 0, -1}, zm1, block_stencil_zm1));
    */
}

template class shammodels::basegodunov::modules::StencilGenerator<f64_3, i64_3>;
