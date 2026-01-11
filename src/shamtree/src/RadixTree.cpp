// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright (c) 2021-2025 Timothée David--Cléris <tim.shamrock@proton.me>
// SPDX-License-Identifier: CeCILL Free Software License Agreement v2.1
// Shamrock is licensed under the CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

/**
 * @file RadixTree.cpp
 * @author Timothée David--Cléris (tim.shamrock@proton.me)
 * @brief
 */

#include "shambase/floats.hpp"
#include "shambase/integer.hpp"
#include "shamalgs/memory.hpp"
#include "shamsys/NodeInstance.hpp"
#include "shamsys/legacy/sycl_handler.hpp"
#include "shamtree/MortonKernels.hpp"
#include "shamtree/RadixTree.hpp"
#include "shamtree/RadixTreeMortonBuilder.hpp"
#include <tuple>
#include <vector>

template<class u_morton, class vec3>
RadixTree<u_morton, vec3>::RadixTree(
    sycl::queue &queue,
    std::tuple<vec3, vec3> treebox,
    sycl::buffer<vec3> &pos_buf,
    u32 cnt_obj,
    u32 reduc_level) {
    if (cnt_obj > i32_max - 1) {
        throw shambase::make_except_with_loc<std::runtime_error>(
            "number of element in patch above i32_max-1");
    }

    shamlog_debug_sycl_ln("RadixTree", "box dim :", std::get<0>(treebox), std::get<1>(treebox));

    bounding_box = treebox;

    tree_morton_codes.build(queue, shammath::CoordRange<vec3>{treebox}, cnt_obj, pos_buf);

    bool one_cell_mode;

    tree_reduced_morton_codes.build(
        queue, tree_morton_codes.obj_cnt, reduc_level, tree_morton_codes, one_cell_mode);

    if (!one_cell_mode) {
        tree_struct.build(
            queue,
            tree_reduced_morton_codes.tree_leaf_count - 1,
            *tree_reduced_morton_codes.buf_tree_morton);
    } else {
        tree_struct.build_one_cell_mode();
    }
}

template<class u_morton, class vec3>
RadixTree<u_morton, vec3>::RadixTree(
    sycl::queue &queue,
    std::tuple<vec3, vec3> treebox,
    const std::unique_ptr<sycl::buffer<vec3>> &pos_buf,
    u32 cnt_obj,
    u32 reduc_level)
    : RadixTree(queue, treebox, shambase::get_check_ref(pos_buf), cnt_obj, reduc_level) {}

template<class u_morton, class Tvec>
RadixTree<u_morton, Tvec>::RadixTree(
    sham::DeviceScheduler_ptr dev_sched,
    std::tuple<Tvec, Tvec> treebox,
    sham::DeviceBuffer<Tvec> &pos_buf,
    u32 cnt_obj,
    u32 reduc_level) {

    sycl::queue &queue = dev_sched->get_queue().q;

    if (cnt_obj > i32_max - 1) {
        throw shambase::make_except_with_loc<std::runtime_error>(
            "number of element in patch above i32_max-1");
    }

    shamlog_debug_sycl_ln("RadixTree", "box dim :", std::get<0>(treebox), std::get<1>(treebox));

    bounding_box = treebox;

    tree_morton_codes.build(dev_sched, shammath::CoordRange<Tvec>{treebox}, cnt_obj, pos_buf);

    bool one_cell_mode;

    tree_reduced_morton_codes.build(
        queue, tree_morton_codes.obj_cnt, reduc_level, tree_morton_codes, one_cell_mode);

    if (!one_cell_mode) {
        tree_struct.build(
            queue,
            tree_reduced_morton_codes.tree_leaf_count - 1,
            *tree_reduced_morton_codes.buf_tree_morton);
    } else {
        tree_struct.build_one_cell_mode();
    }
}

template<class u_morton, class vec3>
void RadixTree<u_morton, vec3>::serialize(shamalgs::SerializeHelper &serializer) {
    StackEntry stack_loc{};

    serializer.write(std::get<0>(bounding_box));
    serializer.write(std::get<1>(bounding_box));
    tree_morton_codes.serialize(serializer);
    tree_reduced_morton_codes.serialize(serializer);
    tree_struct.serialize(serializer);
    tree_cell_ranges.serialize(serializer);
}

template<class u_morton, class pos_t>
shamalgs::SerializeSize RadixTree<u_morton, pos_t>::serialize_byte_size() {
    using H = shamalgs::SerializeHelper;
    return H::serialize_byte_size<pos_t>() * 2 + tree_morton_codes.serialize_byte_size()
           + tree_reduced_morton_codes.serialize_byte_size() + tree_struct.serialize_byte_size()
           + tree_cell_ranges.serialize_byte_size();
}

template<class u_morton, class pos_t>
RadixTree<u_morton, pos_t> RadixTree<u_morton, pos_t>::deserialize(
    shamalgs::SerializeHelper &serializer) {
    StackEntry stack_loc{};

    RadixTree ret;

    serializer.load(std::get<0>(ret.bounding_box));
    serializer.load(std::get<1>(ret.bounding_box));

    using namespace shamrock::tree;

    ret.tree_morton_codes         = TreeMortonCodes<u_morton>::deserialize(serializer);
    ret.tree_reduced_morton_codes = TreeReducedMortonCodes<u_morton>::deserialize(serializer);
    ret.tree_struct               = TreeStructure<u_morton>::deserialize(serializer);
    ret.tree_cell_ranges          = TreeCellRanges<u_morton, pos_t>::deserialize(serializer);

    return ret;
}

template<class u_morton, class vec3>
void RadixTree<u_morton, vec3>::compute_cell_ibounding_box(sycl::queue &queue) {
    StackEntry stack_loc{};
    tree_cell_ranges.build1(queue, tree_reduced_morton_codes, tree_struct);
}

template<class morton_t, class pos_t>
void RadixTree<morton_t, pos_t>::convert_bounding_box(sycl::queue &queue) {
    StackEntry stack_loc{};
    u32 total_count = tree_struct.internal_cell_count + tree_reduced_morton_codes.tree_leaf_count;
    tree_cell_ranges.build2(queue, total_count, bounding_box);
}

template<class u_morton, class vec>
auto RadixTree<u_morton, vec>::compute_int_boxes(
    sycl::queue &queue, sham::DeviceBuffer<coord_t> &int_rad_buf, coord_t tolerance)
    -> RadixTreeField<coord_t> {

    shamlog_debug_sycl_ln("RadixTree", "compute int boxes");

    auto buf_cell_interact_rad = RadixTreeField<coord_t>::make_empty(
        1, tree_struct.internal_cell_count + tree_reduced_morton_codes.tree_leaf_count);
    sycl::range<1> range_leaf_cell{tree_reduced_morton_codes.tree_leaf_count};

    auto &buf_cell_int_rad_buf = buf_cell_interact_rad.radix_tree_field_buf;

    sham::DeviceQueue q = shamsys::instance::get_compute_scheduler().get_queue();
    sham::EventList depends_list;

    auto h = int_rad_buf.get_read_access(depends_list);

    auto e = q.submit(depends_list, [&](sycl::handler &cgh) {
        u32 offset_leaf = tree_struct.internal_cell_count;

        auto h_max_cell
            = buf_cell_int_rad_buf->template get_access<sycl::access::mode::discard_write>(cgh);

        auto cell_particle_ids = tree_reduced_morton_codes.buf_reduc_index_map
                                     ->template get_access<sycl::access::mode::read>(cgh);
        auto particle_index_map = tree_morton_codes.buf_particle_index_map
                                      ->template get_access<sycl::access::mode::read>(cgh);

        coord_t tol = tolerance;

        cgh.parallel_for(range_leaf_cell, [=](sycl::item<1> item) {
            u32 gid = (u32) item.get_id(0);

            u32 min_ids = cell_particle_ids[gid];
            u32 max_ids = cell_particle_ids[gid + 1];
            f32 h_tmp   = 0;

            for (unsigned int id_s = min_ids; id_s < max_ids; id_s++) {

                f32 h_a = h[particle_index_map[id_s]] * tol;
                h_tmp   = (h_tmp > h_a ? h_tmp : h_a);
            }

            h_max_cell[offset_leaf + gid] = h_tmp;
        });
    });

    int_rad_buf.complete_event_state(e);

#if false
    // debug code to track the DPCPP + prime number worker issue
    {

        //172827
        //86413
        //<<<(43207,1,1),(2,1,1)>>>
        //gid = 86412

        shamalgs::memory::print_buf(*tree_struct.buf_rchild_id, tree_struct.internal_cell_count, 16, "{} ");
        shamalgs::memory::print_buf(*tree_struct.buf_lchild_id, tree_struct.internal_cell_count, 16, "{} ");
        shamalgs::memory::print_buf(*tree_struct.buf_rchild_flag, tree_struct.internal_cell_count, 16, "{} ");
        shamalgs::memory::print_buf(*tree_struct.buf_lchild_flag, tree_struct.internal_cell_count, 16, "{} ");


            sycl::host_accessor rchild_id   {*tree_struct.buf_rchild_id  ,sycl::read_only};
            sycl::host_accessor lchild_id   {*tree_struct.buf_lchild_id  ,sycl::read_only};
            sycl::host_accessor rchild_flag {*tree_struct.buf_rchild_flag,sycl::read_only};
            sycl::host_accessor lchild_flag {*tree_struct.buf_lchild_flag,sycl::read_only};

            u32 gid = 86412;
            u32 lid_0 = lchild_id[gid];
            u32 rid_0 = rchild_id[gid];
            u32 lfl_0 = lchild_flag[gid];
            u32 rfl_0 = rchild_flag[gid];
            u32 offset_leaf = tree_struct.internal_cell_count;
            u32 lid = lchild_id[gid] + offset_leaf * lchild_flag[gid];
            u32 rid = rchild_id[gid] + offset_leaf * rchild_flag[gid];

            logger::raw_ln("gid",gid);
            logger::raw_ln("lid_0",lid_0);
            logger::raw_ln("rid_0",rid_0);
            logger::raw_ln("lfl_0",lfl_0);
            logger::raw_ln("rfl_0",rfl_0);
            logger::raw_ln("offset_leaf",offset_leaf);
            logger::raw_ln("lid",lid);
            logger::raw_ln("rid",rid);
            logger::raw_ln("sz =", buf_cell_int_rad_buf->size());
            logger::raw_ln("internal_cell_count =", tree_struct.internal_cell_count);
            logger::raw_ln("tree_leaf_count =", tree_reduced_morton_codes.tree_leaf_count);
    }
#endif

    sycl::range<1> range_tree{tree_struct.internal_cell_count};

    for (u32 i = 0; i < tree_depth; i++) {
        queue.submit([&](sycl::handler &cgh) {
            u32 offset_leaf = tree_struct.internal_cell_count;

            sycl::accessor h_max_cell{*buf_cell_int_rad_buf, cgh, sycl::read_write};

            sycl::accessor rchild_id{*tree_struct.buf_rchild_id, cgh, sycl::read_only};
            sycl::accessor lchild_id{*tree_struct.buf_lchild_id, cgh, sycl::read_only};
            sycl::accessor rchild_flag{*tree_struct.buf_rchild_flag, cgh, sycl::read_only};
            sycl::accessor lchild_flag{*tree_struct.buf_lchild_flag, cgh, sycl::read_only};

            u32 len                  = tree_struct.internal_cell_count;
            constexpr u32 group_size = 64;
            u32 max_len              = len;
            u32 group_cnt            = shambase::group_count(len, group_size);
            u32 corrected_len        = group_cnt * group_size;

            cgh.parallel_for(
                sycl::nd_range<1>{corrected_len, group_size}, [=](sycl::nd_item<1> id) {
                    u32 local_id      = id.get_local_id(0);
                    u32 group_tile_id = id.get_group_linear_id();
                    u32 gid           = group_tile_id * group_size + local_id;

                    if (gid >= max_len)
                        return;

                    u32 lid = lchild_id[gid] + offset_leaf * lchild_flag[gid];
                    u32 rid = rchild_id[gid] + offset_leaf * rchild_flag[gid];

                    coord_t h_l = h_max_cell[lid];
                    coord_t h_r = h_max_cell[rid];

                    h_max_cell[gid] = (h_r > h_l ? h_r : h_l);
                });
        });
    }

    {
        if (shamalgs::reduction::has_nan(
                queue,
                *buf_cell_int_rad_buf,
                tree_struct.internal_cell_count + tree_reduced_morton_codes.tree_leaf_count)) {
            shamalgs::memory::print_buf(
                *buf_cell_int_rad_buf,
                tree_struct.internal_cell_count + tree_reduced_morton_codes.tree_leaf_count,
                8,
                "{} ");
            throw shambase::make_except_with_loc<std::runtime_error>(
                "the structure of the tree as issue in ids");
        }
    }

    return std::move(buf_cell_interact_rad);
}

template<class T>
std::string print_member(const T &a);

template<>
std::string print_member(const u8 &a) {
    return shambase::format_printf("%d", u32(a));
}

template<>
std::string print_member(const u32 &a) {
    return shambase::format_printf("%d", a);
}

template<class u_morton, class vec3>
template<class T>
void RadixTree<u_morton, vec3>::print_tree_field(sycl::buffer<T> &buf_field) {

    sycl::host_accessor acc{buf_field, sycl::read_only};

    u32 total_count = tree_struct.internal_cell_count + tree_reduced_morton_codes.tree_leaf_count;

    u32 offset_leaf = tree_struct.internal_cell_count;

    sycl::host_accessor rchild_id{*tree_struct.buf_rchild_id};
    sycl::host_accessor lchild_id{*tree_struct.buf_lchild_id};
    sycl::host_accessor rchild_flag{*tree_struct.buf_rchild_flag};
    sycl::host_accessor lchild_flag{*tree_struct.buf_lchild_flag};

    // start allow utf-8
    auto printer = [&]() {
        auto get_print_step
            = [&](u32 gid, std::string prefix, bool is_left, auto &step_ref) -> std::string {
            std::string ret_val = "";

            if (!is_left) {
                ret_val += prefix;
            }

            std::string val      = " (" + print_member(acc[gid]) + ") ";
            std::string val_empt = std::string(val.size(), ' ');

            ret_val += (is_left ? "╦══" : "╚══");
            ret_val += val;

            if (gid < offset_leaf) {
                u32 lid = lchild_id[gid] + offset_leaf * lchild_flag[gid];
                u32 rid = rchild_id[gid] + offset_leaf * rchild_flag[gid];

                ret_val += step_ref(
                    lid, prefix + (is_left ? "║  " + val_empt : "   " + val_empt), true, step_ref);
                ret_val += step_ref(
                    rid, prefix + (is_left ? "║  " + val_empt : "   " + val_empt), false, step_ref);
            } else {
                ret_val += "\n";
            }

            return ret_val;
        };

        logger::raw_ln(get_print_step(0, "", false, get_print_step));
    };
    // end allow utf-8

    printer();
}

template void RadixTree<u32, f64_3>::print_tree_field(sycl::buffer<u32> &buf_field);
template void RadixTree<u32, f32_3>::print_tree_field(sycl::buffer<u32> &buf_field);
template void RadixTree<u64, f64_3>::print_tree_field(sycl::buffer<u32> &buf_field);
template void RadixTree<u64, f32_3>::print_tree_field(sycl::buffer<u32> &buf_field);

template void RadixTree<u32, u32_3>::print_tree_field(sycl::buffer<u32> &buf_field);
template void RadixTree<u32, u64_3>::print_tree_field(sycl::buffer<u32> &buf_field);
template void RadixTree<u64, u32_3>::print_tree_field(sycl::buffer<u32> &buf_field);
template void RadixTree<u64, u64_3>::print_tree_field(sycl::buffer<u32> &buf_field);
template void RadixTree<u64, i64_3>::print_tree_field(sycl::buffer<u32> &buf_field);

template<class u_morton, class vec3>
typename RadixTree<u_morton, vec3>::CuttedTree RadixTree<u_morton, vec3>::cut_tree(
    sycl::queue &queue, sycl::buffer<u8> &valid_node) {

    u32 total_count = tree_struct.internal_cell_count + tree_reduced_morton_codes.tree_leaf_count;
    sycl::range<1> range_tree{total_count};

    {

        // flag 1 valid
        // flag 0 to be deleted
        // flag 2 anything below should be deleted (2 if initialy 0 & parent = 1)
        //  basically 2 is le thing that would end up in the excluded lambda part

        { // cascade zeros down the tree

            sycl::buffer<u8> valid_node_new = sycl::buffer<u8>(total_count);

            for (u32 it = 0; it < tree_depth; it++) {

                shamlog_debug_sycl_ln("Radixtree", "cascading zeros step : ", it);
                queue.submit([&](sycl::handler &cgh) {
                    sycl::accessor acc_valid_node_old{valid_node, cgh, sycl::read_only};
                    sycl::accessor acc_valid_node_new{
                        valid_node_new, cgh, sycl::write_only, sycl::no_init};

                    sycl::accessor acc_lchild_id{*tree_struct.buf_lchild_id, cgh, sycl::read_only};
                    sycl::accessor acc_rchild_id{*tree_struct.buf_rchild_id, cgh, sycl::read_only};
                    sycl::accessor acc_lchild_flag{
                        *tree_struct.buf_lchild_flag, cgh, sycl::read_only};
                    sycl::accessor acc_rchild_flag{
                        *tree_struct.buf_rchild_flag, cgh, sycl::read_only};

                    u32 leaf_offset = tree_struct.internal_cell_count;

                    cgh.parallel_for(
                        sycl::range<1>(tree_struct.internal_cell_count), [=](sycl::item<1> item) {
                            u32 lid = acc_lchild_id[item] + leaf_offset * acc_lchild_flag[item];
                            u32 rid = acc_rchild_id[item] + leaf_offset * acc_rchild_flag[item];

                            u8 old_nid_falg = acc_valid_node_old[item];

                            if (item.get_linear_id() == 0) {
                                acc_valid_node_new[item] = old_nid_falg;
                            }

                            if (old_nid_falg == 0 || old_nid_falg == 2) {
                                acc_valid_node_new[lid] = 0;
                                acc_valid_node_new[rid] = 0;
                            } else {
                                u8 old_lid_falg = acc_valid_node_old[lid];
                                u8 old_rid_falg = acc_valid_node_old[rid];

                                if (old_lid_falg == 0) {
                                    old_lid_falg = 2;
                                }
                                if (old_rid_falg == 0) {
                                    old_rid_falg = 2;
                                }

                                acc_valid_node_new[lid] = old_lid_falg;
                                acc_valid_node_new[rid] = old_rid_falg;
                            }
                        });
                });

                std::swap(valid_node, valid_node_new);
            }
        }

        //{
        //    shamlog_debug_sycl_ln("Radixtree", "valid_node_state");
        //    print_tree_field(valid_node);
        //    logger::raw_ln("");
        //}

        sycl::buffer<u8> valid_tree_morton(tree_reduced_morton_codes.tree_leaf_count);

        auto print_valid_morton = [&] {
            shamlog_debug_sycl_ln("Radixtree", "valid_tree_morton");

            sycl::buffer<u32> print_map(total_count);

            {

                sycl::host_accessor acc{print_map};
                sycl::host_accessor acc_leaf{valid_tree_morton};

                for (u32 i = 0; i < tree_reduced_morton_codes.tree_leaf_count; i++) {
                    acc[i + tree_struct.internal_cell_count] = acc_leaf[i];
                }

                for (u32 i = 0; i < tree_struct.internal_cell_count; i++) {
                    acc[i] = acc_leaf[i];
                }
            }

            print_tree_field(print_map);

            logger::raw_ln("");
        };

        queue.submit([&](sycl::handler &cgh) {
            sycl::accessor acc_valid_tree_morton{
                valid_tree_morton, cgh, sycl::write_only, sycl::no_init};

            sycl::accessor acc_valid_node{valid_node, cgh, sycl::read_only};

            u32 leaf_offset = tree_struct.internal_cell_count;

            cgh.parallel_for(
                sycl::range<1>(tree_reduced_morton_codes.tree_leaf_count), [=](sycl::item<1> item) {
                    u8 leaf_val = acc_valid_node[item.get_linear_id() + leaf_offset];

                    if (item.get_linear_id() < leaf_offset) {
                        if (acc_valid_node[item] == 2) {
                            leaf_val = 2;
                        }
                    }

                    acc_valid_tree_morton[item] = leaf_val;
                });
        });

        // print_valid_morton();

        // generate the new tree

        RadixTree ret;

        ret.bounding_box = bounding_box;

        std::vector<u32> extract_id;

        {

            std::vector<u_morton> new_buf_morton;
            std::vector<u32> new_buf_particle_index_map;
            std::vector<u32> new_reduc_index_map;

            u32 leaf_offset = tree_struct.internal_cell_count;

            sycl::host_accessor cell_index_map{
                *tree_reduced_morton_codes.buf_reduc_index_map, sycl::read_only};
            sycl::host_accessor particle_index_map{
                *tree_morton_codes.buf_particle_index_map, sycl::read_only};

            sycl::host_accessor acc_valid_tree_morton{valid_tree_morton, sycl::read_only};

            sycl::host_accessor acc_morton{*tree_morton_codes.buf_morton, sycl::read_only};

            u32 cnt = 0;

            for (u32 i = 0; i < tree_reduced_morton_codes.tree_leaf_count; i++) {
                if (acc_valid_tree_morton[i] != 0) {

                    {
                        // loop on particle indexes
                        uint min_ids = cell_index_map[i];
                        uint max_ids = cell_index_map[i + 1];

                        new_reduc_index_map.push_back(cnt);

                        for (unsigned int id_s = min_ids; id_s < max_ids; id_s++) {

                            // recover old index before morton sort
                            uint id_b = particle_index_map[id_s];

                            // iteration function
                            {
                                extract_id.push_back(id_b);
                                new_buf_morton.push_back(acc_morton[id_b]);
                                new_buf_particle_index_map.push_back(cnt);

                                cnt++;
                            }
                        }
                    }
                }
            }

            new_reduc_index_map.push_back(cnt);

            std::vector<u_morton> new_morton_tree;

            {
                sycl::host_accessor acc_tree_morton{*tree_reduced_morton_codes.buf_tree_morton};

                sycl::host_accessor acc_valid_tree_morton{valid_tree_morton, sycl::read_only};

                for (u32 i = 0; i < tree_reduced_morton_codes.tree_leaf_count; i++) {
                    if (acc_valid_tree_morton[i] != 0) {
                        new_morton_tree.push_back(acc_tree_morton[i]);
                    }
                }
            }

            ret.tree_reduced_morton_codes.tree_leaf_count = new_morton_tree.size();
            ret.tree_struct.internal_cell_count = ret.tree_reduced_morton_codes.tree_leaf_count - 1;

            ret.tree_morton_codes.buf_morton
                = std::make_unique<sycl::buffer<u_morton>>(new_buf_morton.size());
            {
                sycl::host_accessor acc{
                    *ret.tree_morton_codes.buf_morton, sycl::write_only, sycl::no_init};
                for (u32 i = 0; i < new_buf_morton.size(); i++) {
                    acc[i] = new_buf_morton[i];
                }
            }

            ret.tree_morton_codes.buf_particle_index_map
                = std::make_unique<sycl::buffer<u32>>(new_buf_particle_index_map.size());
            {
                sycl::host_accessor acc{
                    *ret.tree_morton_codes.buf_particle_index_map, sycl::write_only, sycl::no_init};
                for (u32 i = 0; i < new_buf_particle_index_map.size(); i++) {
                    acc[i] = new_buf_particle_index_map[i];
                }
            }

            if (ret.tree_reduced_morton_codes.tree_leaf_count > 1) {

                ret.tree_reduced_morton_codes.buf_reduc_index_map
                    = std::make_unique<sycl::buffer<u32>>(new_reduc_index_map.size());
                {
                    sycl::host_accessor acc{
                        *ret.tree_reduced_morton_codes.buf_reduc_index_map,
                        sycl::write_only,
                        sycl::no_init};
                    for (u32 i = 0; i < new_reduc_index_map.size(); i++) {
                        acc[i] = new_reduc_index_map[i];
                    }
                }

                ret.tree_reduced_morton_codes.buf_tree_morton
                    = std::make_unique<sycl::buffer<u_morton>>(new_morton_tree.size());
                {
                    sycl::host_accessor acc{
                        *ret.tree_reduced_morton_codes.buf_tree_morton,
                        sycl::write_only,
                        sycl::no_init};
                    for (u32 i = 0; i < new_morton_tree.size(); i++) {
                        acc[i] = new_morton_tree[i];
                    }
                }

                ret.tree_struct.build(
                    queue,
                    ret.tree_struct.internal_cell_count,
                    *ret.tree_reduced_morton_codes.buf_tree_morton);

            } else {
                throw ShamrockSyclException("not implemented");
            }
        }

        ret.compute_cell_ibounding_box(queue);
        ret.convert_bounding_box(queue);

#if false
        std::unique_ptr<sycl::buffer<u32>> new_node_id_to_old_naive = std::make_unique<sycl::buffer<u32>>(ret.tree_leaf_count + ret.tree_internal_count);

        {
            auto & new_node_id_to_old = new_node_id_to_old_naive;

            //junk fill
            {
                sycl::host_accessor acc{* new_node_id_to_old, sycl::write_only, sycl::no_init};
                for (u32 i = 0 ; i < new_node_id_to_old->size(); i++) {
                    acc[i] = u32_max;
                }
            }


            sycl::host_accessor acc_new_node_id_to_old {*new_node_id_to_old,sycl::write_only, sycl::no_init};

            sycl::host_accessor new_tree_acc_pos_min_cell{*ret.buf_pos_min_cell,sycl::read_only};
            sycl::host_accessor new_tree_acc_pos_max_cell{*ret.buf_pos_max_cell,sycl::read_only};

            sycl::host_accessor old_tree_acc_pos_min_cell{*buf_pos_min_cell,sycl::read_only};
            sycl::host_accessor old_tree_acc_pos_max_cell{*buf_pos_max_cell,sycl::read_only};

            for(u32 i = 0 ; i < ret.tree_leaf_count + ret.tree_internal_count; i++){

                vec3i cur_pos_min_cell_a = new_tree_acc_pos_min_cell[i];
                vec3i cur_pos_max_cell_a = new_tree_acc_pos_max_cell[i];

                for(u32 j = 0 ; j < tree_leaf_count + tree_internal_count; j++){

                    vec3i cur_pos_min_cell_b = old_tree_acc_pos_min_cell[j];
                    vec3i cur_pos_max_cell_b = old_tree_acc_pos_max_cell[j];


                    auto is_same_box = [&]() -> bool {
                        return
                            (cur_pos_min_cell_a.x() == cur_pos_min_cell_b.x()) &&
                            (cur_pos_min_cell_a.y() == cur_pos_min_cell_b.y()) &&
                            (cur_pos_min_cell_a.z() == cur_pos_min_cell_b.z()) &&
                            (cur_pos_max_cell_a.x() == cur_pos_max_cell_b.x()) &&
                            (cur_pos_max_cell_a.y() == cur_pos_max_cell_b.y()) &&
                            (cur_pos_max_cell_a.z() == cur_pos_max_cell_b.z()) ;
                    };

                    if(is_same_box()){

                        u32 store_val = j;

                        logger::raw_ln("i ->",cur_pos_min_cell_a,cur_pos_max_cell_a , "| ptr ->",cur_pos_min_cell_b,cur_pos_max_cell_b);


                        if(store_val >= tree_internal_count){
                            store_val -= tree_internal_count;
                        }

                        acc_new_node_id_to_old[i] = store_val;

                        break;
                    }


                }
            }
        }

        ret.print_tree_field(*new_node_id_to_old_naive);
        std::unique_ptr<sycl::buffer<u32>> new_node_id_to_old_v1 = std::make_unique<sycl::buffer<u32>>(ret.tree_leaf_count + ret.tree_internal_count);

        {
            auto & new_node_id_to_old = new_node_id_to_old_v1;

            //junk fill
            {
                sycl::host_accessor acc{* new_node_id_to_old, sycl::write_only, sycl::no_init};
                for (u32 i = 0 ; i < new_node_id_to_old->size(); i++) {
                    acc[i] = u32_max;
                }
            }


            sycl::host_accessor acc_new_node_id_to_old {*new_node_id_to_old,sycl::write_only, sycl::no_init};

            sycl::host_accessor new_tree_acc_pos_min_cell{*ret.buf_pos_min_cell,sycl::read_only};
            sycl::host_accessor new_tree_acc_pos_max_cell{*ret.buf_pos_max_cell,sycl::read_only};

            sycl::host_accessor old_tree_acc_pos_min_cell{*buf_pos_min_cell,sycl::read_only};
            sycl::host_accessor old_tree_acc_pos_max_cell{*buf_pos_max_cell,sycl::read_only};

            sycl::host_accessor old_tree_lchild_id   {*buf_lchild_id  ,sycl::read_only};
            sycl::host_accessor old_tree_rchild_id   {*buf_rchild_id  ,sycl::read_only};
            sycl::host_accessor old_tree_lchild_flag {*buf_lchild_flag,sycl::read_only};
            sycl::host_accessor old_tree_rchild_flag {*buf_rchild_flag,sycl::read_only};

            u32 old_tree_leaf_offset = tree_internal_count;


            for(u32 i = 0 ; i < ret.tree_leaf_count + ret.tree_internal_count; i++){

                //logger::raw_ln();

                vec3i cur_pos_min_cell_a = new_tree_acc_pos_min_cell[i];
                vec3i cur_pos_max_cell_a = new_tree_acc_pos_max_cell[i];

                u32 cur_id = 0;
                vec3i cur_pos_min_cell_b = old_tree_acc_pos_min_cell[cur_id];
                vec3i cur_pos_max_cell_b = old_tree_acc_pos_max_cell[cur_id];

                while(true){

                    //logger::raw_ln("i ->",cur_pos_min_cell_a,cur_pos_max_cell_a , "| ptr ->",cur_pos_min_cell_b,cur_pos_max_cell_b);

                    auto is_same_box = [&]() -> bool {
                        return
                            (cur_pos_min_cell_a.x() == cur_pos_min_cell_b.x()) &&
                            (cur_pos_min_cell_a.y() == cur_pos_min_cell_b.y()) &&
                            (cur_pos_min_cell_a.z() == cur_pos_min_cell_b.z()) &&
                            (cur_pos_max_cell_a.x() == cur_pos_max_cell_b.x()) &&
                            (cur_pos_max_cell_a.y() == cur_pos_max_cell_b.y()) &&
                            (cur_pos_max_cell_a.z() == cur_pos_max_cell_b.z()) ;
                    };

                    auto potential_cell = [&](vec3i other_min, vec3i other_max) -> bool {
                        return
                            (cur_pos_min_cell_a.x() >= other_min.x()) &&
                            (cur_pos_min_cell_a.y() >= other_min.y()) &&
                            (cur_pos_min_cell_a.z() >= other_min.z()) &&
                            (cur_pos_max_cell_a.x() <= other_max.x()) &&
                            (cur_pos_max_cell_a.y() <= other_max.y()) &&
                            (cur_pos_max_cell_a.z() <= other_max.z()) ;
                    };

                    if(is_same_box()){

                        //logger::raw_ln("id : ",i,"found ",cur_id);

                        u32 store_val = cur_id;

                        if(store_val >= tree_internal_count){
                            store_val -= tree_internal_count;
                        }

                        acc_new_node_id_to_old[i] = store_val;

                        break;
                    }


                    u32 lid = old_tree_lchild_id[cur_id] + old_tree_leaf_offset * old_tree_lchild_flag[cur_id];
                    u32 rid = old_tree_rchild_id[cur_id] + old_tree_leaf_offset * old_tree_rchild_flag[cur_id];

                    vec3i cur_pos_min_cell_bl = old_tree_acc_pos_min_cell[lid];
                    vec3i cur_pos_max_cell_bl = old_tree_acc_pos_max_cell[lid];

                    vec3i cur_pos_min_cell_br = old_tree_acc_pos_min_cell[rid];
                    vec3i cur_pos_max_cell_br = old_tree_acc_pos_max_cell[rid];

                    bool l_ok = potential_cell(cur_pos_min_cell_bl,cur_pos_max_cell_bl);
                    bool r_ok = potential_cell(cur_pos_min_cell_br,cur_pos_max_cell_br);

                    //logger::raw_ln("options l=",lid,cur_pos_min_cell_bl,cur_pos_max_cell_bl,l_ok);
                    //logger::raw_ln("options r=",rid,cur_pos_min_cell_br,cur_pos_max_cell_br,r_ok);

                    if(l_ok){

                        cur_pos_min_cell_b = cur_pos_min_cell_bl;
                        cur_pos_max_cell_b = cur_pos_max_cell_bl;

                        cur_id = lid;
                        //logger::raw_ln("id : ",i,"moving to ",cur_id);

                    }else if(r_ok){
                        cur_pos_min_cell_b = cur_pos_min_cell_br;
                        cur_pos_max_cell_b = cur_pos_max_cell_br;

                        cur_id = rid;
                        //logger::raw_ln("id : ",i,"moving to ",cur_id);

                    }else{
                        throw "";
                    }






                }

            }
        }

        ret.print_tree_field(*new_node_id_to_old_v1);

#endif

        std::unique_ptr<sycl::buffer<u32>> new_node_id_to_old_v2
            = std::make_unique<sycl::buffer<u32>>(
                ret.tree_reduced_morton_codes.tree_leaf_count
                + ret.tree_struct.internal_cell_count);

        {
            auto &new_node_id_to_old = new_node_id_to_old_v2;

            // junk fill
            {
                sycl::host_accessor acc{*new_node_id_to_old, sycl::write_only, sycl::no_init};
                for (u32 i = 0; i < new_node_id_to_old->size(); i++) {
                    acc[i] = u32_max;
                }
            }

            shamsys::instance::get_compute_queue().submit([&](sycl::handler &cgh) {
                sycl::accessor acc_new_node_id_to_old{
                    *new_node_id_to_old, cgh, sycl::write_only, sycl::no_init};

                sycl::accessor new_tree_acc_pos_min_cell{
                    *ret.tree_cell_ranges.buf_pos_min_cell, cgh, sycl::read_write};
                sycl::accessor new_tree_acc_pos_max_cell{
                    *ret.tree_cell_ranges.buf_pos_max_cell, cgh, sycl::read_write};

                sycl::accessor old_tree_acc_pos_min_cell{
                    *tree_cell_ranges.buf_pos_min_cell, cgh, sycl::read_only};
                sycl::accessor old_tree_acc_pos_max_cell{
                    *tree_cell_ranges.buf_pos_max_cell, cgh, sycl::read_only};

                sycl::accessor old_tree_lchild_id{*tree_struct.buf_lchild_id, cgh, sycl::read_only};
                sycl::accessor old_tree_rchild_id{*tree_struct.buf_rchild_id, cgh, sycl::read_only};
                sycl::accessor old_tree_lchild_flag{
                    *tree_struct.buf_lchild_flag, cgh, sycl::read_only};
                sycl::accessor old_tree_rchild_flag{
                    *tree_struct.buf_rchild_flag, cgh, sycl::read_only};

                u32 old_tree_leaf_offset = tree_struct.internal_cell_count;

                sycl::range<1> range_node = sycl::range<1>{
                    ret.tree_reduced_morton_codes.tree_leaf_count
                    + ret.tree_struct.internal_cell_count};

                // auto out = sycl::stream(128, 128, cgh);

                cgh.parallel_for(range_node, [=](sycl::item<1> item) {
                    // logger::raw_ln("\n \n ----------------\n \nnode : ",item.get_id(0));

                    ipos_t cur_pos_min_cell_a = new_tree_acc_pos_min_cell[item];
                    ipos_t cur_pos_max_cell_a = new_tree_acc_pos_max_cell[item];

                    u32 cur_id                = 0;
                    ipos_t cur_pos_min_cell_b = old_tree_acc_pos_min_cell[cur_id];
                    ipos_t cur_pos_max_cell_b = old_tree_acc_pos_max_cell[cur_id];

                    while (true) {

                        // logger::raw_ln("i ->",cur_pos_min_cell_a,cur_pos_max_cell_a , "| ptr
                        // ->",cur_pos_min_cell_b,cur_pos_max_cell_b);

                        auto is_same_box = [&]() -> bool {
                            return (cur_pos_min_cell_a.x() == cur_pos_min_cell_b.x())
                                   && (cur_pos_min_cell_a.y() == cur_pos_min_cell_b.y())
                                   && (cur_pos_min_cell_a.z() == cur_pos_min_cell_b.z())
                                   && (cur_pos_max_cell_a.x() == cur_pos_max_cell_b.x())
                                   && (cur_pos_max_cell_a.y() == cur_pos_max_cell_b.y())
                                   && (cur_pos_max_cell_a.z() == cur_pos_max_cell_b.z());
                        };

                        auto potential_cell = [&](ipos_t other_min, ipos_t other_max) -> bool {
                            return (cur_pos_min_cell_a.x() >= other_min.x())
                                   && (cur_pos_min_cell_a.y() >= other_min.y())
                                   && (cur_pos_min_cell_a.z() >= other_min.z())
                                   && (cur_pos_max_cell_a.x() <= other_max.x())
                                   && (cur_pos_max_cell_a.y() <= other_max.y())
                                   && (cur_pos_max_cell_a.z() <= other_max.z());
                        };

                        auto contain_cell = [&](ipos_t other_min, ipos_t other_max) -> bool {
                            return (cur_pos_min_cell_a.x() <= other_min.x())
                                   && (cur_pos_min_cell_a.y() <= other_min.y())
                                   && (cur_pos_min_cell_a.z() <= other_min.z())
                                   && (cur_pos_max_cell_a.x() >= other_max.x())
                                   && (cur_pos_max_cell_a.y() >= other_max.y())
                                   && (cur_pos_max_cell_a.z() >= other_max.z());
                        };

                        if (is_same_box()) {

                            // logger::raw_ln("found ",cur_id);

                            u32 store_val = cur_id;

                            // if(store_val >= old_tree_leaf_offset){
                            //     store_val -= old_tree_leaf_offset;
                            // }

                            acc_new_node_id_to_old[item] = store_val;

                            break;
                        }

                        u32 lid = old_tree_lchild_id[cur_id]
                                  + old_tree_leaf_offset * old_tree_lchild_flag[cur_id];
                        u32 rid = old_tree_rchild_id[cur_id]
                                  + old_tree_leaf_offset * old_tree_rchild_flag[cur_id];

                        ipos_t cur_pos_min_cell_bl = old_tree_acc_pos_min_cell[lid];
                        ipos_t cur_pos_max_cell_bl = old_tree_acc_pos_max_cell[lid];

                        ipos_t cur_pos_min_cell_br = old_tree_acc_pos_min_cell[rid];
                        ipos_t cur_pos_max_cell_br = old_tree_acc_pos_max_cell[rid];

                        bool l_ok = potential_cell(cur_pos_min_cell_bl, cur_pos_max_cell_bl);
                        bool r_ok = potential_cell(cur_pos_min_cell_br, cur_pos_max_cell_br);

                        // logger::raw_ln("options
                        // l=",lid,cur_pos_min_cell_bl,cur_pos_max_cell_bl,l_ok);
                        // logger::raw_ln("options
                        // r=",rid,cur_pos_min_cell_br,cur_pos_max_cell_br,r_ok);

                        if (l_ok) {

                            cur_pos_min_cell_b = cur_pos_min_cell_bl;
                            cur_pos_max_cell_b = cur_pos_max_cell_bl;

                            cur_id = lid;
                            // logger::raw_ln("moving to ",cur_id);

                        } else if (r_ok) {
                            cur_pos_min_cell_b = cur_pos_min_cell_br;
                            cur_pos_max_cell_b = cur_pos_max_cell_br;

                            cur_id = rid;
                            // logger::raw_ln("moving to ",cur_id);

                        } else {

                            // if nothing is neither the same or a super set of our cell it means
                            // that
                            //  our cell is a superset of one of the child hence the following check
                            bool l_contain = contain_cell(cur_pos_min_cell_bl, cur_pos_max_cell_bl);
                            bool r_contain = contain_cell(cur_pos_min_cell_br, cur_pos_max_cell_br);

                            // logger::raw_ln("options
                            // l=",lid,cur_pos_min_cell_bl,cur_pos_max_cell_bl,l_contain);
                            // logger::raw_ln("options
                            // r=",rid,cur_pos_min_cell_br,cur_pos_max_cell_br,r_contain);

                            if (l_contain) {
                                // logger::raw_ln("found ",cur_id);

                                u32 store_val                = cur_id;
                                acc_new_node_id_to_old[item] = store_val;

                                // TODO : check that no particules are outside of the bound when
                                // restricted by this line
                                new_tree_acc_pos_min_cell[item] = cur_pos_min_cell_bl;
                                new_tree_acc_pos_max_cell[item] = cur_pos_max_cell_bl;

                                break;
                            } else if (r_contain) {
                                // logger::raw_ln("found ",cur_id);

                                u32 store_val                = cur_id;
                                acc_new_node_id_to_old[item] = store_val;

                                // TODO : check that no particules are outside of the bound when
                                // restricted by this line
                                new_tree_acc_pos_min_cell[item] = cur_pos_min_cell_br;
                                new_tree_acc_pos_max_cell[item] = cur_pos_max_cell_br;

                                break;
                            } else {
                                // out << "[CRASH] Tree cut had a weird behavior during old cell
                                // search : \n"; throw "";

                                u32 store_val = cur_id;

                                // if(store_val >= old_tree_leaf_offset){
                                //     store_val -= old_tree_leaf_offset;
                                // }

                                acc_new_node_id_to_old[item] = store_val;

                                break;
                            }
                        }
                    }
                });
            });
        }

        // because we have updated the cell ranges in the tree cut
        shamrock::sfc::MortonKernels<u_morton, vec3, dim>::sycl_irange_to_range(
            queue,
            ret.tree_reduced_morton_codes.tree_leaf_count + ret.tree_struct.internal_cell_count,
            std::get<0>(ret.bounding_box),
            std::get<1>(ret.bounding_box),
            ret.tree_cell_ranges.buf_pos_min_cell,
            ret.tree_cell_ranges.buf_pos_max_cell,
            ret.tree_cell_ranges.buf_pos_min_cell_flt,
            ret.tree_cell_ranges.buf_pos_max_cell_flt);

        // ret.print_tree_field(*new_node_id_to_old_v2);

        shamlog_debug_ln(
            "TreeCutter",
            "tree cut cells:",
            tree_struct.internal_cell_count,
            "->",
            ret.tree_struct.internal_cell_count,
            "obj:",
            tree_morton_codes.obj_cnt,
            "->",
            extract_id.size());

        return CuttedTree{
            std::move(ret),
            std::move(new_node_id_to_old_v2),
            std::make_unique<sycl::buffer<u32>>(shamalgs::memory::vector_to_buf(
                shamsys::instance::get_compute_queue(), std::move(extract_id)))};
    }
}

template class RadixTree<u32, f32_3>;
template class RadixTree<u64, f32_3>;

template class RadixTree<u32, f64_3>;
template class RadixTree<u64, f64_3>;

template class RadixTree<u32, u32_3>;
template class RadixTree<u64, u32_3>;

template class RadixTree<u32, u64_3>;
template class RadixTree<u64, u64_3>;

template class RadixTree<u32, i64_3>;
template class RadixTree<u64, i64_3>;
