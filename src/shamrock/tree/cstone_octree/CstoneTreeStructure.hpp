// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright(C) 2021-2023 Timothée David--Cléris <timothee.david--cleris@ens-lyon.fr>
// Licensed under CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

#pragma once

/**
 * @file CstoneTreeStructure.hpp
 * @author Leodasce sewanou (leodasce.sewanou@ens-lyon.fr)
 * @brief
 */

#include "shambase/aliases_int.hpp"
#include "shambase/exception.hpp"
#include "shambase/integer.hpp"
#include "shamalgs/details/numeric/numeric.hpp"
#include "shamalgs/memory.hpp"
#include "shamalgs/numeric.hpp"
#include "shamalgs/reduction.hpp"
#include "shamalgs/serialize.hpp"
#include "shammath/sfc/morton.hpp"
#include "shamsys/NodeInstance.hpp"
#include <shambackends/sycl.hpp>
#include <cmath>
#include <cstdlib>
#include <memory>

namespace shamrock::cs_tree {

    /**
     * @brief the pos-th octal digit from an morton code starting from the most significant bit
     * @param key the 32-bits morton codes
     * @param pos the digit place to extract
     */
    constexpr u16 octalDigit(u32 key, u16 pos) {
        u16 significatif_bits = shamrock::sfc::MortonCodes<u32, 3>::significant_bits;
        return (key >> (significatif_bits - ((u16) 3) * pos)) & (u16) 7;
    }

    /**
     * @brief the pos-th octal digit from an morton code starting from the most significant bit
     * @param key the 64-bits morton codes
     * @param pos the digit place to extract
     */
    constexpr u32 octalDigit(u64 key, u32 pos) {
        u32 significatif_bits = shamrock::sfc::MortonCodes<u64, 3>::significant_bits;
        return (key >> (significatif_bits - ((u32) 3) * pos)) & (u32) 7;
    }

    /**
     * @brief binary to octree octal digit weight from Eq.10 in Keller et al. 2023
     * Cornerstone: Octree Construction Algorithms for Scalable Particle Simulations
     */
    constexpr i32 digitWeight(i32 digit) {
        i32 _mask = -i32(digit >= 4);
        return ((7 - digit) & _mask) - (digit & ~_mask);
    }

    void group_local_radix_sort(
        sycl::queue &q,
        sycl::buffer<u32> &in_buf,
        sycl::buffer<u32> &out_buf,
        sycl::buffer<u32> &prefix_sums_buf,
        sycl::buffer<u32> &block_sums_buf,
        u32 buf_len,
        u32 group_size,
        u32 bit_shift_wide) {
        q.submit([&](sycl::handler &cgh) {
            auto out_acc         = out_buf.get_access<sycl::access_mode::read_write>(cgh);
            auto in_acc          = in_buf.get_access<sycl::access_mode::read_write>(cgh);
            auto prefix_sums_acc = prefix_sums_buf.get_access<sycl::access_mode::read_write>(cgh);
            auto block_sums_acc  = block_sums_buf.get_access<sycl::access_mode::read_write>(cgh);

            sycl::local_accessor<u32> shared_acc_0{group_size, cgh};
            sycl::local_accessor<u32> shared_acc_1{group_size + 1, cgh};
            sycl::local_accessor<u32> shared_acc_2{group_size, cgh};
            sycl::local_accessor<u32> shared_acc_3{4, cgh};
            sycl::local_accessor<u32> shared_acc_4{4, cgh};
            cgh.parallel_for(sycl::nd_range<1>{buf_len, group_size}, [=](sycl::nd_item<1> it) {
                // copy global data into work-group memory
                auto loc_id    = it.get_local_id(0);
                auto global_id = it.get_global_id(0);
                if (global_id < buf_len) {
                    shared_acc_0[loc_id] = in_acc[global_id];
                } else {
                    shared_acc_0[loc_id] = 0;
                }

                it.barrier(); // wait all threads in block to finish copy into work-group memory

                // extract the current 2-bit from LSB
                u32 curr_key      = shared_acc_0[loc_id];
                u32 _2bit_extract = (curr_key >> bit_shift_wide) & 3;

                for (i32 i = 0; i < 4; i++) {
                    // 0-mask :  generating a bit-mask to each of the four possible digit get from
                    // the extract 2-bits
                    shared_acc_1[loc_id] = 0;
                    if (loc_id == 0)
                        shared_acc_1[group_size]
                            = 0; // this is to be sure that since there are only group_size
                                 // work-item/work-group the last entry is also initialized to 0.

                    it.barrier(); // // wait all threads in block to finish initialization

                    // Bit-mask
                    bool val_i = false;
                    if (global_id < buf_len) {
                        val_i                = (_2bit_extract == i);
                        shared_acc_1[loc_id] = val_i;
                    }

                    it.barrier();

                    // local mask scan using Hillis-Steele parallel algo
                    i32 partner   = 0;
                    u32 sum       = 0;
                    u32 max_steps = (u32) log2(group_size);
                    for (i32 j = 0; j < max_steps; j++) {
                        partner = (i32) loc_id - (1 << j);

                        if (partner >= 0) {
                            sum = shared_acc_1[loc_id] + shared_acc_1[partner];
                        } else {
                            sum = shared_acc_1[loc_id];
                        }
                        it.barrier();
                        shared_acc_1[loc_id] = sum;
                        it.barrier();
                    }
                    // inclusive scan to exclusive scan
                    u32 tmp_val = 0;
                    tmp_val     = shared_acc_1[loc_id];
                    it.barrier();
                    shared_acc_1[loc_id + 1] = tmp_val;
                    it.barrier();

                    if (loc_id == 0) {
                        shared_acc_1[0] = 0;
                        u32 total_sum   = shared_acc_1[group_size];
                        shared_acc_3[i] = total_sum;
                        block_sums_acc[i * it.get_group_range(0) + it.get_group(0)] = total_sum;
                    }

                    it.barrier();

                    if (val_i && (global_id < buf_len))
                        shared_acc_2[loc_id] = shared_acc_1[loc_id];
                    it.barrier();
                }

                // block scan
                if (loc_id == 0) {
                    u32 r_sum = 0;
                    for (i32 i = 0; i < 4; i++) {
                        shared_acc_4[i] = r_sum;
                        r_sum += shared_acc_3[i];
                    }
                }

                it.barrier();

                if (global_id < buf_len) {
                    // new index
                    u32 exc_scan = shared_acc_2[loc_id];
                    u32 new_pos  = exc_scan + shared_acc_4[_2bit_extract];
                    it.barrier();

                    // Local shuffle for coalescing memory transfert
                    shared_acc_0[new_pos] = curr_key;
                    shared_acc_2[new_pos] = exc_scan;

                    it.barrier();

                    // copy block to global memory
                    prefix_sums_acc[global_id] = shared_acc_2[loc_id];
                    out_acc[global_id]         = shared_acc_0[loc_id];
                }

                it.barrier();
            });
        });
    }

    void global_shuffle(
        sycl::queue &q,
        sycl::buffer<u32> &in_buf,
        sycl::buffer<u32> &out_buf,
        sycl::buffer<u32> &exc_scan,
        sycl::buffer<u32> &prefix_sums_buf,
        u32 buf_len,
        u32 group_size,
        u32 bit_shift_wide) {
        q.submit([&](sycl::handler &cgh) {
            auto out_acc         = out_buf.get_access<sycl::access_mode::read_write>(cgh);
            auto in_acc          = in_buf.get_access<sycl::access_mode::read_write>(cgh);
            auto prefix_sums_acc = prefix_sums_buf.get_access<sycl::access_mode::read_write>(cgh);
            auto exc_scan_acc    = exc_scan.get_access<sycl::access_mode::read_write>(cgh);

            cgh.parallel_for(sycl::nd_range<1>{buf_len, group_size}, [=](sycl::nd_item<1> it) {
                u32 loc_id  = (u32) it.get_local_id(0);
                u32 glob_id = (u32) it.get_global_id(0);

                if (glob_id < buf_len) {
                    auto cur_key       = in_acc[glob_id];
                    u32 _2bit_extract  = (cur_key >> bit_shift_wide) & 3;
                    u32 cur_prefix_sum = prefix_sums_acc[glob_id];
                    u32 global_pos
                        = exc_scan_acc[_2bit_extract * it.get_group_range(0) + it.get_group(0)]
                          + cur_prefix_sum;
                    it.barrier();
                    out_acc[global_pos] = cur_key;
                }
            });
        });
    }

    // TODO template to treat both 32-bits and 64-bits SFC keys
    void radixsortFourWay(
        sycl::queue &q, sycl::buffer<u32> &in_buffer, sycl::buffer<u32> &out_buffer, u32 buf_len) {
        constexpr u32 group_size = 128;
        u32 group_cnt            = shambase::group_count(buf_len, group_size);
        // group_cnt =  group_cnt + (group_cnt % 4);

        std::unique_ptr<sycl::buffer<u32>> prefix_sums_buf
            = std::make_unique<sycl::buffer<u32>>(buf_len);
        q.submit([&](sycl::handler &cgh) {
            auto prefix_sums_acc = prefix_sums_buf->get_access<sycl::access_mode::write>(cgh);
            cgh.parallel_for(buf_len, [=](sycl::item<1> it) {
                u32 i              = (u32) it.get_id(0);
                prefix_sums_acc[i] = 0;
            });
        });
        u32 block_sum_len = group_cnt * 4;
        std::unique_ptr<sycl::buffer<u32>> block_sums_buf
            = std::make_unique<sycl::buffer<u32>>(block_sum_len);
        std::unique_ptr<sycl::buffer<u32>> scan_block_sums_buf
            = std::make_unique<sycl::buffer<u32>>(block_sum_len);
        q.submit([&](sycl::handler &cgh) {
            auto block_sums_acc = block_sums_buf->get_access<sycl::access_mode::write>(cgh);
            auto scan_block_sums_acc
                = scan_block_sums_buf->get_access<sycl::access_mode::write>(cgh);
            cgh.parallel_for(block_sum_len, [=](sycl::item<1> it) {
                u32 i                  = (u32) it.get_id(0);
                block_sums_acc[i]      = 0;
                scan_block_sums_acc[i] = 0;
            });
        });

        // auto local_acc_nbElts = (group_size *3 + 1 )  + (2 * 4);

        for (i32 bit_shift_wide = 0; bit_shift_wide <= 30; bit_shift_wide += 2) {
            group_local_radix_sort(
                q,
                in_buffer,
                out_buffer,
                *prefix_sums_buf,
                *block_sums_buf,
                buf_len,
                group_size,
                bit_shift_wide);
            sycl::buffer<u32> excl_sum
                = shamalgs::numeric::exclusive_sum(q, *block_sums_buf, block_sum_len);
            global_shuffle(
                q,
                in_buffer,
                out_buffer,
                excl_sum,
                *prefix_sums_buf,
                buf_len,
                group_size,
                bit_shift_wide);
        }
    }

    template<class u_morton>
    class CstoneTreeStructure {
        public:
        u32 internal_cell_count;
        u32 all_cell_count;
        u32 leaves_cell_count;
        u32 max_octree_depth;

        std::unique_ptr<sycl::buffer<u_morton>>
            buf_all_nodes_morton_buf;                        // size =  internal nodes + leves nodes
        std::unique_ptr<sycl::buffer<u32>> buf_connectivity; // size = internal nodes + leaves nodes
        std::unique_ptr<sycl::buffer<u32>> buf_tree_levels_offsets; // size = max octree depth + 2

        bool is_built() {
            return bool(buf_all_nodes_morton_buf) && bool(buf_connectivity)
                   && bool(buf_tree_levels_offsets);
        }

        inline void build(
            sycl::queue &queue,
            u32 _internal_cell_count,
            u32 _all_cell_count,
            u32 _leaves_cell_count,
            u32 _max_octree_depth,
            sycl::buffer<u_morton> &morton_buf) {
            if (!(_internal_cell_count < morton_buf.size())) {
                throw shambase::make_except_with_loc<std::runtime_error>(
                    "morton buf must be at least with size() greater than internal_cell_count");
            }

            internal_cell_count = _internal_cell_count;
            all_cell_count      = _all_cell_count;
            leaves_cell_count   = _leaves_cell_count;
            max_octree_depth    = _max_octree_depth;

            buf_all_nodes_morton_buf = std::make_unique<sycl::buffer<u_morton>>(all_cell_count);
            buf_connectivity         = std::make_unique<sycl::buffer<u32>>(all_cell_count);
            buf_tree_levels_offsets  = std::make_unique<sycl::buffer<u32>>(max_octree_depth + 2);

            // copy morton code from morton_buf into
            // buf_all_nodes_morton_buf[internal_cell_count:end]
            queue.submit([&](sycl::handler &cgh) {
                auto all_nodes_morton_buff
                    = buf_all_nodes_morton_buf->get_access<sycl::access::mode::discard_write>(cgh);
                auto morton_codes = morton_buf.template get_access<sycl::access::mode::read>(cgh);

                cgh.parallel_for<"leaves_morton_code_copy">(
                    leaves_cell_count, [=](sycl::item<1> it) {
                        i32 i                                          = (i32) it.get_id(0);
                        all_nodes_morton_buff[i + internal_cell_count] = morton_codes[i];
                    });
            });

            // TODO : need a barrier here ?

            //  Generate internal nodes
            queue.submit([&](sycl::handler &cgh) {
                auto all_nodes_morton_buff
                    = buf_all_nodes_morton_buf->get_access<sycl::access::mode::discard_write>(cgh);
                auto morton_codes = morton_buf.template get_access<sycl::access::mode::read>(cgh);

                auto map_leafIdx_to_internalIdx = [=](u_morton m_key, i32 oct_level) {
                    auto res = 0;
                    for (u32 i = 0; i < oct_level + 2; i++) {
                        res += digitWeight(octalDigit(m_key, oct_level));
                    }

                    return res;
                };

                cgh.parallel_for<"internal_nodes">(internal_cell_count, [=](sycl::item<1> it) {
                    int i      = (i32) it.get_id(0);
                    auto DELTA = [=](i32 x, i32 y) {
                        return sham::karras_delta(x, y, leaves_cell_count, morton_codes);
                    };

                    int node_lenght     = DELTA(i, i + 1);
                    bool is_octree_node = ((node_lenght % 3) == 0);
                    if (is_octree_node) {
                        i32 j = (i - map_leafIdx_to_internalIdx(morton_codes[i], node_lenght)) / 7;
                        all_nodes_morton_buff[j] = morton_codes[i];
                    }
                });
            });

            // Radix sort or bitonic sort
        }
    };

} // namespace shamrock::cs_tree
