// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright (c) 2021-2026 Timothée David--Cléris <tim.shamrock@proton.me>
// SPDX-License-Identifier: CeCILL Free Software License Agreement v2.1
// Shamrock is licensed under the CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

#pragma once

/**
 * @file radixSortOnesweep.hpp
 * @author Timothée David--Cléris (tim.shamrock@proton.me)
 * @brief
 *
 */

#include "shambase/integer.hpp"
#include "shambase/type_traits.hpp"
#include "DigitBinner.hpp"
#include "shamalgs/details/numeric/scanDecoupledLookback.hpp"
#include "shamalgs/memory.hpp"
#include "shamalgs/numeric.hpp"
#include "shambackends/sycl.hpp"
#include "shamsys/legacy/log.hpp"
#include <numeric>

namespace shamalgs::algorithm::details {

    /*
    tile histogram :

    element a :

    a = 2 + 1x4^3

    |digit | digit places  |
    |      | 0 | 1 | 2 | 3 |
    ------------------------
    |  0   | 0 | 1 | 1 | 0 |
    |  1   | 0 | 0 | 0 | 1 |
    |  2   | 1 | 0 | 0 | 0 |
    |  3   | 0 | 0 | 0 | 0 |

    sum array on the table

    */

    template<class Tkey, class Tval, u32 group_size, u32 digit_len>
    class SortByKeyRadixOnesweep;

    template<class Tkey, class Tval, u32 group_size, u32 digit_len>
    void sort_by_key_radix_onesweep(
        sycl::queue &q, sycl::buffer<Tkey> &buf_key, sycl::buffer<Tval> &buf_values, u32 len) {

        sycl::buffer<Tkey> tmp_buf_key(len);
        sycl::buffer<Tval> tmp_buf_values(len);

        auto get_in_keys = [&](u32 step) -> sycl::buffer<Tkey> & {
            if (step % 2 == 0) {
                return buf_key;
            } else {
                return tmp_buf_key;
            }
        };

        auto get_out_keys = [&](u32 step) -> sycl::buffer<Tkey> & {
            if (step % 2 == 0) {
                return tmp_buf_key;
            } else {
                return buf_key;
            }
        };

        auto get_in_vals = [&](u32 step) -> sycl::buffer<Tval> & {
            if (step % 2 == 0) {
                return buf_values;
            } else {
                return tmp_buf_values;
            }
        };

        auto get_out_vals = [&](u32 step) -> sycl::buffer<Tval> & {
            if (step % 2 == 0) {
                return tmp_buf_values;
            } else {
                return buf_values;
            }
        };

        u32 group_cnt = shambase::group_count(len, group_size);

        // group_cnt = group_cnt + (group_cnt % 4);
        u32 corrected_len = group_cnt * group_size;

        // memory::print_buf(buf_key, len, 16, "{:4} ");

        using Binner = DigitBinner<Tkey, digit_len>;

        sycl::buffer<u32> digit_histogram
            = Binner::template make_digit_histogram<group_size>(q, buf_key, len);

        // logger::raw_ln("digit histogram");
        // memory::print_buf(digit_histogram, Binner::value_count, Binner::digit_count, "{:4} ");

        {

            sycl::host_accessor acc{digit_histogram, sycl::read_write};

            auto ptr = &(acc[0]);

            for (u32 digit_place = 0; digit_place < Binner::digit_bit_places; digit_place++) {
                u32 offset_ptr = Binner::digit_count * digit_place;
                std::exclusive_scan(
                    ptr + offset_ptr, ptr + offset_ptr + Binner::digit_count, ptr + offset_ptr, 0);
            }
        }

        // logger::raw_ln("digit histogram");
        // memory::print_buf(digit_histogram, Binner::value_count, Binner::digit_count, "{:4} ");

        using namespace shamalgs::numeric::details;

        using DecoupledLookBack
            = ScanDecoupledLoockBack<u32, group_size, Standard, ScanTile30bitint>;

        u32 step = 0;
        for (Tkey cur_digit_place = 0; cur_digit_place < shambase::bitsizeof<Tkey>;
             cur_digit_place += digit_len) {

            DecoupledLookBack dlookbackscan(q, group_cnt, Binner::digit_count);

            atomic::DynamicIdGenerator<i32, group_size> id_gen(q);

            q.submit([&, len, cur_digit_place, step](sycl::handler &cgh) {
                sycl::accessor keys{get_in_keys(step), cgh, sycl::read_only};
                sycl::accessor vals{get_in_vals(step), cgh, sycl::read_only};

                sycl::accessor new_keys{get_out_keys(step), cgh, sycl::write_only, sycl::no_init};
                sycl::accessor new_vals{get_out_vals(step), cgh, sycl::write_only, sycl::no_init};

                sycl::accessor value_write_offsets{digit_histogram, cgh, sycl::read_only};

                sycl::local_accessor<u32, 1> local_digit_counts{Binner::digit_count, cgh};
                sycl::local_accessor<u32, 1> scanned_digit_counts{Binner::digit_count, cgh};

                // sycl::stream dump (4096,1024,cgh);
                auto dyn_id = id_gen.get_access(cgh);

                auto scanop = dlookbackscan.get_access(cgh);

                using at_ref_loc_count = sycl::atomic_ref<
                    u32,
                    sycl::memory_order_relaxed,
                    sycl::memory_scope_work_group,
                    sycl::access::address_space::local_space>;

                u32 histogram_ptr_offset = step * Binner::digit_count;

                cgh.parallel_for<SortByKeyRadixOnesweep<Tkey, Tval, group_size, digit_len>>(
                    sycl::nd_range<1>{corrected_len, group_size}, [=](sycl::nd_item<1> id) {
                        atomic::DynamicId<i32> group_id = dyn_id.compute_id(id);

                        u32 local_id      = id.get_local_id(0);
                        u32 group_tile_id = group_id.dyn_group_id;
                        u32 global_id     = group_id.dyn_global_id;

                        // u32 group_tile_id = id.get_group_linear_id();
                        // u32 global_id = group_tile_id * group_size + local_id;

                        if (local_id == 0) {
                            for (u32 digit_ptr = 0; digit_ptr < Binner::digit_count; digit_ptr++) {
                                local_digit_counts[digit_ptr] = 0;
                            }
                        }
                        id.barrier(sycl::access::fence_space::local_space);

                        bool is_valid_key = (global_id < len);

                        Tkey cur_key = (is_valid_key) ? keys[global_id] : 0;

                        Tkey digit_value = Binner::get_digit_value(cur_key, step);

                        // if(group_tile_id == 0){
                        //     dump << local_digit_counts[0] << " " << local_digit_counts[1] <<
                        //     "\n";
                        // }

                        u32 curr_loc_offset = at_ref_loc_count(local_digit_counts[digit_value])
                                                  .fetch_add((is_valid_key) ? 1U : 0);

                        // if(group_tile_id == 0){
                        //     dump << cur_key << " " <<digit_value << " " << curr_loc_offset  <<
                        //     "\n";
                        // }
                        //
                        //
                        id.barrier(sycl::access::fence_space::local_space);
                        // if(group_tile_id == 0){
                        //     dump << local_digit_counts[0] << " " << local_digit_counts[1] <<
                        //     "\n";
                        // }

                        // generate scanned tile value for each digits
                        for (u32 digit_ptr = 0; digit_ptr < Binner::digit_count; digit_ptr++) {

                            scanop.decoupled_lookback_scan(
                                id,
                                local_id,
                                group_tile_id,
                                [=]() {
                                    return local_digit_counts[digit_ptr];
                                },
                                [=](u32 accum) {
                                    scanned_digit_counts[digit_ptr] = accum;
                                },
                                digit_ptr);
                        }

                        // if(local_id == 0){
                        //     dump << "-- gid" << global_id << "\n";
                        //     for(u32 digit_ptr = 0; digit_ptr < Binner::digit_count; digit_ptr
                        //     ++){
                        //         dump << local_digit_counts[digit_ptr] << " "
                        //         <<scanned_digit_counts[digit_ptr] << "\n";
                        //     }
                        // }

                        // load from global buffer
                        if (global_id < len) {

                            // logger::raw_ln(cur_key,digit_value,curr_loc_offset, step);

                            u32 value_write_offset_global
                                = value_write_offsets[(digit_value) + histogram_ptr_offset];

                            u32 write_offset = curr_loc_offset + scanned_digit_counts[digit_value]
                                               + value_write_offset_global;

                            // if(local_id == 0){
                            //     dump << "-- gid" << global_id << "\n";
                            //     dump << "k="<<cur_key << "\n";
                            //     dump << "d="<<digit_value << "\n";
                            //     dump << "delta="<<curr_loc_offset << "\n";
                            //     dump << "gdelta="<<value_write_offset_global << "\n";
                            //     dump << "sdelta="<<scanned_digit_counts[digit_value] << "\n";
                            //     dump << "wdelta="<<write_offset << "\n";
                            // }

                            new_keys[write_offset]
                                = keys[global_id]; // can be loaded initially and stored only here
                                                   // rather than reload
                            new_vals[write_offset] = vals[global_id];
                        }
                    });
            });

            // q.wait();

            // logger::raw_ln("digit histogram place : ", cur_digit_place);
            // memory::print_buf(get_out_keys(step), len, 16, "{:4} ");

            // return;

            step++;
        }
    }

} // namespace shamalgs::algorithm::details
