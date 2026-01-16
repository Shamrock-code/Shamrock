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
 * @file DigitBinner.hpp
 * @author Timothée David--Cléris (tim.shamrock@proton.me)
 * @brief
 *
 */

#include "shambase/integer.hpp"
#include "shambase/type_traits.hpp"
#include "shamalgs/memory.hpp"
#include "shamalgs/numeric.hpp"
#include "shambackends/sycl.hpp"
#include "shamsys/legacy/log.hpp"
#include <numeric>

namespace shamalgs::algorithm::details {

    template<class T, u32 digit_bit_len>
    class DigitBinner {
        public:
        static constexpr T bitlen_T         = shambase::bitsizeof<T>;
        static constexpr T digit_bit_places = bitlen_T / digit_bit_len;
        static constexpr T digit_count      = (1U << digit_bit_len);
        static constexpr T value_count      = digit_bit_places * digit_count;
        static constexpr T digit_mask       = digit_count - 1;

        static_assert(
            digit_bit_places * digit_bit_len == bitlen_T, "the conversion should be correct");

        template<class Acc>
        inline static void fetch_add_bin(Acc accessor, T digit_val, T digit_place) {
            using atomic_ref_T = sycl::atomic_ref<
                u32,
                sycl::memory_order_relaxed,
                sycl::memory_scope_work_group,
                sycl::access::address_space::local_space>;

            atomic_ref_T(accessor[digit_val + digit_place * digit_count]).fetch_add(1U);
        }

        inline static T get_digit_value(T value, T digit_place) {
            return digit_mask & (value >> (digit_place * digit_bit_len));
        }

        template<class Acc>
        inline static void add_bin_key(Acc accessor, T value_to_bin) {

#pragma unroll
            for (T digit_place = 0; digit_place < digit_bit_places; digit_place++) {
                T shifted = get_digit_value(value_to_bin, digit_place);

                fetch_add_bin(accessor, shifted, digit_place);
            }
        }

        template<u32 group_size, class Tkey>
        inline static sycl::buffer<u32> make_digit_histogram(
            sycl::queue &q, sycl::buffer<Tkey> &buf_key, u32 len) {

            u32 group_cnt = shambase::group_count(len, group_size);

            group_cnt         = group_cnt + (group_cnt % 4);
            u32 corrected_len = group_cnt * group_size;

            sycl::buffer<u32> digit_histogram(value_count);

            shamalgs::memory::buf_fill_discard(q, digit_histogram, 0U);

            // logger::raw_ln("digit binning");
            // memory::print_buf(digit_histogram, value_count, digit_count, "{:4} ");

            q.submit([&, len](sycl::handler &cgh) {
                sycl::accessor keys{buf_key, cgh, sycl::read_only};
                sycl::accessor histogram{digit_histogram, cgh, sycl::read_write};

                sycl::local_accessor<u32, 1> local_histogram{value_count, cgh};

                cgh.parallel_for(
                    sycl::nd_range<1>{corrected_len, group_size}, [=](sycl::nd_item<1> id) {
                        u32 local_id      = id.get_local_id(0);
                        u32 group_tile_id = id.get_group_linear_id();
                        u32 global_id     = group_tile_id * group_size + local_id;

                        if (local_id == 0) {
                            for (u32 idx = 0; idx < value_count; idx++) {
                                local_histogram[idx] = 0;
                            }
                        }
                        id.barrier(sycl::access::fence_space::local_space);

                        // load from global buffer
                        if (global_id < len) {
                            add_bin_key(local_histogram, keys[global_id]);
                        }

                        id.barrier(sycl::access::fence_space::local_space);

                        for (u32 i = local_id; i < value_count; i += group_size) {
                            u32 dcount = local_histogram[i];

                            if (dcount != 0) {

                                using atomic_ref_t = sycl::atomic_ref<
                                    u32,
                                    sycl::memory_order_relaxed,
                                    sycl::memory_scope_device,
                                    sycl::access::address_space::global_space>;

                                atomic_ref_t(histogram[i]).fetch_add(dcount);
                            }
                        }
                    });
            });

            return digit_histogram;
        }
    };

} // namespace shamalgs::algorithm::details
