// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright(C) 2021-2023 Timothée David--Cléris <timothee.david--cleris@ens-lyon.fr>
// Licensed under CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

/**
 * @file streamCompactExclScan.cpp
 * @author Timothée David--Cléris (timothee.david--cleris@ens-lyon.fr)
 * @brief
 *
 */

#include "shamalgs/details/numeric/streamCompactExclScan.hpp"
#include "shambase/integer.hpp"
#include "shamalgs/details/numeric/numericFallback.hpp"
#include "shamalgs/memory.hpp"
#include "shamalgs/numeric.hpp"

class StreamCompactionAlg;

namespace shamalgs::numeric::details {

    std::tuple<std::optional<sycl::buffer<u32>>, u32>
    stream_compact_excl_scan(sycl::queue &q, sycl::buffer<u32> &buf_flags, u32 len) {

        if (len < 2) {
            return stream_compact_fallback(q, buf_flags, len);
        }

        // perform the exclusive sum of the buf flag
        sycl::buffer<u32> excl_sum = exclusive_sum(q, buf_flags, len);

        // recover the end value of the sum to know the new size
        u32 new_len = memory::extract_element(q, excl_sum, len - 1);

        u32 end_flag = memory::extract_element(q, buf_flags, len - 1);

        if (end_flag) {
            new_len++;
        }

        logger::debug_sycl_ln("StreamCompact", "number of element : ", new_len);

        if (new_len == 0) {
            return {{}, 0};
        }

        constexpr u32 group_size = 256;
        u32 max_len              = len;
        u32 group_cnt            = shambase::group_count(len, group_size);
        group_cnt                = group_cnt + (group_cnt % 4);
        u32 corrected_len        = group_cnt * group_size;

        // create the index buffer that we will return
        sycl::buffer<u32> index_map{new_len};

        q.submit([&, max_len](sycl::handler &cgh) {
            sycl::accessor sum_vals{excl_sum, cgh, sycl::read_only};
            sycl::accessor new_idx{index_map, cgh, sycl::write_only, sycl::no_init};

            u32 last_idx  = len - 1;
            u32 last_flag = end_flag;

            cgh.parallel_for<StreamCompactionAlg>(

                sycl::nd_range<1>{corrected_len, group_size}, [=](sycl::nd_item<1> id) {
                    u32 local_id      = id.get_local_id(0);
                    u32 group_tile_id = id.get_group_linear_id();
                    u32 idx           = group_tile_id * group_size + local_id;

                    if (idx >= max_len)
                        return;

                    u32 current_val = sum_vals[idx];

                    bool _if1 = (idx < last_idx);
                    bool should_write
                        = (_if1 && (current_val < sum_vals[idx + 1])) || (bool(last_flag) && !_if1);

                    if (should_write) {
                        new_idx[current_val] = idx;
                    }
                });
        });

        return {std::move(index_map), new_len};
    };

} // namespace shamalgs::numeric::details
