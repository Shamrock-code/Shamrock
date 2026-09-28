// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright (c) 2021-2026 Timothée David--Cléris <tim.shamrock@proton.me>
// SPDX-License-Identifier: CeCILL Free Software License Agreement v2.1
// Shamrock is licensed under the CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

#include "shamalgs/primitives/digit_histogram.hpp"
#include "shamalgs/primitives/mock_vector.hpp"
#include "shambackends/DeviceBuffer.hpp"
#include "shamsys/NodeInstance.hpp"
#include "shamtest/shamtest.hpp"
#include <string>
#include <vector>

namespace {

    template<class Tkey, u32 radix_bits>
    std::vector<u32> reference_histogram(const std::vector<Tkey> &keys, u32 len) {
        constexpr u32 nbuckets = 1u << radix_bits;
        constexpr u32 npasses  = (sizeof(Tkey) * 8) / radix_bits;
        std::vector<u32> hist(npasses * nbuckets, 0);
        for (u32 i = 0; i < len; i++) {
            for (u32 p = 0; p < npasses; p++) {
                hist[p * nbuckets + (u32(keys[i] >> (p * radix_bits)) & (nbuckets - 1))]++;
            }
        }
        return hist;
    }

    template<class Tkey, u32 radix_bits>
    void check_case(u32 buf_size, u32 len, u64 seed, const std::string &name) {
        auto sched = shamsys::instance::get_compute_scheduler_ptr();

        std::vector<Tkey> keys = shamalgs::primitives::mock_vector<Tkey>(seed, buf_size);

        sham::DeviceBuffer<Tkey> buf_key(buf_size, sched);
        if (buf_size > 0) {
            buf_key.copy_from_stdvec(keys);
        }

        // start from a wrongly sized, non zero buffer : it must be resized and zeroed
        sham::DeviceBuffer<u32> buf_hist(3, sched);
        buf_hist.fill(7);

        if (len == buf_size) {
            shamalgs::primitives::digit_histogram<Tkey, radix_bits>(sched, buf_key, buf_hist);
        } else {
            shamalgs::primitives::digit_histogram<Tkey, radix_bits>(sched, buf_key, buf_hist, len);
        }

        std::vector<u32> expected = reference_histogram<Tkey, radix_bits>(keys, len);
        std::vector<u32> result   = buf_hist.copy_to_stdvec();

        REQUIRE_EQUAL_NAMED(
            name + " radix_bits=" + std::to_string(radix_bits) + " len=" + std::to_string(len),
            result,
            expected);
    }

    template<class Tkey, u32 radix_bits>
    void run_cases(const std::string &tname) {
        // whole buffer (sizes around the chunk size of 2048 keys and with many chunks)
        for (u32 len : {0u, 1u, 255u, 2047u, 2048u, 2049u, 100003u, (1u << 21) + 1}) {
            check_case<Tkey, radix_bits>(len, len, len * 3 + 1, tname);
        }
        // only a prefix of a larger buffer
        for (u32 len : {0u, 1u, 2049u, 100003u}) {
            check_case<Tkey, radix_bits>(len + 777, len, len + 5, tname + " prefix");
        }
    }

} // namespace

NEW_TEST(Unittest, "shamalgs/primitives/digit_histogram", 1) {
    run_cases<u32, 1>("u32");
    run_cases<u32, 2>("u32");
    run_cases<u32, 4>("u32");
    run_cases<u32, 8>("u32");
    run_cases<u64, 1>("u64");
    run_cases<u64, 2>("u64");
    run_cases<u64, 4>("u64");
    run_cases<u64, 8>("u64");
}
