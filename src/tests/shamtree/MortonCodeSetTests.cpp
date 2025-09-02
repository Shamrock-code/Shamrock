// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright (c) 2021-2025 Timothée David--Cléris <tim.shamrock@proton.me>
// SPDX-License-Identifier: CeCILL Free Software License Agreement v2.1
// Shamrock is licensed under the CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

#include "shambase/integer.hpp"
#include "shambackends/DeviceBuffer.hpp"
#include "shambackends/fmt_bindings/fmt_defs.hpp"
#include "shamcomm/logs.hpp"
#include "shammath/AABB.hpp"
#include "shamsys/NodeInstance.hpp"
#include "shamtest/details/TestResult.hpp"
#include "shamtest/shamtest.hpp"
#include "shamtree/MortonCodeSet.hpp"
#include "shamtree/MortonCodeSortedSet.hpp"
#include <vector>

using Tvec    = f64_3;
using Tmorton = u64;

inline std::vector<Tvec> partpos{
    Tvec(0, 0, 0),
    Tvec(0.1, 0.0, 0.0),
    Tvec(0.0, 0.1, 0.0),
    Tvec(0.0, 0.0, 0.1),
    Tvec(0.1, 0.1, 0.0),
    Tvec(0.0, 0.1, 0.1),
    Tvec(0.1, 0.0, 0.1),
    Tvec(0.1, 0.1, 0.1),
    Tvec(0.2, 0.2, 0.2),
    Tvec(0.3, 0.3, 0.3),
    Tvec(0.4, 0.4, 0.4),
    Tvec(1, 1, 1),
    Tvec(2, 2, 2),
    Tvec(-1, -1, -1)};

inline std::vector<Tmorton> test_mortons
    = {0U,
       10135573666873380U,
       5067786833436690U,
       2533893416718345U,
       15203360500310070U,
       7601680250155035U,
       12669467083591725U,
       17737253917028415U,
       141898031336227320U,
       1011023473270619655U,
       1135184250689818560U,
       9223372036854775807U,
       9223372036854775807U,
       0U,
       18446744073709551615U,
       18446744073709551615U};

inline std::vector<Tmorton> test_mortons_sorted
    = {0U,
       0U,
       2533893416718345U,
       5067786833436690U,
       7601680250155035U,
       10135573666873380U,
       12669467083591725U,
       15203360500310070U,
       17737253917028415U,
       141898031336227320U,
       1011023473270619655U,
       1135184250689818560U,
       9223372036854775807U,
       9223372036854775807U,
       18446744073709551615U,
       18446744073709551615U};

inline std::vector<u32> index_map_obj_idx = {13, 0, 3, 2, 5, 1, 6, 4, 7, 8, 9, 10, 11, 12, 15, 14};

TestStart(Unittest, "shamtree/MortonCodeSet", test_morton_codeset, 1) {

    shammath::AABB<Tvec> bb = shammath::AABB<Tvec>(Tvec(0, 0, 0), Tvec(1, 1, 1));

    sham::DeviceBuffer<Tvec> partpos_buf(
        partpos.size(), shamsys::instance::get_compute_scheduler_ptr());

    partpos_buf.copy_from_stdvec(partpos);

    auto set = shamtree::morton_code_set_from_positions<Tmorton, Tvec, 3>(
        shamsys::instance::get_compute_scheduler_ptr(), bb, partpos_buf, partpos.size(), 16);

    REQUIRE_EQUAL(set.cnt_obj, partpos.size());
    REQUIRE_EQUAL(set.morton_count, 16);
    REQUIRE_EQUAL(set.morton_codes.get_size(), 16);
    REQUIRE_EQUAL(set.morton_codes.copy_to_stdvec(), test_mortons);
}

TestStart(Unittest, "shamtree/MortonCodeSortedSet", test_morton_code_sort_set, 1) {

    shammath::AABB<Tvec> bb = shammath::AABB<Tvec>(Tvec(0, 0, 0), Tvec(1, 1, 1));

    sham::DeviceBuffer<Tvec> partpos_buf(
        partpos.size(), shamsys::instance::get_compute_scheduler_ptr());

    partpos_buf.copy_from_stdvec(partpos);

    auto set = shamtree::morton_code_set_from_positions<Tmorton, Tvec, 3>(
        shamsys::instance::get_compute_scheduler_ptr(), bb, partpos_buf, partpos.size(), 16);

    std::vector<Tmorton> mortons = set.morton_codes.copy_to_stdvec();

    REQUIRE_EQUAL(set.cnt_obj, partpos.size());
    REQUIRE_EQUAL(set.morton_count, 16);
    REQUIRE_EQUAL(set.morton_codes.get_size(), 16);
    REQUIRE_EQUAL(set.morton_codes.copy_to_stdvec(), test_mortons);

    auto sorted_set
        = shamtree::sort_morton_set(shamsys::instance::get_compute_scheduler_ptr(), std::move(set));

    REQUIRE_EQUAL(sorted_set.cnt_obj, partpos.size());
    REQUIRE_EQUAL(sorted_set.morton_count, 16);
    REQUIRE_EQUAL(sorted_set.sorted_morton_codes.get_size(), 16);
    REQUIRE_EQUAL(sorted_set.sorted_morton_codes.copy_to_stdvec(), test_mortons_sorted);
    REQUIRE_EQUAL(sorted_set.map_morton_id_to_obj_id.get_size(), 16);
    REQUIRE_EQUAL(sorted_set.map_morton_id_to_obj_id.copy_to_stdvec(), index_map_obj_idx);
}
