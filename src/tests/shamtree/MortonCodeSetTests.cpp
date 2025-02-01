// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright (c) 2021-2024 Timothée David--Cléris <tim.shamrock@proton.me>
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
#include <vector>

TestStart(Unittest, "shamtree/MortonCodeSet", test_morton_codeset, 1) {

    using Tvec              = f64_3;
    using Tmorton           = u64;
    shammath::AABB<Tvec> bb = shammath::AABB<Tvec>(Tvec(0, 0, 0), Tvec(1, 1, 1));

    std::vector<Tvec> partpos{
        Tvec(0, 0, 0),
        Tvec(0.1, 0.1, 0.1),
        Tvec(0.2, 0.2, 0.2),
        Tvec(0.3, 0.3, 0.3),
        Tvec(0.4, 0.4, 0.4),
        Tvec(0.5, 0.5, 0.5),
        Tvec(0.9, 0.9, 0.9),
        Tvec(1, 1, 1),
        Tvec(2, 2, 2),
        Tvec(-1, -1, -1)};

    shamcomm::logs::raw_ln(partpos);

    sham::DeviceBuffer<Tvec> partpos_buf(
        partpos.size(), shamsys::instance::get_compute_scheduler_ptr());

    auto set = shammath::tree::MortonCodeSet<Tmorton, Tvec, 3>(
        shamsys::instance::get_compute_scheduler_ptr(), bb, partpos_buf, partpos.size(), 16);

    std::vector<Tmorton> test_mortons
        = {0U,
           17737253917028415U,
           141898031336227320U,
           1011023473270619655U,
           1135184250689818560U,
           8070450532247928832U,
           9205634782937747392U,
           9223372036854775807U,
           9223372036854775807U,
           0U,
           18446744073709551615U,
           18446744073709551615U,
           18446744073709551615U,
           18446744073709551615U,
           18446744073709551615U,
           18446744073709551615U};

    logger::raw_ln("test mortons: ", test_mortons);
    std::vector<Tmorton> mortons = set.morton_codes.copy_to_stdvec();
    logger::raw_ln("calculated mortons: ", mortons);

    REQUIRE(set.cnt_obj == partpos.size());
    REQUIRE(set.morton_count == 16);
    REQUIRE(set.morton_codes.get_size() == 16);
    REQUIRE(set.morton_codes.copy_to_stdvec() == test_mortons);
}
