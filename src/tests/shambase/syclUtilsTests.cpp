// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright (c) 2021-2025 Timothée David--Cléris <tim.shamrock@proton.me>
// SPDX-License-Identifier: CeCILL Free Software License Agreement v2.1
// Shamrock is licensed under the CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

#include "shambackends/sycl_utils.hpp"
#include "shamsys/NodeInstance.hpp"
#include "shamsys/legacy/log.hpp"
#include "shamtest/details/TestResult.hpp"
#include "shamtest/shamtest.hpp"

TestStart(Unittest, "shambase::parallel_for", test_par_for_1d, 1) {
    u32 len = 10000;
    sycl::buffer<u64> buf(len);

    shamsys::instance::get_compute_queue().submit([&](sycl::handler &cgh) {
        sycl::accessor acc{buf, cgh, sycl::write_only, sycl::no_init};
        shambase::parallel_for(cgh, len, "test 1d par for", [=](u64 id) {
            acc[id] = id;
        });
    });

    bool correct = true;
    {
        sycl::host_accessor acc{buf, sycl::read_only};
        for (u32 x = 0; x < len; x++) {
            if (acc[x] != x) {
                correct = false;
            }
        }
    }
    REQUIRE(correct);
}

TestStart(Unittest, "shambase::parallel_for", test_par_for_2d, 1) {
    u64 len_x = 1000;
    u64 len_y = 1000;
    sycl::buffer<u64, 2> buf(sycl::range<2>{len_x, len_y});

    shamsys::instance::get_compute_queue().submit([&](sycl::handler &cgh) {
        sycl::accessor acc{buf, cgh, sycl::write_only, sycl::no_init};
        shambase::parallel_for_2d(cgh, len_x, len_y, "test 2d par for", [=](u64 id_x, u64 id_y) {
            acc[{id_x, id_y}] = id_x + len_x * id_y;
        });
    });

    bool correct = true;
    {
        sycl::host_accessor acc{buf, sycl::read_only};
        for (u32 x = 0; x < len_x; x++) {
            for (u32 y = 0; y < len_y; y++) {
                if (acc[{x, y}] != x + len_x * y) {
                    correct = false;
                    logger::err_ln("Test", "fail : ", x, y, ":", acc[{x, y}], "!=", x + len_x * y);
                    break;
                }
            }
        }
    }
    REQUIRE(correct);
}

TestStart(Unittest, "shambase::parallel_for", test_par_for_3d, 1) {
    u64 len_x = 100;
    u64 len_y = 100;
    u64 len_z = 100;
    sycl::buffer<u64, 3> buf(sycl::range<3>{len_x, len_y, len_z});

    shamsys::instance::get_compute_queue().submit([&](sycl::handler &cgh) {
        sycl::accessor acc{buf, cgh, sycl::write_only, sycl::no_init};
        shambase::parallel_for_3d(
            cgh, len_x, len_y, len_z, "test 2d par for", [=](u64 id_x, u64 id_y, u64 id_z) {
                acc[{id_x, id_y, id_z}] = id_x + len_x * id_y + len_x * len_y * id_z;
            });
    });

    bool correct = true;
    {
        sycl::host_accessor acc{buf, sycl::read_only};
        for (u32 x = 0; x < len_x; x++) {
            for (u32 y = 0; y < len_y; y++) {
                for (u32 z = 0; z < len_z; z++) {
                    if (acc[{x, y, z}] != x + len_x * y + len_x * len_y * z) {
                        correct = false;
                        logger::err_ln(
                            "Test",
                            "fail : ",
                            x,
                            y,
                            z,
                            ":",
                            acc[{x, y, z}],
                            "!=",
                            x + len_x * y + len_x * len_y * z);
                        break;
                    }
                }
            }
        }
    }
    REQUIRE(correct);
}
