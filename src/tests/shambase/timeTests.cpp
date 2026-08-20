// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright (c) 2021-2026 Timothée David--Cléris <tim.shamrock@proton.me>
// SPDX-License-Identifier: CeCILL Free Software License Agreement v2.1
// Shamrock is licensed under the CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

#include "shambase/time.hpp"
#include "shamtest/shamtest.hpp"
#include <thread>

NEW_TEST(Unittest, "shambase/time/start_stop_elapsed_gt_zero", 1) {

    shambase::Timer timer;

    timer.start();
    std::this_thread::sleep_for(std::chrono::milliseconds(50));
    timer.stop();

    REQUIRE(timer.elapsed_sec() > 0);

    std::string time_str = timer.get_time_str();
    REQUIRE(!time_str.empty());
}

NEW_TEST(Unittest, "shambase/time/stop_overwrites_nanosec", 1) {

    shambase::Timer timer;

    timer.start();
    std::this_thread::sleep_for(std::chrono::milliseconds(300));
    timer.stop();
    f64 elapsed1 = timer.elapsed_sec();

    std::this_thread::sleep_for(std::chrono::milliseconds(50));
    timer.stop();
    f64 elapsed2 = timer.elapsed_sec();

    REQUIRE(elapsed1 < elapsed2);
}

NEW_TEST(Unittest, "shambase/time/reusability", 1) {

    shambase::Timer timer;

    timer.start();
    std::this_thread::sleep_for(std::chrono::milliseconds(200));
    timer.stop();
    f64 elapsed1 = timer.elapsed_sec();

    timer.start();
    timer.stop();
    f64 elapsed2 = timer.elapsed_sec();

    REQUIRE(elapsed1 > elapsed2);
}

NEW_TEST(Unittest, "shambase/time/get_time_str_has_unit", 1) {

    shambase::Timer timer;
    timer.start();
    std::this_thread::sleep_for(std::chrono::milliseconds(50));
    timer.stop();

    std::string s = timer.get_time_str();

    REQUIRE(!s.empty());
    REQUIRE(s.find("ms") != std::string::npos || s.find("us") != std::string::npos);
}
