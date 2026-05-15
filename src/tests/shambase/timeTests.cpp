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

TestStart(Unittest, "shambase/time/start_stop_elapsed_gt_zero", unitt_timer_start_stop_elapsed, 1) {

    shambase::Timer timer;

    timer.start();
    std::this_thread::sleep_for(std::chrono::milliseconds(50));
    timer.stop();

    REQUIRE(timer.elapsed_sec() > 0);

    std::string time_str = timer.get_time_str();
    REQUIRE(!time_str.empty());
}

TestStart(Unittest, "shambase/time/sleep_200ms_precision", unitt_timer_sleep_200ms, 1) {

    shambase::Timer timer;
    timer.start();
    std::this_thread::sleep_for(std::chrono::milliseconds(200));
    timer.stop();

    REQUIRE_FLOAT_EQUAL(timer.elapsed_sec(), 0.2, 0.05);
}

TestStart(Unittest, "shambase/time/stop_overwrites_nanosec", unitt_timer_stop_overwrites, 1) {

    shambase::Timer timer;

    timer.start();
    std::this_thread::sleep_for(std::chrono::milliseconds(300));
    timer.stop();
    f64 elapsed1 = timer.elapsed_sec();

    timer.stop();
    f64 elapsed2 = timer.elapsed_sec();

    REQUIRE(elapsed1 < elapsed2);
}

TestStart(Unittest, "shambase/time/reusability", unitt_timer_reusability, 1) {

    shambase::Timer timer;

    timer.start();
    std::this_thread::sleep_for(std::chrono::milliseconds(200));
    timer.stop();
    f64 elapsed1 = timer.elapsed_sec();

    timer.start();
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
    timer.stop();
    f64 elapsed2 = timer.elapsed_sec();

    REQUIRE(elapsed1 > elapsed2);
}

TestStart(Unittest, "shambase/time/get_time_str_has_unit", unitt_timer_get_time_str_format, 1) {

    shambase::Timer timer;
    timer.start();
    std::this_thread::sleep_for(std::chrono::milliseconds(50));
    timer.stop();

    std::string s = timer.get_time_str();

    REQUIRE(!s.empty());
    REQUIRE(s.find("ms") != std::string::npos || s.find("us") != std::string::npos);
}

TestStart(
    Unittest, "shambase/time/nanosec_to_time_str_all_units", unitt_nanosec_to_time_str_various, 1) {

    using namespace shambase;

    REQUIRE(nanosec_to_time_str(0) == "0.00 ns");
    REQUIRE(nanosec_to_time_str(500) == "500.00 ns");
    REQUIRE(nanosec_to_time_str(2500) == "2.50 us");
    REQUIRE(nanosec_to_time_str(5000000) == "5.00 ms");
    REQUIRE(nanosec_to_time_str(2500000) == "2.50 ms");
    REQUIRE(nanosec_to_time_str(5000000000) == "5.00 s");
    REQUIRE(nanosec_to_time_str(2500000000) == "2.50 s");
    REQUIRE(nanosec_to_time_str(2500000000000) == "2.50 ks");
    REQUIRE(nanosec_to_time_str(2500000000000000) == "2.50 Ms");
    REQUIRE(nanosec_to_time_str(2500000000000000000) == "2.50 Gs");
}
