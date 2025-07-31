// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright (c) 2021-2025 Timothée David--Cléris <tim.shamrock@proton.me>
// SPDX-License-Identifier: CeCILL Free Software License Agreement v2.1
// Shamrock is licensed under the CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

#include "shamtest/PyScriptHandle.hpp"
#include "shamtest/details/TestResult.hpp"
#include "shamtest/shamtest.hpp"
#include <memory>
#include <vector>

TestStart(Unittest, "shamtest/PyScriptHandle(plot)", shamtestpyscriptplot, 1) {

    std::vector<f64> x = {0, 1, 2, 4, 5};
    std::vector<f64> y = {1, 2, 4, 6, 1};

    PyScriptHandle hdnl{};

    hdnl.data()["x"] = x;
    hdnl.data()["y"] = y;

    hdnl.exec(R"(
        import matplotlib.pyplot as plt
        plt.plot(x,y)
        plt.savefig("tests/figures/test.pdf")
    )");
}

TestStart(Unittest, "shamtest/PyScriptHandle(run)", shamtestpyscriptrun, 1) {

    PyScriptHandle hdnl{};

    REQUIRE_NAMED("succesfull", hdnl.exec(R"(
            a=0
        )"));
}

TestStart(Unittest, "shamtest/PyScriptHandle(run)", shamtestpyscriptrunfail, 1) {

    PyScriptHandle hdnl{};

    REQUIRE_NAMED("fail", !hdnl.exec(R"(
            a=b
        )"));
}

TestStart(Unittest, "shamtest/PyScriptHandle(shamrock)", shamtestpyscriptrunshamrockmodule, 1) {

    PyScriptHandle hdnl{};

    REQUIRE_NAMED("success", hdnl.exec(R"(
            import shamrock

        )"));
}
