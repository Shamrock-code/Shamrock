// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright(C) 2021-2023 Timothée David--Cléris <timothee.david--cleris@ens-lyon.fr>
// Licensed under CeCILL 2.1 License, see LICENSE for more information
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

    shamtest::asserts().assert_bool("succesfull", hdnl.exec(R"(
            a=0
        )"));
}

TestStart(Unittest, "shamtest/PyScriptHandle(run)", shamtestpyscriptrunfail, 1) {

    PyScriptHandle hdnl{};

    shamtest::asserts().assert_bool("fail", !hdnl.exec(R"(
            a=b
        )"));
}

TestStart(Unittest, "shamtest/PyScriptHandle(shamrock)", shamtestpyscriptrunshamrockmodule, 1) {

    PyScriptHandle hdnl{};

    shamtest::asserts().assert_bool("success", hdnl.exec(R"(
            import shamrock

        )"));
}
