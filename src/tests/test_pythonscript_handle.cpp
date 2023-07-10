// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright(C) 2021-2023 Timothée David--Cléris <timothee.david--cleris@ens-lyon.fr>
// Licensed under CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

#include "shamtest/details/TestResult.hpp"
#include "shamtest/shamtest.hpp"
#include <memory>
#include <vector>

#include "shamtest/PyScriptHandle.hpp"

TestStart(Unittest, "shamtest/PyScriptHandle", shamtestpyscript, 1) {

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
