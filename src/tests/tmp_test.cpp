// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright(C) 2021-2023 Timothée David--Cléris <timothee.david--cleris@ens-lyon.fr>
// Licensed under CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

#include "shamtest/details/TestResult.hpp"
#include "shamtest/shamtest.hpp"

#include "shambindings/pytypealias.hpp"

#include <memory>
#include <pybind11/embed.h>
#include <pybind11/eval.h>
#include <pybind11/pybind11.h>
#include <pybind11/pytypes.h>
#include <vector>

struct PyscriptHandle {
    std::unique_ptr<pybind11::dict> locals;

    PyscriptHandle() {
        py::initialize_interpreter();
        locals = std::make_unique<pybind11::dict>();
    }

    ~PyscriptHandle() {
        locals.reset();
        py::finalize_interpreter();
    }

    pybind11::dict &data() { return *locals; }

    template <size_t N>
    inline void exec(const char (&expr)[N], pybind11::object &&global = pybind11::globals()) {
        try{
            py::exec(expr, std::forward<pybind11::object>(global), *locals);
        } catch (const std::exception &e) {
            std::cout << e.what() << std::endl;
        }
    }
};

TestStart(Unittest, "testpostscript", testpostscript, 1) {

    std::vector<f64> x = {0, 1, 2, 4, 5};
    std::vector<f64> y = {1, 2, 4, 6, 1};

    PyscriptHandle hdnl{};

    hdnl.data()["x"] = x;
    hdnl.data()["y"] = y;

    hdnl.exec(R"(
        print("startpy")
        import matplotlib.pyplot as plt
        plt.plot(x,y)
        plt.show()
    )");
}