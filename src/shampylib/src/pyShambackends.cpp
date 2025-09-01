// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright (c) 2021-2025 Timothée David--Cléris <tim.shamrock@proton.me>
// SPDX-License-Identifier: CeCILL Free Software License Agreement v2.1
// Shamrock is licensed under the CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

/**
 * @file pyShambackends.cpp
 * @author Timothée David--Cléris (tim.shamrock@proton.me)
 * @brief
 */

#include "shambackends/DeviceBuffer.hpp"
#include "shambindings/pybind11_stl.hpp"
#include "shambindings/pybindaliases.hpp"
#include "shambindings/pytypealias.hpp"
#include "shamcomm/logs.hpp"
#include "shamsys/NodeInstance.hpp"
#include <pybind11/complex.h>

template<class T>
inline void register_DeviceBuffer(py::module &m, const char *class_name) {

    shamcomm::logs::debug_ln("[Py]", "registering shamrock.backends." + std::string(class_name));
    py::class_<sham::DeviceBuffer<T>>(m, class_name)
        .def(py::init([]() {
            return std::make_unique<sham::DeviceBuffer<T>>(
                sham::DeviceBuffer<T>{0, shamsys::instance::get_compute_scheduler_ptr()});
        }))
        .def(
            "get_size",
            [](sham::DeviceBuffer<T> &self) {
                return self.get_size();
            })
        .def(
            "resize",
            [](sham::DeviceBuffer<T> &self, u32 sz) {
                self.resize(sz);
            })
        .def(
            "get_val_at_idx",
            [](sham::DeviceBuffer<T> &self, u32 idx) {
                return self.get_val_at_idx(idx);
            })
        .def(
            "set_val_at_idx",
            [](sham::DeviceBuffer<T> &self, u32 idx, T val) {
                self.set_val_at_idx(idx, val);
            })
        .def(
            "fill",
            [](sham::DeviceBuffer<T> &self, T val) {
                self.fill(val);
            })
        .def(
            "copy_from_stdvec",
            [](sham::DeviceBuffer<T> &self, const std::vector<T> &v) {
                self.copy_from_stdvec(v);
            })
        .def("copy_to_stdvec", [](sham::DeviceBuffer<T> &self) {
            return self.copy_to_stdvec();
        });
}

Register_pymod(shambackendslibinit) {

    py::module shambackends_module = m.def_submodule("backends", "backend library");

    register_DeviceBuffer<u8>(shambackends_module, "DeviceBuffer_u8");
    register_DeviceBuffer<u32>(shambackends_module, "DeviceBuffer_u32");
    register_DeviceBuffer<f64>(shambackends_module, "DeviceBuffer_f64");
    register_DeviceBuffer<f64_2>(shambackends_module, "DeviceBuffer_f64_2");
    register_DeviceBuffer<f64_3>(shambackends_module, "DeviceBuffer_f64_3");
}
