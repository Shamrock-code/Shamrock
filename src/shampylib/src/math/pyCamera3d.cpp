// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright (c) 2021-2026 Timothée David--Cléris <tim.shamrock@proton.me>
// SPDX-License-Identifier: CeCILL Free Software License Agreement v2.1
// Shamrock is licensed under the CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

/**
 * @file pyCamera3d.cpp
 * @author Yann Bernard (yann.bernard@univ-grenoble-alpes.fr)
 * @brief
 */

#include "shampylib/math/pyCamera3d.hpp"
#include "shambindings/pybindaliases.hpp"
#include "shambindings/pytypealias.hpp"
#include "shammath/Camera3d.hpp"
#include <pybind11/stl.h>

namespace shampylib {

    template<class Tvec>
    void init_shamrock_math_Camera3d(py::module &m, std::string name) {
        using Camera = shammath::Camera3d<Tvec>;
        using Tscal  = typename Camera::Tscal;

        py::class_<Camera>(m, name.c_str(), R"==(
    Perspective camera following the OpenGL conventions (gluLookAt + gluPerspective).

    Pixel ``(ix, iy)`` covers ``ix`` from left to right and ``iy`` from bottom to top,
    its ray goes through the pixel center, starts on the near plane and extends to
    infinity, so that nothing behind the image plane is rendered.
)==")
            .def(
                py::init<Tvec, Tvec, Tvec, u32, u32, Tscal, Tscal, Tscal>(),
                py::arg("pos"),
                py::arg("dir"),
                py::arg("up"),
                py::arg("nx"),
                py::arg("ny"),
                py::arg("fovy"),
                py::arg("znear"),
                py::arg("zfar"),
                "Camera with an aspect ratio of nx / ny, fovy being the vertical field of view in "
                "radians")
            .def_static(
                "from_aspect",
                &Camera::from_aspect,
                py::arg("pos"),
                py::arg("dir"),
                py::arg("up"),
                py::arg("nx"),
                py::arg("aspect"),
                py::arg("fovy"),
                py::arg("znear"),
                py::arg("zfar"),
                "Camera with ny = round(nx / aspect), fovy being the vertical field of view in "
                "radians")
            .def_static(
                "look_at",
                &Camera::look_at,
                py::arg("pos"),
                py::arg("target"),
                py::arg("up"),
                py::arg("nx"),
                py::arg("ny"),
                py::arg("fovy"),
                py::arg("znear"),
                py::arg("zfar"),
                "Camera at pos looking at target (gluLookAt convention), fovy being the vertical "
                "field of view in radians")
            .def("get_pos", &Camera::get_pos)
            .def("get_dir", &Camera::get_dir)
            .def("get_up", &Camera::get_up)
            .def("get_right", &Camera::get_right)
            .def("get_nx", &Camera::get_nx)
            .def("get_ny", &Camera::get_ny)
            .def("get_aspect", &Camera::get_aspect)
            .def("get_fovy", &Camera::get_fovy)
            .def("get_znear", &Camera::get_znear)
            .def("get_zfar", &Camera::get_zfar)
            .def("get_pixel_ray", &Camera::get_pixel_ray, py::arg("ix"), py::arg("iy"))
            .def(
                "get_rays",
                &Camera::get_rays,
                "Rays of all the pixels, pixel (ix, iy) being at index iy * nx + ix")
            .def("get_view_matrix", &Camera::get_view_matrix)
            .def("get_view_matrix_inv", &Camera::get_view_matrix_inv)
            .def("get_projection_matrix", &Camera::get_projection_matrix)
            .def("get_projection_matrix_inv", &Camera::get_projection_matrix_inv)
            .def("get_camera_transform", &Camera::get_camera_transform)
            .def("get_camera_transform_inv", &Camera::get_camera_transform_inv);
    }

    template void init_shamrock_math_Camera3d<f64_3>(py::module &m, std::string name);

} // namespace shampylib
