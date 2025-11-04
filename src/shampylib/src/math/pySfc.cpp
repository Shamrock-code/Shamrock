// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright (c) 2021-2025 Timothée David--Cléris <tim.shamrock@proton.me>
// SPDX-License-Identifier: CeCILL Free Software License Agreement v2.1
// Shamrock is licensed under the CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

/**
 * @file pySfc.cpp
 * @author Timothée David--Cléris (tim.shamrock@proton.me)
 * @brief
 */

#include "shambindings/pybindaliases.hpp"
#include "shambindings/pytypealias.hpp"
#include "shammath/sfc/bmi.hpp"
#include "shammath/sfc/hilbert.hpp"
#include "shammath/sfc/morton.hpp"
#include <variant>

enum ReprType { _u32, _u64 };

using namespace shamrock::sfc;

namespace shampylib {

    void init_shamrock_math_sfc(py::module &m) {

        py::module sfc_module = m.def_submodule("sfc", "Space Filling curve Library");

        sfc_module.def(
            "icoord_to_morton_3d_u64",
            [](u32 x, u32 y, u32 z) -> u64 {
                return MortonCodes<u64, 3>::icoord_to_morton(x, y, z);
            },
            R"pbdoc(
        Convert a 3d integer coord to morton code
    )pbdoc");

        sfc_module.def(
            "morton_to_icoord_3d_u64",
            [](u64 m) -> std::array<u32, 3> {
                auto ret = MortonCodes<u64, 3>::morton_to_icoord(m);
                return {u32{ret.x()}, u32{ret.y()}, u32{ret.z()}};
            },
            R"pbdoc(
        Convert a morton code to 3d integer coord
    )pbdoc");

        sfc_module.def(
            "icoord_to_morton_3d_u32",
            [](u16 x, u16 y, u16 z) -> u32 {
                return MortonCodes<u32, 3>::icoord_to_morton(x, y, z);
            },
            R"pbdoc(
        Convert a 3d integer coord to morton code
    )pbdoc");

        sfc_module.def(
            "morton_to_icoord_3d_u32",
            [](u32 m) -> std::array<u16, 3> {
                auto ret = MortonCodes<u32, 3>::morton_to_icoord(m);
                return {u16{ret.x()}, u16{ret.y()}, u16{ret.z()}};
            },
            R"pbdoc(
        Convert a morton code to 3d integer coord
    )pbdoc");

        sfc_module.def(
            "to_morton_grid_3d_u32_f32_3",
            [](std::array<f32, 3> pos,
               std::array<f32, 3> bmin,
               std::array<f32, 3> bmax) -> std::array<u16, 3> {
                f32_3 pos_{pos[0], pos[1], pos[2]};
                f32_3 bmin_{bmin[0], bmin[1], bmin[2]};
                f32_3 bmax_{bmax[0], bmax[1], bmax[2]};
                auto tmp = MortonConverter<u32, f32_3, 3>::get_transform(bmin_, bmax_);
                auto ret = MortonConverter<u32, f32_3, 3>::to_morton_grid(pos_, tmp);
                return {u16{ret.x()}, u16{ret.y()}, u16{ret.z()}};
            });

        sfc_module.def(
            "to_morton_grid_3d_u32_f64_3",
            [](std::array<f64, 3> pos,
               std::array<f64, 3> bmin,
               std::array<f64, 3> bmax) -> std::array<u16, 3> {
                f64_3 pos_{pos[0], pos[1], pos[2]};
                f64_3 bmin_{bmin[0], bmin[1], bmin[2]};
                f64_3 bmax_{bmax[0], bmax[1], bmax[2]};
                auto tmp = MortonConverter<u32, f64_3, 3>::get_transform(bmin_, bmax_);
                auto ret = MortonConverter<u32, f64_3, 3>::to_morton_grid(pos_, tmp);
                return {u16{ret.x()}, u16{ret.y()}, u16{ret.z()}};
            });

        sfc_module.def(
            "to_morton_grid_3d_u64_f32_3",
            [](std::array<f32, 3> pos,
               std::array<f32, 3> bmin,
               std::array<f32, 3> bmax) -> std::array<u32, 3> {
                f32_3 pos_{pos[0], pos[1], pos[2]};
                f32_3 bmin_{bmin[0], bmin[1], bmin[2]};
                f32_3 bmax_{bmax[0], bmax[1], bmax[2]};
                auto tmp = MortonConverter<u64, f32_3, 3>::get_transform(bmin_, bmax_);
                auto ret = MortonConverter<u64, f32_3, 3>::to_morton_grid(pos_, tmp);
                return {u32{ret.x()}, u32{ret.y()}, u32{ret.z()}};
            });

        sfc_module.def(
            "to_morton_grid_3d_u64_f64_3",
            [](std::array<f64, 3> pos,
               std::array<f64, 3> bmin,
               std::array<f64, 3> bmax) -> std::array<u32, 3> {
                f64_3 pos_{pos[0], pos[1], pos[2]};
                f64_3 bmin_{bmin[0], bmin[1], bmin[2]};
                f64_3 bmax_{bmax[0], bmax[1], bmax[2]};
                auto tmp = MortonConverter<u64, f64_3, 3>::get_transform(bmin_, bmax_);
                auto ret = MortonConverter<u64, f64_3, 3>::to_morton_grid(pos_, tmp);
                return {u32{ret.x()}, u32{ret.y()}, u32{ret.z()}};
            });

        sfc_module.def(
            "coord_to_hilbert_3d_u64",
            [](f64 x, f64 y, f64 z) {
                return HilbertCurve<u64, 3>::coord_to_hilbert(x, y, z);
            },
            R"pbdoc(
        Convert a 3d coordinate in the unit cube to hilbert codes
    )pbdoc");
    }

} // namespace shampylib
