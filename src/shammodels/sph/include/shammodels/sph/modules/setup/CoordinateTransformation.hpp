// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright (c) 2021-2025 Timothée David--Cléris <tim.shamrock@proton.me>
// SPDX-License-Identifier: CeCILL Free Software License Agreement v2.1
// Shamrock is licensed under the CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

#pragma once

/**
 * @file CoordinateTransformation.hpp
 * @author David Fang (david.fang@ikmail.com)
 * @brief Should be moved to shammath probably
 *
 */

#include "shambase/aliases_int.hpp"
#include "shamrock/patch/PatchDataLayer.hpp"
#include <shambackends/sycl.hpp>
#include <cstddef>
#include <string>
#include <vector>

namespace {
    size_t hashAxis(std::string str, std::string system) {
        if (system == "cartesian") {
            if (str == "x")
                return 0;
            if (str == "y")
                return 1;
            if (str == "z")
                return 2;
            else
                shamlog_error("ModifierApplyStretchMapping", "Ah non hein");
        }

        if (system == "spherical") {
            if (str == "r")
                return 0;
            if (str == "theta")
                return 1;
            if (str == "phi")
                return 2;
            else
                shamlog_error("ModifierApplyStretchMapping", "Ah non hein");
        }

        if (system == "cylindrical") {
            if (str == "r")
                return 0;
            if (str == "theta")
                return 1;
            if (str == "z")
                return 2;
            else
                shamlog_error("ModifierApplyStretchMapping", "Ah non hein");
        }

        else {
            shamlog_error("ModifierApplyStretchMapping", "Ah non hein");
        };
        return 3;
    }

    template<class Tvec>
    struct CartToSpherical {
        using Tscal = shambase::VecComponent<Tvec>;

        std::array<std::array<std::function<Tscal(Tscal)>, 3>, 3> matrix{
            {{[](Tscal r) {
                  return r;
              },
              [](Tscal theta) {
                  return sycl::sin(theta);
              },
              [](Tscal phi) {
                  return sycl::cos(phi);
              }},
             {[](Tscal r) {
                  return r;
              },
              [](Tscal theta) {
                  return sycl::sin(theta);
              },
              [](Tscal phi) {
                  return sycl::sin(phi);
              }},
             {[](Tscal r) {
                  return r;
              },
              [](Tscal theta) {
                  return sycl::cos(theta);
              },
              [](Tscal phi) {
                  return 1;
              }}}};
    };

    template<class Tvec>
    struct CartToCart { // LOL
        using Tscal = shambase::VecComponent<Tvec>;

        std::array<std::array<std::function<Tscal(Tscal)>, 3>, 3> matrix{
            {{[](Tscal x) {
                  return x;
              },
              [](Tscal y) {
                  return y;
              },
              [](Tscal z) {
                  return z;
              }},
             {[](Tscal x) {
                  return x;
              },
              [](Tscal y) {
                  return y;
              },
              [](Tscal z) {
                  return z;
              }},
             {[](Tscal x) {
                  return x;
              },
              [](Tscal y) {
                  return y;
              },
              [](Tscal z) {
                  return z;
              }}}};
    };

    template<class Tvec>
    struct CartToCylindrical {
        using Tscal = shambase::VecComponent<Tvec>;

        std::array<std::array<std::function<Tscal(Tscal)>, 3>, 3> matrix{
            {{[](Tscal r) {
                  return r;
              },
              [](Tscal theta) {
                  return sycl::cos(theta);
              },
              [](Tscal z) {
                  return 1;
              }},
             {[](Tscal r) {
                  return r;
              },
              [](Tscal theta) {
                  return sycl::sin(theta);
              },
              [](Tscal z) {
                  return 1;
              }},
             {[](Tscal r) {
                  return 1;
              },
              [](Tscal theta) {
                  return 1;
              },
              [](Tscal z) {
                  return z;
              }}}};
    };

} // namespace
