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
 * @file constants.hpp
 * @author Timothée David--Cléris (tim.shamrock@proton.me)
 * @brief Class holding the value of numerous constants
 * generated from the following source

        from math import *

        def add(name, value):
            print("template<class T> constexpr T {} = {:.20g};".format(name,value))


        add("unity",1)
        add("zero",0)
        add("pi",pi)
        add("tau",2*pi)
        add("pi_square",pi*pi)
        add("pi2_sqrt",(2*pi)**(1/2))
        add("gamma_1_6", gamma(1/6))
        add("gamma_1_5", gamma(1/5))
        add("gamma_1_4", gamma(1/4))
        add("gamma_1_3", gamma(1/3))
        add("gamma_2_5", gamma(2/5))
        add("gamma_1_2", gamma(1/2))
        add("gamma_3_5", gamma(3/5))
        add("gamma_2_3", gamma(2/3))
        add("gamma_3_4", gamma(3/4))
        add("gamma_4_5", gamma(4/5))
        add("gamma_5_6", gamma(5/6))
        add("gamma_1", gamma(1))
        add("sqrt_2", 2**(1/2))
        add("e", e)
  */

namespace shambase::constants {

    // clang-format off
    template<class T> constexpr T unity = 1;
    template<class T> constexpr T zero = 0;
    template<class T> constexpr T pi = 3.141592653589793116;
    template<class T> constexpr T tau = 6.283185307179586232;
    template<class T> constexpr T pi_square = 9.8696044010893579923;
    template<class T> constexpr T pi2_sqrt = 2.5066282746310002416;
    template<class T> constexpr T gamma_1_6 = 5.5663160017802360002;
    template<class T> constexpr T gamma_1_5 = 4.590843711998803478;
    template<class T> constexpr T gamma_1_4 = 3.6256099082219086505;
    template<class T> constexpr T gamma_1_3 = 2.6789385347077478983;
    template<class T> constexpr T gamma_2_5 = 2.2181595437576877572;
    template<class T> constexpr T gamma_1_2 = 1.7724538509055158819;
    template<class T> constexpr T gamma_3_5 = 1.4891922488128168656;
    template<class T> constexpr T gamma_2_3 = 1.3541179394264004632;
    template<class T> constexpr T gamma_3_4 = 1.225416702465177865;
    template<class T> constexpr T gamma_4_5 = 1.1642297137253030392;
    template<class T> constexpr T gamma_5_6 = 1.1287870299081257386;
    template<class T> constexpr T gamma_1 = 1;
    template<class T> constexpr T sqrt_2 = 1.4142135623730951455;
    template<class T> constexpr T e = 2.7182818284590450908;
    // clang-format on

} // namespace shambase::constants
