// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright(C) 2021-2023 Timothée David--Cléris <timothee.david--cleris@ens-lyon.fr>
// Licensed under CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

#pragma once

/**
 * @file print.hpp
 * @author Timothée David--Cléris (tim.shamrock@proton.me)
 * @brief
 *
 */

#include <string>

namespace shambase {

    void print(std::string s);
    void println(std::string s);
    void flush();

    void change_printer(
        void (*func_printer_normal)(std::string),
        void (*func_printer_ln)(std::string),
        void (*func_flush_func)());

    void reset_std_behavior();

} // namespace shambase
