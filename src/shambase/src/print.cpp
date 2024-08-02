// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright(C) 2021-2023 Timothée David--Cléris <timothee.david--cleris@ens-lyon.fr>
// Licensed under CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

/**
 * @file print.cpp
 * @author Timothée David--Cléris (tim.shamrock@proton.me)
 * @brief
 *
 */

#include "shambase/print.hpp"
#include <iostream>
#include <ostream>

namespace shambase {

    void (*_printer)(std::string)   = nullptr; ///< The print function pointer to use if not null
    void (*_printerln)(std::string) = nullptr; ///< The println function pointer to use if not null
    void (*_flush)()                = nullptr; ///< The flush function pointer to use if not null

    void print(std::string s) {
        if (_printer == nullptr) {
            std::cout << s;
        } else {
            _printer(s);
        }
    }
    void println(std::string s) {
        if (_printerln == nullptr) {
            std::cout << s << "\n";
        } else {
            _printerln(s);
        }
    }
    void flush() {
        if (_flush == nullptr) {
            std::cout << std::flush;
        } else {
            flush();
        }
    }

    void change_printer(
        void (*func_printer_normal)(std::string),
        void (*func_printer_ln)(std::string),
        void (*func_flush_func)()) {
        _printer   = func_printer_normal;
        _printerln = func_printer_ln;
        _flush     = func_flush_func;
    }

    void reset_std_behavior() { change_printer(nullptr, nullptr, nullptr); }

} // namespace shambase
