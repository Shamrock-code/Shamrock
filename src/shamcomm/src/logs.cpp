// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright(C) 2021-2023 Timothée David--Cléris <timothee.david--cleris@ens-lyon.fr>
// Licensed under CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

/**
 * @file logs.cpp
 * @author Timothée David--Cléris (timothee.david--cleris@ens-lyon.fr)
 * @brief
 *
 */
 
#include "shamcomm/logs.hpp"
#include <cmath>

namespace shamcomm::logs {

    reformat_func_ptr _reformat_all    = nullptr;
    reformat_func_ptr _reformat_simple = nullptr;

    void change_formaters(reformat_func_ptr full, reformat_func_ptr simple) {
        _reformat_simple = simple;
        _reformat_all    = full;
    }

} // namespace shamcomm::logs

inline std::string
reformat_all(std::string color, const char *name, std::string module_name, std::string content) {
    if (shamcomm::logs::_reformat_all == nullptr) {
        // old form
        return "[" + (color) + module_name + shambase::term_colors::reset() + "] " + (color)
               + (name) + shambase::term_colors::reset() + ": " + content;
    }

    return shamcomm::logs::_reformat_all({color, name, module_name, content});
}

inline std::string
reformat_simple(std::string color, const char *name, std::string module_name, std::string content) {

    if (shamcomm::logs::_reformat_simple == nullptr) {
        // old form
        return "[" + (color) + module_name + shambase::term_colors::reset() + "] " + (color)
               + (name) + shambase::term_colors::reset() + ": " + content;
    }

    return shamcomm::logs::_reformat_simple({color, name, module_name, content});
}

std::string LogLevel_DebugAlloc::reformat(const std::string &in, std::string module_name) {
    return ::reformat_all(shambase::term_colors::col8b_red(), level_name, module_name, in);
}

std::string LogLevel_DebugMPI::reformat(const std::string &in, std::string module_name) {
    return ::reformat_all(shambase::term_colors::col8b_blue(), level_name, module_name, in);
}

std::string LogLevel_DebugSYCL::reformat(const std::string &in, std::string module_name) {
    return ::reformat_all(shambase::term_colors::col8b_magenta(), level_name, module_name, in);
}

std::string LogLevel_Debug::reformat(const std::string &in, std::string module_name) {
    return ::reformat_all(shambase::term_colors::col8b_green(), level_name, module_name, in);
}

std::string LogLevel_Info::reformat(const std::string &in, std::string module_name) {
    return ::reformat_all(shambase::term_colors::col8b_cyan(), "Info", module_name, in);
}

std::string LogLevel_Normal::reformat(const std::string &in, std::string module_name) {
    return ::reformat_simple(shambase::term_colors::empty(), level_name, module_name, in);
}

std::string LogLevel_Warning::reformat(const std::string &in, std::string module_name) {
    return ::reformat_all(shambase::term_colors::col8b_yellow(), level_name, module_name, in);
}

std::string LogLevel_Error::reformat(const std::string &in, std::string module_name) {
    return ::reformat_all(shambase::term_colors::col8b_red(), level_name, module_name, in);
}
