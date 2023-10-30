// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright(C) 2021-2023 Timothée David--Cléris <timothee.david--cleris@ens-lyon.fr>
// Licensed under CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

#pragma once

/**
 * @file macroLocation.hpp
 * @author Timothée David--Cléris (timothee.david--cleris@ens-lyon.fr)
 * @brief 
 * 
 */

#include <cstring>
#include <string>

inline std::string __file_to_loc(const char* filename){
    return std::string(std::strstr(filename, "/src/") ? std::strstr(filename, "/src/")+1  : filename);
}

inline std::string __loc_prefix(const char* filename, int line){
    return __file_to_loc(filename)+":" + std::to_string(line);
}

#define __FILENAME__ __file_to_loc(__FILE__)
#define __LOC_PREFIX__  __loc_prefix(__FILE__,__LINE__)

#define __LOC_POSTFIX__  ("("+__LOC_PREFIX__+")")