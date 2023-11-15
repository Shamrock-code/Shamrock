// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright(C) 2021-2023 Timothée David--Cléris <timothee.david--cleris@ens-lyon.fr>
// Licensed under CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

#pragma once

/**
 * @file exception.hpp
 * @author Timothée David--Cléris (timothee.david--cleris@ens-lyon.fr)
 * @brief 
 * 
 */
 
#include "shambase/SourceLocation.hpp"

namespace shambase {

    std::string exception_format(SourceLocation loc);

    /**
     * @brief Throw an exception and append the source location to it
     *
     * Usage : 
     * ~~~~~{.cpp}
     * throw shambase::throw_with_loc<MyException>("message");
     * ~~~~~
     * 
     * @tparam ExcptTypes 
     * @param message 
     * @param loc 
     * @return ExcptTypes 
     */
    template<class ExcptTypes>
    inline ExcptTypes throw_with_loc(std::string message, SourceLocation loc = SourceLocation{}) {
        return ExcptTypes(message + exception_format(loc));
    }

    inline void throw_unimplemented(SourceLocation loc = SourceLocation{}) {
        throw throw_with_loc<std::runtime_error>("unimplemented",loc);
    }

    inline void throw_unimplemented(std::string message,SourceLocation loc = SourceLocation{}) {
        throw throw_with_loc<std::runtime_error>(message + "\nunimplemented",loc);
    }

} // namespace shambase
