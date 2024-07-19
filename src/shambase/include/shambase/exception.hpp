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
 *
 * @brief This header file contains utility functions related to exception handling in the code.
 *
 * These functions are used to format the exception messages with the source location
 * information.
 */

 
#include "shambase/SourceLocation.hpp"
#include <stdexcept>

namespace shambase {

    /**
     * @brief Format the exception message with the source location information
     *
     * This function formats a string that contains the source location
     * information. It is usefull to add this information to the exception
     * message, in order to have a better understanding of where the exception
     * was thrown.
     *
     * @param loc The location from where the exception was thrown
     * @return std::string The formatted exception message
     */
    std::string exception_format(SourceLocation loc);


    /**
     * @brief Create an exception with a message and a location
     *
     * This function creates an exception with a message that is richer,
     * as it also contains the source location where the exception was thrown.
     *
     * @tparam ExcptTypes The type of the exception to create
     * @param message The message of the exception
     * @param loc The location from where the exception was thrown
     * @return ExcptTypes The exception with a message that also contains
     * the source location.
     */
    template<class ExcptTypes>
    inline ExcptTypes make_except_with_loc(std::string message, SourceLocation loc = SourceLocation{}) {
        return ExcptTypes(message + exception_format(loc));
    }

    /**
     * @brief Throw an exception and append the source location to it
     *
     * This function allows to throw an exception with a message
     * that is richer, as it also contains the source location
     * where the exception was thrown.
     *
     * Usage : 
     * ~~~~~{.cpp}
     * shambase::throw_with_loc<MyException>("message");
     * ~~~~~
     * 
     * @tparam ExcptTypes The type of the exception to throw
     * @param message The message of the exception
     * @param loc The location from where the exception was thrown
     *
     * @throw ExcptTypes The exception with a message that also contains
     * the source location.
     */
    template<class ExcptTypes>
    inline void throw_with_loc(std::string message, SourceLocation loc = SourceLocation{}) {
        throw make_except_with_loc<ExcptTypes>(message + exception_format(loc));
    }

    /**
     * @brief Throw a std::runtime_error saying that the function is unimplemented
     *
     * This function is a convenient way to throw a
     * std::runtime_error saying that the function is unimplemented.
     * It also includes the source location where the exception was thrown.
     *
     * Usage : 
     * ~~~~~{.cpp}
     * shambase::throw_unimplemented();
     * ~~~~~
     * 
     * @param loc The location from where the exception was thrown
     *
     * @throw std::runtime_error The exception with a message saying that the
     * function is unimplemented, and the source location.
     */
    inline void throw_unimplemented(SourceLocation loc = SourceLocation{}) {
        throw_with_loc<std::runtime_error>("unimplemented",loc);
    }

    /**
     * @brief Throw a std::runtime_error with a message saying that the function is unimplemented
     *
     * This function is a convenient way to throw a std::runtime_error
     * saying that the function is unimplemented. It also includes the
     * source location where the exception was thrown.
     *
     * Usage : 
     * ~~~~~{.cpp}
     * shambase::throw_unimplemented("my function");
     * ~~~~~
     *
     * @param message The message to add to the exception message
     * @param loc The location from where the exception was thrown
     *
     * @throw std::runtime_error The exception with a message saying that the
     * function is unimplemented, the given message, and the source location.
     */
    inline void throw_unimplemented(std::string message,SourceLocation loc = SourceLocation{}) {
        throw_with_loc<std::runtime_error>(message + "\nunimplemented",loc);
    }

} // namespace shambase
