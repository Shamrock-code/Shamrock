// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright(C) 2021-2023 Timothée David--Cléris <timothee.david--cleris@ens-lyon.fr>
// Licensed under CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

#pragma once

/**
 * @file pybindings.hpp
 * @author Timothée David--Cléris (timothee.david--cleris@ens-lyon.fr)
 * @brief 
 * 
 */

#include <shambindings/pybindaliases.hpp>

namespace shambindings {

    /**
     * @brief Init python bindings and register them to Python API
     * 
     * @param m the python module to bind definitions on
     */
    void init(py::module & m);

}