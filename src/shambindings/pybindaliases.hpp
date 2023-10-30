// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright(C) 2021-2023 Timothée David--Cléris <timothee.david--cleris@ens-lyon.fr>
// Licensed under CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//


#pragma once

/**
 * @file pybindaliases.hpp
 * @author Timothée David--Cléris (timothee.david--cleris@ens-lyon.fr)
 * @brief 
 * 
 */
 
#ifdef SHAMROCK_EXECUTABLE_BUILD
#include <pybind11/embed.h>
#define SHAMROCK_PY_MODULE(name,module) PYBIND11_EMBEDDED_MODULE(name, module)
#endif

#ifdef SHAMROCK_LIB_BUILD
#include <pybind11/pybind11.h>
#define SHAMROCK_PY_MODULE(name,module) PYBIND11_MODULE(name, module)
#endif


namespace py = pybind11;


using fct_sig = std::function<void(py::module &)>;
inline std::vector<fct_sig> static_init_shamrock_pybind{};

struct PyBindStaticInit{
    inline explicit PyBindStaticInit(fct_sig t){
        static_init_shamrock_pybind.push_back(std::move(t));
    }
};


#define Register_pymod(placeholdername) void pymod_##placeholdername (py::module & m);\
void (*pymod_ptr_##placeholdername)(py::module & m) = pymod_##placeholdername;\
PyBindStaticInit pymod_class_obj_##placeholdername (pymod_ptr_##placeholdername);\
void pymod_##placeholdername (py::module & m)


