// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright(C) 2021-2023 Timothée David--Cléris <timothee.david--cleris@ens-lyon.fr>
// Licensed under CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

#include "pybindaliases.hpp"

#include <pybind11/iostream.h>


SHAMROCK_PY_MODULE(shamrock,m){

    #ifdef SHAMROCK_LIB_BUILD
    m.attr("redirect_output") = py::capsule(new py::scoped_ostream_redirect(
        std::cout,                               // std::ostream&
        py::module_::import("sys").attr("stdout") // Python output
    ),
    [](void *sor) { delete static_cast<py::scoped_ostream_redirect *>(sor); });
    #endif

    for(auto fct : static_init_shamrock_pybind){
        fct(m);
    }
}
