// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright(C) 2021-2023 Timothée David--Cléris <timothee.david--cleris@ens-lyon.fr>
// Licensed under CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

/**
 * @file pyPhantomDump.cpp
 * @author Timothée David--Cléris (timothee.david--cleris@ens-lyon.fr)
 * @brief 
 */
 
#include <memory>

#include "shambase/fortran_io.hpp"
#include "shambindings/pybindaliases.hpp"
#include "shambindings/pytypealias.hpp"
#include "shammodels/sph/io/PhantomDump.hpp"

Register_pymod(pyphantomdump) {

    using T = shammodels::sph::PhantomDump;

    shamcomm::logs::debug_ln("Bindings","registering shamrock.PhantomDump");
    py::class_<T>(m, "PhantomDump")
        .def("save_dump",[](T & self, std::string fname){
            auto file = self.gen_file();
            file.write_to_file(fname);
        })
        .def("read_header_float",[](T & self, std::string s){
            return self.read_header_float<f64>(s);
        })
        .def("read_header_int",[](T & self, std::string s){
            return self.read_header_int<i64>(s);
        })
        .def("print_state",&shammodels::sph::PhantomDump::print_state);

    m.def("load_phantom_dump", [](std::string fname){
        shambase::FortranIOFile phfile = shambase::load_fortran_file(fname);
        return shammodels::sph::PhantomDump::from_file(phfile);
    });



}