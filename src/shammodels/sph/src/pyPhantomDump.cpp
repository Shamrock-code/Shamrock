// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright (c) 2021-2024 Timothée David--Cléris <tim.shamrock@proton.me>
// SPDX-License-Identifier: CeCILL Free Software License Agreement v2.1
// Shamrock is licensed under the CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

/**
 * @file pyPhantomDump.cpp
 * @author Timothée David--Cléris (timothee.david--cleris@ens-lyon.fr)
 * @brief
 */

#include "shambase/fortran_io.hpp"
#include "shambindings/pybindaliases.hpp"
#include "shambindings/pytypealias.hpp"
#include "shammodels/sph/io/Phantom2Shamrock.hpp"
#include "shammodels/sph/io/PhantomDump.hpp"
#include <memory>

Register_pymod(pyphantomdump) {

    using T = shammodels::sph::PhantomDump;

    shamcomm::logs::debug_ln("[Py]", "registering shamrock.PhantomDump");
    py::class_<T>(m, "PhantomDump")
        .def(
            "save_dump",
            [](T &self, std::string fname) {
                auto file = self.gen_file();
                file.write_to_file(fname);
            })
        .def(
            "read_header_float",
            [](T &self, std::string s) {
                return self.read_header_float<f64>(s);
            })
        .def(
            "read_header_int",
            [](T &self, std::string s) {
                return self.read_header_int<i64>(s);
            })
        .def("print_state", &shammodels::sph::PhantomDump::print_state);

    m.def("load_phantom_dump", [](std::string fname) {
        shambase::FortranIOFile phfile = shambase::load_fortran_file(fname);
        return shammodels::sph::PhantomDump::from_file(phfile);
    });

    m.def(
        "compare_phantom_dumps",
        [](shammodels::sph::PhantomDump &dump_1, shammodels::sph::PhantomDump &dump_2) {
            return shammodels::sph::compare_phantom_dumps(dump_1, dump_2);
        });
}
