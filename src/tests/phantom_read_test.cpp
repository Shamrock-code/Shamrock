// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright(C) 2021-2023 Timothée David--Cléris <timothee.david--cleris@ens-lyon.fr>
// Licensed under CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

#include "shammodels/sph/io/PhantomDump.hpp"
#include "shamtest/shamtest.hpp"
#include "tests/ref_files.hpp"

TestStart(Unittest, "phantom-read-write", pahntomread, 1) {

    std::string fname_in  = get_reffile_path("blast_00010");
    std::string fname_out = get_reffile_path("zout_phantom");

    shambase::FortranIOFile phfile = shambase::load_fortran_file(fname_in);

    i32 fortran_byte;

    shammodels::sph::PhantomDump phdump = shammodels::sph::PhantomDump::from_file(phfile);

    logger::raw_ln(phdump.fileid);

    phdump.gen_file().write_to_file(fname_out);

    std::string cmd = "cmp " + fname_in + " " + fname_out;

    int ret = system(cmd.c_str());

    _Assert(ret == 0);
}
