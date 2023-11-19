// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright(C) 2021-2023 Timothée David--Cléris <timothee.david--cleris@ens-lyon.fr>
// Licensed under CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

#include "shammodels/sph/io/PhantomDump.hpp"
#include "shamtest/shamtest.hpp"


TestStart(Unittest, "phantom-read-write", pahntomread, 1) {

    std::string fname = "reference-files/blast_00010";

    shambase::FortranIOFile phfile = shambase::load_fortran_file(fname);

    i32 fortran_byte;

    shammodels::sph::PhantomDump phdump = shammodels::sph::PhantomDump::from_file(phfile);

    logger::raw_ln(phdump.fileid);

    phdump.gen_file().write_to_file("reference-files/zout_phantom");

    int ret = system("cmp reference-files/blast_00010 reference-files/zout_phantom");

    _Assert(ret == 0);
}