// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright (c) 2021-2026 Timothée David--Cléris <tim.shamrock@proton.me>
// SPDX-License-Identifier: CeCILL Free Software License Agreement v2.1
// Shamrock is licensed under the CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

#include "shamcomm/logs.hpp"
#include "shammodels/sph/io/PhantomDump.hpp"
#include "shamtest/shamtest.hpp"
#include "tests/ref_files.hpp"

NEW_TEST(Unittest, "phantom-read-write", 1) {

    std::string fname_in  = get_reffile_path("blast_00010");
    std::string fname_out = get_reffile_path("zout_phantom");

    shambase::FortranIOFile phfile = shambase::load_fortran_file(fname_in);

    i32 fortran_byte;

    shammodels::sph::PhantomDump phdump = shammodels::sph::PhantomDump::from_file(phfile);

    logger::raw_ln(phdump.fileid);

    phdump.gen_file().write_to_file(fname_out);

    std::string cmd = "cmp " + fname_in + " " + fname_out;

    int ret = system(cmd.c_str());

    REQUIRE(ret == 0);
}

NEW_TEST(Unittest, "phantom-small-dump-read-write", 1) {

    // Phantom small dumps store the default phantom real in single precision (r1, the real header
    // and the real arrays). Build one from a full dump and check that it is detected & read back.
    std::string fname_in = get_reffile_path("blast_00010");

    shambase::FortranIOFile phfile           = shambase::load_fortran_file(fname_in);
    shammodels::sph::PhantomDump phdump_full = shammodels::sph::PhantomDump::from_file(phfile);

    REQUIRE(!phdump_full.single_prec_real);

    shammodels::sph::PhantomDump phdump_small = phdump_full;
    phdump_small.single_prec_real             = true;
    phdump_small.fileid[0]                    = 'S';

    shambase::FortranIOFile small_file  = phdump_small.gen_file();
    std::basic_string<byte> small_bytes = small_file.get_internal_buf().str();
    u64 small_len                       = small_bytes.size();

    shambase::FortranIOFile small_file_in(std::basic_stringstream<byte>(small_bytes), small_len);
    shammodels::sph::PhantomDump phdump_read
        = shammodels::sph::PhantomDump::from_file(small_file_in);

    REQUIRE(phdump_read.single_prec_real);
    REQUIRE(phdump_read.is_small_dump());
    REQUIRE(small_file_in.finished_read());

    // values must match the full dump up to single precision
    f64 t_full = phdump_full.read_header_float<f64>("time");
    f64 t_read = phdump_read.read_header_float<f64>("time");
    REQUIRE_EQUAL(t_read, f64(f32(t_full)));

    std::vector<f64> x_full, x_read;
    phdump_full.blocks[0].fill_vec("x", x_full);
    phdump_read.blocks[0].fill_vec("x", x_read);
    REQUIRE_EQUAL(x_full.size(), x_read.size());
    bool all_match = true;
    for (u64 i = 0; i < x_full.size(); i++) {
        all_match = all_match && (x_read[i] == f64(f32(x_full[i])));
    }
    REQUIRE(all_match);

    // writing it back must give the exact same bytes
    REQUIRE(phdump_read.gen_file().get_internal_buf().str() == small_bytes);
}
