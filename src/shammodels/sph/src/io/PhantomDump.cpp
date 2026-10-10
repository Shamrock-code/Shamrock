// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright (c) 2021-2026 Timothée David--Cléris <tim.shamrock@proton.me>
// SPDX-License-Identifier: CeCILL Free Software License Agreement v2.1
// Shamrock is licensed under the CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

/**
 * @file PhantomDump.cpp
 * @author Timothée David--Cléris (tim.shamrock@proton.me)
 * @brief
 *
 * \todo clean classes name to make it more readable
 *
 */

#include "shambase/aliases_int.hpp"
#include "shambase/exception.hpp"
#include "shambase/sets.hpp"
#include "shambase/stacktrace.hpp"
#include "shambase/string.hpp"
#include "shambackends/typeAliasVec.hpp"
#include "shamcomm/logs.hpp"
#include "shammodels/common/EOSConfig.hpp"
#include "shammodels/sph/config/AVConfig.hpp"
#include "shammodels/sph/io/PhantomDump.hpp"
#include "shamsys/legacy/log.hpp"
#include "shamunits/UnitSystem.hpp"
#include <type_traits>
#include <unordered_map>
#include <string>
#include <vector>

namespace {

    /**
     * @brief Read an array of values, optionally stored as 32-bit floats in the file
     *
     * Phantom small dumps store the default `real` in single precision, this converts them back
     * to the in-memory type `T`.
     */
    template<class T>
    void read_vals(
        shambase::FortranIOFile &phfile, std::vector<T> &vals, u32 count, bool single_prec) {
        if (!single_prec) {
            phfile.read_val_array(vals, count);
            return;
        }
        if constexpr (std::is_floating_point_v<T>) {
            std::vector<f32> tmp;
            phfile.read_val_array(tmp, count);
            vals.assign(tmp.begin(), tmp.end());
        } else {
            shambase::throw_with_loc<std::invalid_argument>(
                "single precision storage is only valid for floating point values");
        }
    }

    /// Write an array of values, optionally converting them to 32-bit floats (phantom small dumps)
    template<class T>
    void write_vals(
        shambase::FortranIOFile &phfile, std::vector<T> &vals, u32 count, bool single_prec) {
        if (!single_prec) {
            phfile.write_val_array(vals, count);
            return;
        }
        if constexpr (std::is_floating_point_v<T>) {
            std::vector<f32> tmp(vals.begin(), vals.end());
            phfile.write_val_array(tmp, count);
        } else {
            shambase::throw_with_loc<std::invalid_argument>(
                "single precision storage is only valid for floating point values");
        }
    }

} // namespace

template<class T>
shammodels::sph::PhantomDumpBlockArray<T> shammodels::sph::PhantomDumpBlockArray<T>::from_file(
    shambase::FortranIOFile &phfile, i64 tot_count, bool single_prec) {
    StackEntry stack_loc{};
    PhantomDumpBlockArray tmp;
    phfile.read_fixed_string(tmp.tag, 16);
    read_vals(phfile, tmp.vals, tot_count, single_prec);
    return tmp;
}

template<class T>
void shammodels::sph::PhantomDumpBlockArray<T>::write(
    shambase::FortranIOFile &phfile, i64 tot_count, bool single_prec) {
    StackEntry stack_loc{};
    phfile.write_fixed_string(tag, 16);
    write_vals(phfile, vals, tot_count, single_prec);
}

template<class T>
void shammodels::sph::PhantomDumpBlockArray<T>::print_state() {
    logger::raw_ln("tag =", tag, "size =", vals.size());
}

template<class T>
shammodels::sph::PhantomDumpTableHeader<T> shammodels::sph::PhantomDumpTableHeader<T>::from_file(
    shambase::FortranIOFile &phfile, bool single_prec) {
    StackEntry stack_loc{};

    shammodels::sph::PhantomDumpTableHeader<T> tmp;

    int nvars;

    phfile.read(nvars);

    if (nvars == 0) {
        return tmp;
    }

    std::vector<std::string> tags;
    phfile.read_string_array(tags, 16, nvars);

    std::vector<T> vals;
    read_vals(phfile, vals, nvars, single_prec);

    for (u32 i = 0; i < nvars; i++) {
        tmp.entries.push_back({tags[i], vals[i]});
    }

    return tmp;
}

template<class T>
void shammodels::sph::PhantomDumpTableHeader<T>::write(
    shambase::FortranIOFile &phfile, bool single_prec) {
    StackEntry stack_loc{};

    int nvars = entries.size();
    phfile.write(nvars);

    if (nvars == 0) {
        return;
    }

    std::vector<std::string> tags;
    std::vector<T> vals;
    for (u32 i = 0; i < nvars; i++) {
        auto [a, b] = entries[i];
        tags.push_back(a);
        vals.push_back(b);
    }

    phfile.write_string_array(tags, 16, nvars);
    write_vals(phfile, vals, nvars, single_prec);
}

template<class T>
void shammodels::sph::PhantomDumpTableHeader<T>::print_state() {

    for (auto [key, val] : entries) {
        logger::raw_ln(key, val);
    }
}

void shammodels::sph::PhantomDumpBlock::print_state() {

    logger::raw_ln("--blocks_fort_int --");
    for (auto b : blocks_fort_int) {
        b.print_state();
    }
    logger::raw_ln("--blocks_i8       --");
    for (auto b : blocks_i8) {
        b.print_state();
    }
    logger::raw_ln("--blocks_i16      --");
    for (auto b : blocks_i16) {
        b.print_state();
    }
    logger::raw_ln("--blocks_i32      --");
    for (auto b : blocks_i32) {
        b.print_state();
    }
    logger::raw_ln("--blocks_i64      --");
    for (auto b : blocks_i64) {
        b.print_state();
    }
    logger::raw_ln("--blocks_fort_real--");
    for (auto b : blocks_fort_real) {
        b.print_state();
    }
    logger::raw_ln("--blocks_f32      --");
    for (auto b : blocks_f32) {
        b.print_state();
    }
    logger::raw_ln("--blocks_f64      --");
    for (auto b : blocks_f64) {
        b.print_state();
    }
}

shammodels::sph::PhantomDumpBlock shammodels::sph::PhantomDumpBlock::from_file(
    shambase::FortranIOFile &phfile,
    i64 tot_count,
    std::array<i32, 8> numarray,
    bool single_prec_real) {
    PhantomDumpBlock block;

    block.tot_count = tot_count;

    for (u32 j = 0; j < numarray[0]; j++) {
        block.blocks_fort_int.push_back(
            PhantomDumpBlockArray<fort_int>::from_file(phfile, block.tot_count));
    }
    for (u32 j = 0; j < numarray[1]; j++) {
        block.blocks_i8.push_back(PhantomDumpBlockArray<i8>::from_file(phfile, block.tot_count));
    }
    for (u32 j = 0; j < numarray[2]; j++) {
        block.blocks_i16.push_back(PhantomDumpBlockArray<i16>::from_file(phfile, block.tot_count));
    }
    for (u32 j = 0; j < numarray[3]; j++) {
        block.blocks_i32.push_back(PhantomDumpBlockArray<i32>::from_file(phfile, block.tot_count));
    }
    for (u32 j = 0; j < numarray[4]; j++) {
        block.blocks_i64.push_back(PhantomDumpBlockArray<i64>::from_file(phfile, block.tot_count));
    }
    for (u32 j = 0; j < numarray[5]; j++) {
        block.blocks_fort_real.push_back(
            PhantomDumpBlockArray<fort_real>::from_file(phfile, block.tot_count, single_prec_real));
    }
    for (u32 j = 0; j < numarray[6]; j++) {
        block.blocks_f32.push_back(PhantomDumpBlockArray<f32>::from_file(phfile, block.tot_count));
    }
    for (u32 j = 0; j < numarray[7]; j++) {
        block.blocks_f64.push_back(PhantomDumpBlockArray<f64>::from_file(phfile, block.tot_count));
    }

    return block;
}

void shammodels::sph::PhantomDumpBlock::write(
    shambase::FortranIOFile &phfile,
    i64 tot_count,
    std::array<i32, 8> numarray,
    bool single_prec_real) {
    StackEntry stack_loc{};

    for (u32 j = 0; j < numarray[0]; j++) {
        blocks_fort_int[j].write(phfile, tot_count);
    }
    for (u32 j = 0; j < numarray[1]; j++) {
        blocks_i8[j].write(phfile, tot_count);
    }
    for (u32 j = 0; j < numarray[2]; j++) {
        blocks_i16[j].write(phfile, tot_count);
    }
    for (u32 j = 0; j < numarray[3]; j++) {
        blocks_i32[j].write(phfile, tot_count);
    }
    for (u32 j = 0; j < numarray[4]; j++) {
        blocks_i64[j].write(phfile, tot_count);
    }
    for (u32 j = 0; j < numarray[5]; j++) {
        blocks_fort_real[j].write(phfile, tot_count, single_prec_real);
    }
    for (u32 j = 0; j < numarray[6]; j++) {
        blocks_f32[j].write(phfile, tot_count);
    }
    for (u32 j = 0; j < numarray[7]; j++) {
        blocks_f64[j].write(phfile, tot_count);
    }
}

u64 shammodels::sph::PhantomDumpBlock::get_ref_fort_real(std::string s) {

    s            = sham::format("{:16s}", s);
    auto &blocks = blocks_fort_real;

    for (u32 i = 0; i < blocks_fort_real.size(); i++) {
        if (blocks_fort_real[i].tag == s) {
            return i;
        }
    }

    PhantomDumpBlockArray<fort_real> tmp;
    tmp.tag = s;
    blocks_fort_real.push_back(std::move(tmp));

    for (u32 i = 0; i < blocks_fort_real.size(); i++) {
        if (blocks_fort_real[i].tag == s) {
            return i;
        }
    }

    return 0;
}

u64 shammodels::sph::PhantomDumpBlock::get_ref_f32(std::string s) {

    s = sham::format("{:16s}", s);

    auto &blocks = blocks_f32;

    for (u32 i = 0; i < blocks_f32.size(); i++) {
        if (blocks_f32[i].tag == s) {
            return i;
        }
    }

    PhantomDumpBlockArray<f32> tmp;
    tmp.tag = s;
    blocks_f32.push_back(std::move(tmp));

    for (u32 i = 0; i < blocks_f32.size(); i++) {
        if (blocks_f32[i].tag == s) {
            return i;
        }
    }

    return 0;
}

shambase::FortranIOFile shammodels::sph::PhantomDump::gen_file() {
    StackEntry stack_loc{};

    shambase::FortranIOFile phfile;
    if (single_prec_real) {
        f32 r1_single = r1;
        phfile.write(i1, r1_single, i2, iversion, i3);
    } else {
        phfile.write(i1, r1, i2, iversion, i3);
    }

    phfile.write_fixed_string(fileid, 100);

    table_header_fort_int.write(phfile);
    table_header_i8.write(phfile);
    table_header_i16.write(phfile);
    table_header_i32.write(phfile);
    table_header_i64.write(phfile);
    table_header_fort_real.write(phfile, single_prec_real);
    table_header_f32.write(phfile);
    table_header_f64.write(phfile);

    int nblocks = blocks.size();
    phfile.write(nblocks);

    std::vector<i64> block_tot_counts;
    std::vector<std::array<i32, 8>> block_numarray;
    for (u32 i = 0; i < nblocks; i++) {

        i64 tot_count = blocks[i].tot_count;
        std::array<i32, 8> counts
            = {i32(blocks[i].blocks_fort_int.size()),
               i32(blocks[i].blocks_i8.size()),
               i32(blocks[i].blocks_i16.size()),
               i32(blocks[i].blocks_i32.size()),
               i32(blocks[i].blocks_i64.size()),
               i32(blocks[i].blocks_fort_real.size()),
               i32(blocks[i].blocks_f32.size()),
               i32(blocks[i].blocks_f64.size())};

        phfile.write(tot_count, counts);
        block_tot_counts.push_back(tot_count);
        block_numarray.push_back(counts);
    }

    for (u32 i = 0; i < nblocks; i++) {
        blocks[i].write(phfile, block_tot_counts[i], block_numarray[i], single_prec_real);
    }

    return phfile;
}

shammodels::sph::PhantomDump shammodels::sph::PhantomDump::from_file(
    shambase::FortranIOFile &phfile) {
    PhantomDump phdump;

    // first line
    //<4 bytes>i1,r1,i2,iversion,i3<4 bytes>
    // r1 is a default phantom real, which is written in single precision in small dumps.
    // The length of this record therefore tells us how the default reals are stored.
    constexpr i32 len_first_line_double = 4 * sizeof(fort_int) + sizeof(f64);
    constexpr i32 len_first_line_single = 4 * sizeof(fort_int) + sizeof(f32);

    i32 len_first_line = phfile.peek_record_length();
    if (len_first_line == len_first_line_double) {
        phdump.single_prec_real = false;
        phfile.read(phdump.i1, phdump.r1, phdump.i2, phdump.iversion, phdump.i3);
    } else if (len_first_line == len_first_line_single) {
        phdump.single_prec_real = true;
        f32 r1_single;
        phfile.read(phdump.i1, r1_single, phdump.i2, phdump.iversion, phdump.i3);
        phdump.r1 = r1_single;
    } else {
        shambase::throw_with_loc<std::runtime_error>(sham::format(
            "unexpected length of the first record of the phantom dump: {} bytes (expected {} "
            "for double precision reals or {} for single precision reals)",
            len_first_line,
            len_first_line_double,
            len_first_line_single));
    }
    phdump.check_magic_numbers();

    // The second line contains a 100-character file identifier:
    // <4 bytes>fileid<4 bytes>
    phfile.read_fixed_string(phdump.fileid, 100);

    // loop i=1,8
    //    <4 bytes>nvars<4 bytes>
    //    <4 bytes>tags(1:nvars)<4 bytes>
    //    <4 bytes>vals(1:nvals)<4 bytes>
    // end loop
    phdump.table_header_fort_int = PhantomDumpTableHeader<fort_int>::from_file(phfile);
    phdump.table_header_i8       = PhantomDumpTableHeader<i8>::from_file(phfile);
    phdump.table_header_i16      = PhantomDumpTableHeader<i16>::from_file(phfile);
    phdump.table_header_i32      = PhantomDumpTableHeader<i32>::from_file(phfile);
    phdump.table_header_i64      = PhantomDumpTableHeader<i64>::from_file(phfile);
    phdump.table_header_fort_real
        = PhantomDumpTableHeader<fort_real>::from_file(phfile, phdump.single_prec_real);
    phdump.table_header_f32 = PhantomDumpTableHeader<f32>::from_file(phfile);
    phdump.table_header_f64 = PhantomDumpTableHeader<f64>::from_file(phfile);

    int nblocks;
    phfile.read(nblocks);

    std::vector<i64> block_tot_counts;
    std::vector<std::array<i32, 8>> block_numarray;

    for (u32 i = 0; i < nblocks; i++) {

        i64 tot_count;
        std::array<i32, 8> counts;

        phfile.read(tot_count, counts);

        block_tot_counts.push_back(tot_count);
        block_numarray.push_back(counts);
    }
    for (u32 i = 0; i < nblocks; i++) {
        phdump.blocks.push_back(
            PhantomDumpBlock::from_file(
                phfile, block_tot_counts[i], block_numarray[i], phdump.single_prec_real));
    }

    if (!phfile.finished_read()) {
        logger::warn_ln("[PhantomReader]", "some data was not read");
    }

    return phdump;
}

void shammodels::sph::PhantomDump::print_state() {
    logger::raw_ln("--- dump state ---");

    logger::raw_ln("table_header_fort_int len  =", table_header_fort_int.entries.size());
    table_header_fort_int.print_state();
    logger::raw_ln("table_header_i8 len        =", table_header_i8.entries.size());
    table_header_i8.print_state();
    logger::raw_ln("table_header_i16 len       =", table_header_i16.entries.size());
    table_header_i16.print_state();
    logger::raw_ln("table_header_i32 len       =", table_header_i32.entries.size());
    table_header_i32.print_state();
    logger::raw_ln("table_header_i64 len       =", table_header_i64.entries.size());
    table_header_i64.print_state();
    logger::raw_ln("table_header_fort_real len =", table_header_fort_real.entries.size());
    table_header_fort_real.print_state();
    logger::raw_ln("table_header_f32 len       =", table_header_f32.entries.size());
    table_header_f32.print_state();
    logger::raw_ln("table_header_f64 len       =", table_header_f64.entries.size());
    table_header_f64.print_state();

    for (u32 i = 0; i < blocks.size(); i++) {
        logger::raw_ln("block ", i, ":");
        blocks[i].print_state();
    }
    logger::raw_ln("------------------");
}

bool shammodels::sph::compare_phantom_dumps(PhantomDump &dump_ref, PhantomDump &dump_comp) {

    u64 offenses = 0;

    auto load_header = [](PhantomDump &dump) {
        std::unordered_map<std::string, f64> header;

        dump.table_header_fort_int.add_to_map(header);
        dump.table_header_i8.add_to_map(header);
        dump.table_header_i16.add_to_map(header);
        dump.table_header_i32.add_to_map(header);
        dump.table_header_i64.add_to_map(header);
        dump.table_header_fort_real.add_to_map(header);
        dump.table_header_f32.add_to_map(header);
        dump.table_header_f64.add_to_map(header);
        return header;
    };

    std::unordered_map<std::string, f64> header_ref  = load_header(dump_ref);
    std::unordered_map<std::string, f64> header_comp = load_header(dump_comp);

    auto get_keys = [](std::unordered_map<std::string, f64> &map) {
        std::set<std::string> ret;
        for (auto [key, val] : map) {
            ret.insert(key);
        }
        return ret;
    };

    std::set<std::string> header_key_ref  = get_keys(header_ref);
    std::set<std::string> header_key_comp = get_keys(header_comp);

    std::vector<std::string> missing  = {};
    std::vector<std::string> matching = {};
    std::vector<std::string> extra    = {};

    shambase::set_diff(header_key_comp, header_key_ref, missing, matching, extra);

    for (std::string &missing_key : missing) {
        logger::warn_ln(
            "PhantomDump",
            "The dump we are comparing against is missing the key",
            missing_key,
            "in the header");
        offenses++;
    }
    for (std::string &matching_key : matching) {
        if (header_ref[matching_key] != header_comp[matching_key]) {
            logger::warn_ln(
                "PhantomDump",
                sham::format(
                    "Mismatch in the header key {}, ref={}, comp={}",
                    matching_key,
                    header_ref[matching_key],
                    header_comp[matching_key]));
            offenses++;
        }
    }
    for (std::string &extra_key : extra) {
        logger::warn_ln(
            "PhantomDump",
            "The dump we are comparing against has the extra key",
            extra_key,
            "in the header");
        offenses++;
    }

    logger::info_ln("PhantomDump", "This comparison reported", offenses, "offenses");

    return offenses == 0;
}
