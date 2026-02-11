// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright (c) 2021-2026 Timothée David--Cléris <tim.shamrock@proton.me>
// SPDX-License-Identifier: CeCILL Free Software License Agreement v2.1
// Shamrock is licensed under the CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

/**
 * @file PhantomDumpEOSUtils.cpp
 * @author Timothée David--Cléris (tim.shamrock@proton.me)
 * @author Yona Lapeyre (yona.lapeyre@ens-lyon.fr)
 * @brief
 *
 */

#include "shambase/exception.hpp"
#include "shambase/string.hpp"
#include "shambackends/sycl.hpp"
#include "shamcomm/logs.hpp"
#include "shamcomm/worldInfo.hpp"
#include "shammodels/sph/io/PhantomDump.hpp"
#include <stdexcept>
#include <string>

namespace {

    /// The EOS config of phantom
    struct EOSPhConfig {
        i32 isink;
        f64 gamma = 1;
        f64 polyk;
        f64 polyk2;
        f64 qfacdisc  = 0.75;
        f64 qfacdisc2 = 0.75;

        // if (ieos == 7) {
        int istrat;
        f64 alpha_z;
        f64 beta_z;
        f64 z0;
        //}
    };

    /// Converted directly from Phantom eos.f90
    void write_headeropts_eos(int ieos, shammodels::sph::PhantomDump &hdr, EOSPhConfig &eos) {
        hdr.table_header_i32.add("isink", eos.isink);
        hdr.table_header_fort_real.add("gamma", eos.gamma);
        hdr.table_header_fort_real.add("RK2", 1.5 * eos.polyk);
        hdr.table_header_fort_real.add("polyk2", eos.polyk2);
        hdr.table_header_fort_real.add("qfacdisc", eos.qfacdisc);
        hdr.table_header_fort_real.add("qfacdisc2", eos.qfacdisc2);

        if (ieos == 7) {
            hdr.table_header_i32.add("istrat", eos.istrat);
            hdr.table_header_fort_real.add("alpha_z", eos.alpha_z);
            hdr.table_header_fort_real.add("beta_z", eos.beta_z);
            hdr.table_header_fort_real.add("z0", eos.z0);
        }
    }

    /// Converted directly from Phantom eos.f90
    EOSPhConfig read_headeropts_eos(const shammodels::sph::PhantomDump &hdr, int ieos) {
        EOSPhConfig eos;

        f64 RK2;

        eos.gamma = hdr.read_header_float<f64>("gamma");
        RK2       = hdr.read_header_float<f64>("RK2");
        eos.polyk = 2.0 / 3.0 * RK2;

        bool use_krome = false; // How do i get this one from a dump Daniel ...
        int maxvxyzu   = (eos.gamma != 1.) ? 4 : 3;

        if (shamcomm::world_rank() == 0) {
            if (maxvxyzu >= 4) {
                if (use_krome) {
                    logger::raw_ln("KROME eos: initial gamma = 1.666667");
                } else {
                    logger::raw_ln(shambase::format("adiabatic eos: gamma = {}", eos.gamma));
                }
            } else {
                logger::raw_ln(
                    shambase::format(
                        "setting isothermal sound speed^2 (polyk) = {}, gamma = {}",
                        eos.polyk,
                        eos.gamma));
                if (eos.polyk <= std::numeric_limits<f64>::epsilon()) {
                    logger::raw_ln(
                        shambase::format(
                            "WARNING! sound speed zero in dump!, polyk = {}", eos.polyk));
                }
            }
        }

        eos.polyk2    = hdr.read_header_float<f64>("polyk2");
        eos.qfacdisc  = hdr.read_header_float<f64>("qfacdisc");
        eos.qfacdisc2 = hdr.read_header_float<f64>("qfacdisc2");
        eos.isink     = hdr.read_header_int<int>("isink");

        if (std::abs(eos.gamma - 1.0) > std::numeric_limits<f64>::epsilon() && maxvxyzu < 4) {
            logger::raw_ln(
                shambase::format(
                    "WARNING! compiled for isothermal equation of state but gamma /= 1, gamma={}",
                    eos.gamma));
        }

        if (ieos == 3 || ieos == 6 || ieos == 7) {
            if (eos.qfacdisc <= std::numeric_limits<f64>::epsilon()) {
                if (shamcomm::world_rank() == 0)
                    logger::raw_ln(shambase::format("ERROR: qfacdisc <= 0"));
            } else {
                if (shamcomm::world_rank() == 0)
                    logger::raw_ln(shambase::format("qfacdisc = {}", eos.qfacdisc));
            }
        }

        if (ieos == 7) {
            eos.istrat  = hdr.read_header_int<int>("istrat");
            eos.alpha_z = hdr.read_header_float<f64>("alpha_z");
            eos.beta_z  = hdr.read_header_float<f64>("beta_z");
            eos.z0      = hdr.read_header_float<f64>("z0");
            if (std::abs(eos.qfacdisc2) <= std::numeric_limits<f64>::epsilon()) {
                if (shamcomm::world_rank() == 0)
                    logger::raw_ln(shambase::format("ERROR: qfacdisc2 == 0"));
            } else {
                if (shamcomm::world_rank() == 0)
                    logger::raw_ln(shambase::format("qfacdisc2 = {}", eos.qfacdisc2));
            }
        }

        return eos;
    }

} // namespace

namespace shammodels::sph::phdump {

    bool is_maxvxyzu_at_least_4(const PhantomDump &dump) { return dump.has_header_entry("alphau"); }

    /// Check that the eos in the dump is the expected one
    inline void assert_ieos_val(const PhantomDump &dump, int ieos) {
        i64 ieos_dump = dump.read_header_int<i64>("ieos");
        if (ieos_dump != ieos) {
            shambase::throw_with_loc<std::invalid_argument>(shambase::format(
                "You are querying phantom dump eos {} parameters, even though ieos is {}",
                ieos,
                ieos_dump));
        }
    }

    /*
    * EOS 1
    !
    !--Isothermal eos
    !
    !  :math:`P = c_s^2 \rho`
    !
    !  where :math:`c_s^2 \equiv K` is a constant stored in the dump file header
    !
    */

    void eos1_load(const PhantomDump &dump, f64 &cs) {
        assert_ieos_val(dump, 1);
        EOSPhConfig eos = read_headeropts_eos(dump, 1);

        cs = sycl::sqrt(eos.polyk);
    }

    void eos1_write(PhantomDump &dump, const f64 &cs) {
        EOSPhConfig eos;

        eos.polyk = cs * cs;

        dump.table_header_i32.add("ieos", 1);
        write_headeropts_eos(1, dump, eos);
    }

    /*
    case(2,5,17)
    !
    !--Adiabatic equation of state (code default)
    !
    !  :math:`P = (\gamma - 1) \rho u`
    !
    !  if the code is compiled with ISOTHERMAL=yes, ieos=2 gives a polytropic eos:
    !
    !  :math:`P = K \rho^\gamma`
    !
    !  where K is a global constant specified in the dump header
    !

    For now I will support only 2
    */
    void eos2_load(const PhantomDump &dump, f64 &gamma) {
        assert_ieos_val(dump, 2);
        EOSPhConfig eos = read_headeropts_eos(dump, 2);

        gamma = eos.gamma;
    }

    void eos2_write(PhantomDump &dump, const f64 &gamma) {
        EOSPhConfig eos;

        eos.gamma = gamma;

        dump.table_header_i32.add("ieos", 2);
        write_headeropts_eos(2, dump, eos);
    }

    /*
    case(3)
    !
    !--Locally isothermal disc as in Lodato & Pringle (2007) where
    !
    !  :math:`P = c_s^2 (r) \rho`
    !
    !  sound speed (temperature) is prescribed as a function of radius using:
    !
    !  :math:`c_s = c_{s,0} r^{-q}` where :math:`r = \sqrt{x^2 + y^2 + z^2}`
    !

    ponrhoi  = polyk*(xi**2 + yi**2 + zi**2)**(-qfacdisc) ! polyk is cs^2, so this is (R^2)^(-q)
    spsoundi = sqrt(ponrhoi)
    tempi    = temperature_coef*mui*ponrhoi
    */

    void eos3_load(const PhantomDump &dump, f64 &cs0, f64 &q, f64 &r0) {
        assert_ieos_val(dump, 3);
        EOSPhConfig eos = read_headeropts_eos(dump, 3);

        cs0 = sycl::sqrt(eos.polyk);
        q   = eos.qfacdisc;
        r0  = 1; // the polyk in phantom include the 1/r0^2 ?
    }

    void eos3_write(PhantomDump &dump, const f64 &cs0, const f64 &q, const f64 &r0) {
        EOSPhConfig eos;

        eos.polyk    = cs0 * cs0 / (r0 * r0);
        eos.qfacdisc = q;

        dump.table_header_i32.add("ieos", 3);
        write_headeropts_eos(3, dump, eos);
    }

    /*
     case(13)
    !
    !--Locally isothermal eos for generic hierarchical system
    !
    !  Assuming all sink particles are stars.
    !  Generalisation of Farris et al. (2014; for binaries) to N stars.
    !  For two sink particles this is identical to ieos=14
    !
    */

    void eos13_load(const PhantomDump &dump, f64 &cs0, f64 &q, f64 &r0) {
        assert_ieos_val(dump, 13);
        EOSPhConfig eos = read_headeropts_eos(dump, 13);

        cs0 = sycl::sqrt(eos.polyk);
        q   = eos.qfacdisc;
        r0  = 1; // the polyk in phantom include the 1/r0^2 ?
    }

    void eos13_write(PhantomDump &dump, const f64 &cs0, const f64 &q, const f64 &r0) {
        EOSPhConfig eos;

        eos.polyk    = cs0 * cs0 / (r0 * r0);
        eos.qfacdisc = q;

        dump.table_header_i32.add("ieos", 13);
        write_headeropts_eos(13, dump, eos);
    }

    /*
    case(14)
    !
    !--Locally isothermal eos from Farris et al. (2014) for binary system
    !
    !  uses the locations of the first two sink particles
    !
    */

    void eos14_load(const PhantomDump &dump, f64 &cs0, f64 &q, f64 &r0) {
        assert_ieos_val(dump, 14);
        EOSPhConfig eos = read_headeropts_eos(dump, 14);

        cs0 = sycl::sqrt(eos.polyk);
        q   = eos.qfacdisc;
        r0  = 1; // the polyk in phantom include the 1/r0^2 ?
    }

    void eos14_write(PhantomDump &dump, const f64 &cs0, const f64 &q, const f64 &r0) {
        EOSPhConfig eos;

        eos.polyk    = cs0 * cs0 / (r0 * r0);
        eos.qfacdisc = q;

        dump.table_header_i32.add("ieos", 14);
        write_headeropts_eos(14, dump, eos);
    }
} // namespace shammodels::sph::phdump
