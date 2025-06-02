// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright (c) 2021-2024 Timothée David--Cléris <tim.shamrock@proton.me>
// SPDX-License-Identifier: CeCILL Free Software License Agreement v2.1
// Shamrock is licensed under the CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

/**
 * @file Phantom2Shamrock.cpp
 * @author Timothée David--Cléris (timothee.david--cleris@ens-lyon.fr)
 * @brief
 *
 */

#include "shambase/constants.hpp"
#include "shammodels/sph/config/BCConfig.hpp"
#include "shammodels/sph/io/Phantom2Shamrock.hpp"
#include "shammodels/sph/io/PhantomDump.hpp"
#include "shammodels/sph/io/PhantomDumpEOSUtils.hpp"

namespace shammodels::sph {

    template<class Tvec>
    EOSConfig<Tvec> get_shamrock_eosconfig(PhantomDump &phdump, bool bypass_error) {

        EOSConfig<Tvec> cfg{};

        i64 ieos = phdump.read_header_int<i64>("ieos");

        logger::debug_ln("PhantomDump", "read ieos :", ieos);

        if (ieos == 1) {
            f64 cs;
            phdump::eos1_load(phdump, cs);
            cfg.set_isothermal(cs);
        } else if (ieos == 2) {
            f64 gamma;
            phdump::eos2_load(phdump, gamma);
            cfg.set_adiabatic(gamma);
        } else if (ieos == 3) {
            f64 cs0, q, r0;
            phdump::eos3_load(phdump, cs0, q, r0);
            cfg.set_locally_isothermalLP07(cs0, q, r0);
        } else {
            const std::string msg = "loading phantom ieos=" + std::to_string(ieos)
                                    + " is not implemented in shamrock";
            if (bypass_error) {
                logger::warn_ln("SPH", msg);
            } else {
                shambase::throw_unimplemented(msg);
            }
        }

        return cfg;
    }

    template<class Tvec>
    void
    write_shamrock_eos_in_phantom_dump(EOSConfig<Tvec> &cfg, PhantomDump &dump, bool bypass_error) {

        using EOS_Isothermal              = typename EOSConfig<Tvec>::Isothermal;
        using EOS_Adiabatic               = typename EOSConfig<Tvec>::Adiabatic;
        using EOS_LocallyIsothermal       = typename EOSConfig<Tvec>::LocallyIsothermal;
        using EOS_LocallyIsothermalLP07   = typename EOSConfig<Tvec>::LocallyIsothermalLP07;
        using EOS_LocallyIsothermalFA2014 = typename EOSConfig<Tvec>::LocallyIsothermalFA2014;

        if (EOS_Isothermal *eos_config = std::get_if<EOS_Isothermal>(&cfg.config)) {
            phdump::eos1_write(dump, eos_config->cs);
        } else if (EOS_Adiabatic *eos_config = std::get_if<EOS_Adiabatic>(&cfg.config)) {
            phdump::eos2_write(dump, eos_config->gamma);
        } else if (
            EOS_LocallyIsothermalLP07 *eos_config
            = std::get_if<EOS_LocallyIsothermalLP07>(&cfg.config)) {
            phdump::eos3_write(dump, eos_config->cs0, eos_config->q, eos_config->r0);
        } else {
            const std::string msg
                = "The current shamrock EOS is not implemented in phantom dump conversion";
            if (bypass_error) {
                logger::warn_ln("SPH", msg);
            } else {
                shambase::throw_unimplemented(msg);
            }
        }
    }

    /// explicit instanciation for f32_3
    template EOSConfig<f32_3> get_shamrock_eosconfig<f32_3>(PhantomDump &phdump, bool bypass_error);
    /// explicit instanciation for f64_3
    template EOSConfig<f64_3> get_shamrock_eosconfig<f64_3>(PhantomDump &phdump, bool bypass_error);

    /// explicit instanciation for f32_3
    template void write_shamrock_eos_in_phantom_dump<f32_3>(
        EOSConfig<f32_3> &cfg, PhantomDump &dump, bool bypass_error);
    /// explicit instanciation for f64_3
    template void write_shamrock_eos_in_phantom_dump<f64_3>(
        EOSConfig<f64_3> &cfg, PhantomDump &dump, bool bypass_error);

} // namespace shammodels::sph

namespace shammodels::sph {
    template<class Tvec>
    AVConfig<Tvec> get_shamrock_avconfig(PhantomDump &phdump) {
        AVConfig<Tvec> cfg{};

        cfg.set_varying_cd10(0, 1, 0.1, phdump.read_header_float<f64>("alphau"), 2);

        return cfg;
    }

    /// explicit instanciation for f32_3
    template AVConfig<f32_3> get_shamrock_avconfig<f32_3>(PhantomDump &phdump);
    /// explicit instanciation for f64_3
    template AVConfig<f64_3> get_shamrock_avconfig<f64_3>(PhantomDump &phdump);

    template<class Tscal>
    shamunits::UnitSystem<Tscal> get_shamrock_units(PhantomDump &phdump) {

        f64 udist  = phdump.read_header_float<f64>("udist");
        f64 umass  = phdump.read_header_float<f64>("umass");
        f64 utime  = phdump.read_header_float<f64>("utime");
        f64 umagfd = phdump.read_header_float<f64>("umagfd");

        return shamunits::UnitSystem<Tscal>(
            utime, udist, umass
            // unit_current = 1 ,
            // unit_temperature = 1 ,
            // unit_qte = 1 ,
            // unit_lumint = 1
        );
    }

    template<class Tscal>
    void write_shamrock_units_in_phantom_dump(
        std::optional<shamunits::UnitSystem<Tscal>> &units, PhantomDump &dump, bool bypass_error) {

        if (units) {
            dump.table_header_f64.add("udist", units->m_inv);
            dump.table_header_f64.add("umass", units->kg_inv);
            dump.table_header_f64.add("utime", units->s_inv);

            f64 umass = units->template to<shamunits::units::kg>();
            f64 utime = units->template to<shamunits::units::s>();
            f64 udist = units->template to<shamunits::units::m>();

            shamunits::Constants<Tscal> ctes{*units};
            f64 ccst    = ctes.c();
            f64 ucharge = sqrt(umass * udist / (4. * shambase::constants::pi<f64> /*mu_0 in cgs*/));

            f64 umagfd = umass / (utime * ucharge);

            dump.table_header_f64.add("umagfd", umagfd);
        } else {
            logger::warn_ln("SPH", "no units are set, defaulting to SI");

            dump.table_header_f64.add("udist", 1);
            dump.table_header_f64.add("umass", 1);
            dump.table_header_f64.add("utime", 1);
            dump.table_header_f64.add("umagfd", 3.54491);
        }
    }

    /// explicit instanciation for f32_3
    template shamunits::UnitSystem<f32> get_shamrock_units<f32>(PhantomDump &phdump);
    /// explicit instanciation for f64_3
    template shamunits::UnitSystem<f64> get_shamrock_units<f64>(PhantomDump &phdump);

    template void write_shamrock_units_in_phantom_dump<f64>(
        std::optional<shamunits::UnitSystem<f64>> &units, PhantomDump &dump, bool bypass_error);
} // namespace shammodels::sph

namespace shammodels::sph {
    template<class Tvec>
    BCConfig<Tvec> get_shamrock_boundary_config(PhantomDump &phdump) {
        BCConfig<Tvec> cfg;
        // xmin, xmax, y... z... are in the header only in periodic mode in phantom
        if (phdump.has_header_entry("xmin")) {
            logger::raw_ln("Setting periodic boundaries from phantmdump");
            cfg.set_periodic();
        } else {
            logger::raw_ln("Setting free boundaries from phantmdump");
            cfg.set_free();
        }
        return cfg;
    }

    template<class Tvec>
    void write_shamrock_boundaries_in_phantom_dump(
        BCConfig<Tvec> &cfg,
        std::tuple<Tvec, Tvec> box_size,
        PhantomDump &dump,
        bool bypass_error) {

        auto [bmin, bmax]              = box_size;
        using SolverConfigBC           = BCConfig<Tvec>;
        using SolverBCFree             = typename SolverConfigBC::Free;
        using SolverBCPeriodic         = typename SolverConfigBC::Periodic;
        using SolverBCShearingPeriodic = typename SolverConfigBC::ShearingPeriodic;

        // boundary condition selections
        if (SolverBCFree *c = std::get_if<SolverBCFree>(&cfg.config)) {
            // do nothing
        } else if (SolverBCPeriodic *c = std::get_if<SolverBCPeriodic>(&cfg.config)) {
            dump.table_header_fort_real.add("xmin", bmin.x());
            dump.table_header_fort_real.add("xmax", bmax.x());
            dump.table_header_fort_real.add("ymin", bmin.y());
            dump.table_header_fort_real.add("ymax", bmax.x());
            dump.table_header_fort_real.add("zmin", bmin.z());
            dump.table_header_fort_real.add("zmax", bmax.x());
        } else if (
            SolverBCShearingPeriodic *c = std::get_if<SolverBCShearingPeriodic>(&cfg.config)) {
            std::string err_msg
                = "Phantom does not support shearing periodic boundaries but your are "
                  "making a phantom dump with them, set bypass_error_check=True to ignore";

            if (!bypass_error) {
                throw std::runtime_error(err_msg);
            } else {
                logger::warn_ln("PhantomDump", err_msg);
            }
        }
    }

    /// explicit instanciation for f32_3
    template BCConfig<f32_3> get_shamrock_boundary_config<f32_3>(PhantomDump &phdump);
    /// explicit instanciation for f64_3
    template BCConfig<f64_3> get_shamrock_boundary_config<f64_3>(PhantomDump &phdump);

    /// explicit instanciation for f32_3
    template void write_shamrock_boundaries_in_phantom_dump<f32_3>(
        BCConfig<f32_3> &cfg,
        std::tuple<f32_3, f32_3> box_size,
        PhantomDump &dump,
        bool bypass_error);
    /// explicit instanciation for f64_3
    template void write_shamrock_boundaries_in_phantom_dump<f64_3>(
        BCConfig<f64_3> &cfg,
        std::tuple<f64_3, f64_3> box_size,
        PhantomDump &dump,
        bool bypass_error);
} // namespace shammodels::sph
