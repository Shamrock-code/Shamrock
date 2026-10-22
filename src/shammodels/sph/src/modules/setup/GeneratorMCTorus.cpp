// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright (c) 2021-2026 Timothée David--Cléris <tim.shamrock@proton.me>
// SPDX-License-Identifier: CeCILL Free Software License Agreement v2.1
// Shamrock is licensed under the CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

/**
 * @file GeneratorMCTorus.cpp
 * @author Yona Lapeyre (yona.lapeyre@ens-lyon.fr)
 * @brief
 *
 */

#include "shambase/constants.hpp"
#include "shambase/exception.hpp"
#include "sham/format/format.hpp"
#include "shamalgs/primitives/mock_value.hpp"
#include "shamcomm/logs.hpp"
#include "shamcomm/worldInfo.hpp"
#include "shammath/unimodal_max.hpp"
#include "shammodels/sph/math/density.hpp"
#include "shammodels/sph/modules/setup/GeneratorMCTorus.hpp"
#include <array>

template<class Tvec, template<class> class SPHKernel>
auto shammodels::sph::modules::GeneratorMCTorus<Tvec, SPHKernel>::TorusIterator::compute_fmax()
    -> Tscal {
    // rho_shape is maximal on the midplane (W decreases with |z|), and
    // r_cyl * rho_shape(r_cyl, 0) vanishes at both edges with a single max in between
    return shammath::unimodal_max_value<Tscal>(
        [&](Tscal r_cyl) {
            return f_func(r_cyl, 0);
        },
        torus.get_r_in(),
        torus.get_r_out());
}

template<class Tvec, template<class> class SPHKernel>
auto shammodels::sph::modules::GeneratorMCTorus<Tvec, SPHKernel>::TorusIterator::next(u64 seed)
    -> TorusOutput {

    std::mt19937_64 eng_local(seed); // ensure that 1 part = 1 random draw

    using shamalgs::primitives::mock_value;

    Tscal r_in  = torus.get_r_in();
    Tscal r_out = torus.get_r_out();
    Tscal z_max = torus.get_z_max();

    // rejection sampling of r_cyl * rho in (r_cyl, z)
    auto find_r_cyl_z = [&]() {
        while (true) {
            Tscal u     = mock_value<Tscal>(eng_local, 0, fmax);
            Tscal r_cyl = mock_value<Tscal>(eng_local, r_in, r_out);
            Tscal z     = mock_value<Tscal>(eng_local, -z_max, z_max);
            Tscal W     = torus.enthalpy(r_cyl, z);
            Tscal f     = r_cyl * torus.rho_shape_from_enthalpy(W);
            // fmax is only converged to a relative 1e-8, so allow for slight overshoots
            if (f > fmax * Tscal(1 + 1e-6)) {
                throw shambase::make_except_with_loc<std::runtime_error>(sham::format(
                    "GeneratorMCTorus: r_cyl * rho_shape = {} at (r_cyl, z) = ({}, {}) exceeds "
                    "the rejection sampling bound {}",
                    f,
                    r_cyl,
                    z,
                    fmax));
            }
            if (u < f) {
                return std::array<Tscal, 3>{r_cyl, z, W};
            }
        }
    };

    auto phi = mock_value<Tscal>(eng_local, 0, 2 * shambase::constants::pi<Tscal>);

    // depends on the torus density
    auto [r_cyl, z, W] = find_r_cyl_z();

    Tscal cos_phi = sycl::cos(phi);
    Tscal sin_phi = sycl::sin(phi);

    return TorusOutput{
        .pos     = sycl::vec<Tscal, 3>{r_cyl * cos_phi, r_cyl * sin_phi, z},
        .r_cyl   = r_cyl,
        .cos_phi = cos_phi,
        .sin_phi = sin_phi,
        .W       = W,
        // density of the torus equilibrium
        .rho = rho0 * torus.rho_shape_from_enthalpy(W)};
}

template<class Tvec, template<class> class SPHKernel>
shammodels::sph::modules::GeneratorMCTorus<Tvec, SPHKernel>::GeneratorMCTorus(
    ShamrockCtx &context,
    Config &solver_config,
    shamphys::PolytropicTorus torus,
    Tscal part_mass,
    Tscal torus_mass,
    std::optional<Tscal> plasma_beta,
    Tscal A_cut,
    std::mt19937_64 eng,
    Tscal init_h_factor)
    : context(context), solver_config(solver_config), pmass(part_mass),
      generator(make_generator(torus, part_mass, torus_mass, eng)), init_h_factor(init_h_factor),
      A_cut(A_cut) {

    using Adiabatic = typename Config::EOSConfig::Adiabatic;

    if (const Adiabatic *eos = std::get_if<Adiabatic>(&solver_config.eos_config.config)) {
        gamma = eos->gamma;
    } else if (!solver_config.is_eos_locally_isothermal()) {
        throw shambase::make_except_with_loc<std::invalid_argument>(
            "GeneratorMCTorus: only the adiabatic and locally isothermal EOS are supported, "
            "the torus pressure can not be set with the current EOS");
    }

    Tscal rho0 = generator.get_rho0();

    if (plasma_beta) {
        if (!solver_config.has_field_b_on_rho()) {
            throw shambase::make_except_with_loc<std::invalid_argument>(
                "GeneratorMCTorus: a plasma_beta was given but the solver has no magnetic "
                "field, enable MHD in the solver config first");
        }
        if (!(*plasma_beta > 0)) {
            throw shambase::make_except_with_loc<std::invalid_argument>(
                sham::format("GeneratorMCTorus: plasma_beta must be > 0, got {}", *plasma_beta));
        }

        Tscal I_B = torus.b_poloidal_unit_sq_integral(A_cut);
        if (!(I_B > 0)) {
            throw shambase::make_except_with_loc<std::invalid_argument>(sham::format(
                "GeneratorMCTorus: no magnetic field with A_cut = {}, it must be in [0, 1[",
                A_cut));
        }

        // beta = rho0 int P~ dV / (A0^2 int b~^2 dV / (2 mu_0))
        Tscal mu_0 = solver_config.get_constant_mu_0();
        A0         = sycl::sqrt(2 * mu_0 * rho0 * torus.pressure_integral() / (*plasma_beta * I_B));
    }

    if (shamcomm::world_rank() == 0) {
        logger::info_ln(
            "GeneratorMCTorus",
            sham::format(
                "r_in = {}, r0 = {}, r_out = {}, z_max = {}, r_cusp = {}, rho0 = {}",
                torus.get_r_in(),
                torus.get_r0(),
                torus.get_r_out(),
                torus.get_z_max(),
                torus.get_r_cusp(),
                rho0));
    }
}

template<class Tvec, template<class> class SPHKernel>
bool shammodels::sph::modules::GeneratorMCTorus<Tvec, SPHKernel>::is_done() {
    return generator.is_done();
}

template<class Tvec, template<class> class SPHKernel>
shamrock::patch::PatchDataLayer shammodels::sph::modules::GeneratorMCTorus<Tvec, SPHKernel>::next_n(
    u32 nmax) {

    using namespace shamrock::patch;
    PatchScheduler &sched = shambase::get_check_ref(context.sched);

    std::vector<TorusOutput> pos_data;

    if (!generator.is_done()) {
        pos_data = generator.next_n(nmax);
    }

    const shamphys::PolytropicTorus &torus = generator.get_torus();
    Tscal rho0                             = generator.get_rho0();

    u32 len = pos_data.size();

    bool has_soundspeed = solver_config.is_eos_locally_isothermal();
    bool has_u          = solver_config.is_eos_adiabatic();
    bool has_B          = solver_config.has_field_b_on_rho();

    std::vector<Tvec> vec_pos(len), vec_vel(len), vec_B_on_rho;
    std::vector<Tscal> vec_h(len), vec_u, vec_cs;
    if (has_u) {
        vec_u.resize(len);
    }
    if (has_soundspeed) {
        vec_cs.resize(len);
    }
    if (has_B) {
        vec_B_on_rho.resize(len);
    }

    // the smoothing length is only an initial guess, the floor avoids giant particles at the
    // surface where rho -> 0
    Tscal rho_floor = Tscal(1e-3) * rho0;

    for (u32 i = 0; i < len; i++) {
        const TorusOutput &o = pos_data[i];
        Tvec pos             = o.pos;
        Tscal rho            = o.rho;
        Tscal r_cyl = o.r_cyl, z = pos.z();
        Tscal cos_p = o.cos_phi, sin_p = o.sin_phi;

        Tscal p_on_rho = torus.p_over_rho_from_enthalpy(o.W);
        Tscal v_phi    = torus.v_phi(r_cyl);

        vec_pos[i] = pos;
        vec_vel[i] = Tvec{-v_phi * sin_p, v_phi * cos_p, 0};
        vec_h[i]   = shamrock::sph::h_rho(pmass, sycl::fmax(rho, rho_floor), Kernel::hfactd)
                     * init_h_factor;

        if (has_u) {
            vec_u[i] = p_on_rho / (*gamma - 1);
        }
        if (has_soundspeed) {
            // isothermal sound speed, so that P = cs^2 rho matches the torus pressure
            vec_cs[i] = sycl::sqrt(p_on_rho);
        }
        if (has_B) {
            auto [B_r_cyl, B_z] = torus.b_poloidal_unit_from_enthalpy(r_cyl, z, o.W, A_cut);
            // no plasma_beta given: the field is left at 0
            Tscal fact      = (rho > 0) ? A0.value_or(0) / rho : 0;
            vec_B_on_rho[i] = Tvec{fact * B_r_cyl * cos_p, fact * B_r_cyl * sin_p, fact * B_z};
        }
    }

    PatchDataLayer tmp(sched.get_layout_ptr_old());
    if (len == 0) {
        return tmp;
    }

    tmp.resize(len);
    tmp.fields_raz();

    auto set_field = [&](const char *name, auto &vec) {
        using T              = typename std::remove_reference_t<decltype(vec)>::value_type;
        PatchDataField<T> &f = tmp.get_field<T>(sched.pdl_old().get_field_idx<T>(name));
        sycl::buffer<T> buf(vec.data(), len);
        f.override(buf, len);
    };

    set_field("xyz", vec_pos);
    set_field("vxyz", vec_vel);
    set_field("hpart", vec_h);
    if (has_u) {
        set_field("uint", vec_u);
    }
    if (has_soundspeed) {
        set_field("soundspeed", vec_cs);
    }
    if (has_B) {
        set_field("B/rho", vec_B_on_rho);
    }

    return tmp;
}

using namespace shammath;
template class shammodels::sph::modules::GeneratorMCTorus<f64_3, M4>;
template class shammodels::sph::modules::GeneratorMCTorus<f64_3, M6>;
template class shammodels::sph::modules::GeneratorMCTorus<f64_3, M8>;

template class shammodels::sph::modules::GeneratorMCTorus<f64_3, C2>;
template class shammodels::sph::modules::GeneratorMCTorus<f64_3, C4>;
template class shammodels::sph::modules::GeneratorMCTorus<f64_3, C6>;
