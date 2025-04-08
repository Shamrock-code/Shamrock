// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright (c) 2021-2024 Timothée David--Cléris <tim.shamrock@proton.me>
// SPDX-License-Identifier: CeCILL Free Software License Agreement v2.1
// Shamrock is licensed under the CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

/**
 * @file FluxDivergence.cpp
 * @author Timothée David--Cléris (timothee.david--cleris@ens-lyon.fr)
 * @brief
 *
 */

#include "shambase/stacktrace.hpp"
#include "shambackends/sycl.hpp"
#include "shammodels/ramses/modules/ComputeFlux.hpp"
#include "shammodels/ramses/modules/ComputeFluxUtilities.hpp"
#include "shammodels/ramses/modules/FluxDivergence.hpp"
#include "shamrock/scheduler/SchedulerUtility.hpp"

template<class T>
using NGLink = shammodels::basegodunov::modules::NeighGraphLinkField<T>;

template<class Tvec, class TgridVec>
void shammodels::basegodunov::modules::FluxDivergence<Tvec, TgridVec>::
    eval_flux_divergence_hydro_fields() {

    StackEntry stack_loc{};

    // const to prim
    modules::ConsToPrim ctop(context, solver_config, storage);
    ctop.cons_to_prim();
    if (solver_config.is_dust_on())
        ctop.cons_to_prim_dust();

    // compute & limit gradients /* maybe compute fields slopes is a better name ?*/
    modules::Slopes slopes(context, solver_config, storage);
    slopes.slope_rho();
    slopes.slope_v();
    slopes.slope_P();
    if (solver_config.is_dust_on()) {
        slopes.slope_rho_dust();
        slopes.slope_v_dust();
    }

    // shift values /* Field reconstruction (and interpolation for muscl-hancock ) ?*/
    modules::FaceInterpolate face_interpolator(
        context, solver_config, storage); /** change the class name to reconstruction is more
                                             general than interpolation ?*/
    bool is_muscl = solver_config.is_muscl_scheme();
    face_interpolator.interpolate_rho_to_face(dt_input, is_muscl);
    face_interpolator.interpolate_v_to_face(dt_input, is_muscl);
    face_interpolator.interpolate_P_to_face(dt_input, is_muscl);

    if (solver_config.is_dust_on()) {
        face_interpolator.interpolate_rho_dust_to_face(dt_input, is_muscl);
        face_interpolator.interpolate_v_dust_to_face(dt_input, is_muscl);
    }

    // flux at cell faces
    modules::ComputeFlux flux_compute(context, solver_config, storage);
    flux_compute.compute_flux();
    if (solver_config.is_dust_on()) {
        flux_compute.compute_flux_dust();
    }
    // compute dt fields or  flux divergence operator evaluation
    modules::ComputeTimeDerivative dt_compute(context, solver_config, storage);
    dt_compute.compute_dt_fields();
    if (solver_config.is_dust_on()) {
        dt_compute.compute_dt_dust_fields();
    }
}

template<class Tvec, class TgridVec>
void shammodels::basegodunov::modules::FluxDivergence<Tvec, TgridVec>::
    eval_flux_divergence_dust_fields() {

    StackEntry stack_loc{};

    // const to prim
    modules::ConsToPrim ctop(context, solver_config, storage);
    ctop.cons_to_prim_dust();

    // compute & limit gradients /* maybe compute fields slopes is a better name ?*/
    modules::Slopes slopes(context, solver_config, storage);
    slopes.slope_rho_dust();
    slopes.slope_v_dust();

    // shift values /* Field reconstruction (and interpolation for muscl-hancock ) ?*/
    modules::FaceInterpolate face_interpolator(
        context, solver_config, storage); /** change the class name to reconstruction is more
                                             general than interpolation ?*/
    bool is_muscl = solver_config.is_muscl_scheme();
    face_interpolator.interpolate_rho_dust_to_face(dt_input, is_muscl);
    face_interpolator.interpolate_v_dust_to_face(dt_input, is_muscl);

    // flux at cell faces
    modules::ComputeFlux flux_compute(context, solver_config, storage);
    flux_compute.compute_flux_dust();

    // compute dt fields or  flux divergence operator evaluation
    modules::ComputeTimeDerivative dt_compute(context, solver_config, storage);
    dt_compute.compute_dt_dust_fields();
}

template<class Tvec, class TgridVec>
void shammodels::basegodunov::modules::FluxDivergence<Tvec, TgridVec>::
    reset_storage_buffers_hydro() {
    StackEntry stack_loc{};

    storage.vel.reset();
    storage.press.reset();

    storage.grad_rho.reset();
    storage.dx_v.reset();
    storage.dy_v.reset();
    storage.dz_v.reset();
    storage.grad_P.reset();

    storage.rho_face_xp.reset();
    storage.rho_face_xm.reset();
    storage.rho_face_yp.reset();
    storage.rho_face_ym.reset();
    storage.rho_face_zp.reset();
    storage.rho_face_zm.reset();

    storage.vel_face_xp.reset();
    storage.vel_face_xm.reset();
    storage.vel_face_yp.reset();
    storage.vel_face_ym.reset();
    storage.vel_face_zp.reset();
    storage.vel_face_zm.reset();

    storage.press_face_xp.reset();
    storage.press_face_xm.reset();
    storage.press_face_yp.reset();
    storage.press_face_ym.reset();
    storage.press_face_zp.reset();
    storage.press_face_zm.reset();

    storage.flux_rho_face_xp.reset();
    storage.flux_rho_face_xm.reset();
    storage.flux_rho_face_yp.reset();
    storage.flux_rho_face_ym.reset();
    storage.flux_rho_face_zp.reset();
    storage.flux_rho_face_zm.reset();
    storage.flux_rhov_face_xp.reset();
    storage.flux_rhov_face_xm.reset();
    storage.flux_rhov_face_yp.reset();
    storage.flux_rhov_face_ym.reset();
    storage.flux_rhov_face_zp.reset();
    storage.flux_rhov_face_zm.reset();
    storage.flux_rhoe_face_xp.reset();
    storage.flux_rhoe_face_xm.reset();
    storage.flux_rhoe_face_yp.reset();
    storage.flux_rhoe_face_ym.reset();
    storage.flux_rhoe_face_zp.reset();
    storage.flux_rhoe_face_zm.reset();

    storage.dtrho.reset();
    storage.dtrhov.reset();
    storage.dtrhoe.reset();
}

template<class Tvec, class TgridVec>
void shammodels::basegodunov::modules::FluxDivergence<Tvec, TgridVec>::
    reset_storage_buffers_dust() {
    StackEntry stack_loc{};

    storage.vel_dust.reset();

    storage.grad_rho_dust.reset();
    storage.dx_v_dust.reset();
    storage.dy_v_dust.reset();
    storage.dz_v_dust.reset();

    storage.rho_dust_face_xm.reset();
    storage.rho_dust_face_yp.reset();
    storage.rho_dust_face_ym.reset();
    storage.rho_dust_face_xp.reset();
    storage.rho_dust_face_zp.reset();
    storage.rho_dust_face_zm.reset();

    storage.vel_dust_face_xp.reset();
    storage.vel_dust_face_xm.reset();
    storage.vel_dust_face_yp.reset();
    storage.vel_dust_face_ym.reset();
    storage.vel_dust_face_zp.reset();
    storage.vel_dust_face_zm.reset();

    storage.flux_rho_dust_face_xp.reset();
    storage.flux_rho_dust_face_xm.reset();
    storage.flux_rho_dust_face_yp.reset();
    storage.flux_rho_dust_face_ym.reset();
    storage.flux_rho_dust_face_zp.reset();
    storage.flux_rho_dust_face_zm.reset();
    storage.flux_rhov_dust_face_xp.reset();
    storage.flux_rhov_dust_face_xm.reset();
    storage.flux_rhov_dust_face_yp.reset();
    storage.flux_rhov_dust_face_ym.reset();
    storage.flux_rhov_dust_face_zp.reset();
    storage.flux_rhov_dust_face_zm.reset();

    storage.dtrho_dust.reset();
    storage.dtrhov_dust.reset();
}

template class shammodels::basegodunov::modules::FluxDivergence<f64_3, i64_3>;
