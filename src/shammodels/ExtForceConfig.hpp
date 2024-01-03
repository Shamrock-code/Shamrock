// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright(C) 2021-2023 Timothée David--Cléris <timothee.david--cleris@ens-lyon.fr>
// Licensed under CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

#pragma once

/**
 * @file ExtForceConfig.hpp
 * @author Timothée David--Cléris (timothee.david--cleris@ens-lyon.fr)
 * @brief
 *
 */

#include "shambackends/math.hpp"
#include "shambase/exception.hpp"
#include "shambase/sycl_utils/vectorProperties.hpp"
#include "shamsys/legacy/log.hpp"
#include <string>
#include <type_traits>
#include <variant>

namespace shammodels {

    template<class Tvec>
    struct ExtForceConfig {

        using Tscal              = shambase::VecComponent<Tvec>;
        static constexpr u32 dim = shambase::VectorProperties<Tvec>::dimension;

        struct PointMass {
            Tscal central_mass;
            Tscal Racc;
        };

        struct LenseThirring {
            Tscal central_mass;
            Tscal Racc;
            Tscal a_spin;
            Tvec dir_spin;
        };

        /**
         * \brief Shearing box forces as in athena
         * \cite Stone2010_shear_box
         * \f[
         *  \mathbf{f} = 2\Omega_0 \left(  q \Omega_0 x +  v_y \right) \basevec{x} -2\Omega_0 v_x \basevec{y} - \Omega_0^2 z \basevec{z}  \f]
         * Shear speed :
         * \f[
         *  \omega = q \Omega_0 L_x \f]
         */
        struct ShearingBoxForce {
            i32_3 shear_base = {1,0,0};
            i32_3 shear_dir = {0,1,0};

            Tscal Omega_0;
            Tscal eta;
            Tscal q;

            inline Tscal shear_speed(Tscal box_lenght){
                return q*Omega_0*box_lenght;
            }

            ShearingBoxForce()= default;
            ShearingBoxForce(Tscal Omega_0,Tscal eta,Tscal q) : Omega_0(Omega_0), eta(eta), q(q){};

        };

        using VariantForce = std::variant<PointMass, LenseThirring, ShearingBoxForce>;

        std::vector<VariantForce> ext_forces;

        inline void add_point_mass(Tscal central_mass, Tscal Racc) {
            ext_forces.push_back(PointMass{central_mass, Racc});
        }

        inline void
        add_lense_thirring(Tscal central_mass, Tscal Racc, Tscal a_spin, Tvec dir_spin) {
            if (sham::abs(sycl::length(dir_spin) - 1) > 1e-8) {
                shambase::throw_with_loc<std::invalid_argument>(
                    "the sping direction should be a unit vector");
            }
            ext_forces.push_back(LenseThirring{central_mass, Racc, a_spin, dir_spin});
        }

        /**
         * @brief
         * \todo add check for norm of shear vectors
         */
        inline void add_shearing_box(
            Tscal Omega_0,Tscal eta,Tscal q) {

            ext_forces.push_back(
                ShearingBoxForce{Omega_0, eta, q});
        }
    };

} // namespace shammodels