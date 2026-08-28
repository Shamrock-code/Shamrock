// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright (c) 2021-2026 Timothée David--Cléris <tim.shamrock@proton.me>
// SPDX-License-Identifier: CeCILL Free Software License Agreement v2.1
// Shamrock is licensed under the CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

#pragma once

/**
 * @file riemann_common.hpp
 * @author Léodasce Sewanou (leodasce.sewanou@ens-lyon.fr) --no git blame--
 * @author Thomas Guillet (T.A.Guillet@exeter.ac.uk) --no git blame--
 * @author Timothée David--Cléris (tim.shamrock@proton.me)
 * @brief Gas conservative/primitive states and axis-transform helpers shared
 *        by every gas Riemann solver
 * From original version by Thomas Guillet (T.A.Guillet@exeter.ac.uk)
 */

#include "shambackends/math.hpp"
#include "shambackends/typeAliasVec.hpp"
#include "shambackends/vec.hpp"
#include <algorithm>
#include <array>
#include <cmath>
#include <iostream>
namespace shammath {

    template<class Tvec_>
    struct ConsState {
        using Tvec  = Tvec_;
        using Tscal = shambase::VecComponent<Tvec>;

        Tscal rho{}, rhoe{};
        Tvec rhovel{};

        const ConsState &operator+=(const ConsState &);
        const ConsState &operator-=(const ConsState &);
        const ConsState &operator*=(const Tscal);
    };

    template<class Tvec_>
    struct PrimState {
        using Tvec  = Tvec_;
        using Tscal = shambase::VecComponent<Tvec>;

        Tscal rho{}, press{};
        Tvec vel{};
    };

    template<class Tvec>
    const ConsState<Tvec> &ConsState<Tvec>::operator+=(const ConsState<Tvec> &cst) {
        rho += cst.rho;
        rhoe += cst.rhoe;
        rhovel += cst.rhovel;
        return *this;
    }

    template<class Tvec>
    const ConsState<Tvec> operator+(const ConsState<Tvec> &lhs, const ConsState<Tvec> &rhs) {
        return ConsState<Tvec>(lhs) += rhs;
    }

    template<class Tvec>
    const ConsState<Tvec> &ConsState<Tvec>::operator-=(const ConsState<Tvec> &cst) {
        rho -= cst.rho;
        rhoe -= cst.rhoe;
        rhovel -= cst.rhovel;
        return *this;
    }

    template<class Tvec>
    const ConsState<Tvec> operator-(const ConsState<Tvec> &lhs, const ConsState<Tvec> &rhs) {
        return ConsState<Tvec>(lhs) -= rhs;
    }

    template<class Tvec>
    const ConsState<Tvec> &ConsState<Tvec>::operator*=(
        const typename ConsState<Tvec>::Tscal factor) {
        rho *= factor;
        rhoe *= factor;
        rhovel *= factor;
        return *this;
    }

    template<class Tvec>
    const ConsState<Tvec> operator*(
        const typename ConsState<Tvec>::Tscal factor, const ConsState<Tvec> &rhs) {
        return ConsState<Tvec>(rhs) *= factor;
    }

    template<class Tvec>
    const ConsState<Tvec> operator*(
        const ConsState<Tvec> &lhs, const typename ConsState<Tvec>::Tscal factor) {
        return ConsState<Tvec>(lhs) *= factor;
    }

    template<class Tvec_>
    struct Fluxes {
        using Tvec  = Tvec_;
        using Tscal = shambase::VecComponent<Tvec>;

        std::array<ConsState<Tvec>, 3> F;
    };

    template<class Tvec>
    inline constexpr shambase::VecComponent<Tvec> rhoekin(
        shambase::VecComponent<Tvec> rho, Tvec v) {
        using Tscal    = shambase::VecComponent<Tvec>;
        const Tscal v2 = v[0] * v[0] + v[1] * v[1] + v[2] * v[2];
        return 0.5 * rho * v2;
    }

    template<class Tvec>
    inline constexpr ConsState<Tvec> prim_to_cons(
        const PrimState<Tvec> prim, typename PrimState<Tvec>::Tscal gamma) {
        ConsState<Tvec> cons;

        cons.rho = prim.rho;

        const auto rhoeint = prim.press / (gamma - 1.0);
        cons.rhoe          = rhoeint + rhoekin(prim.rho, prim.vel);

        cons.rhovel[0] = prim.rho * prim.vel[0];
        cons.rhovel[1] = prim.rho * prim.vel[1];
        cons.rhovel[2] = prim.rho * prim.vel[2];

        return cons;
    }

    template<class Tvec>
    inline constexpr PrimState<Tvec> cons_to_prim(
        const ConsState<Tvec> cons, typename ConsState<Tvec>::Tscal gamma) {
        PrimState<Tvec> prim;

        prim.rho = cons.rho;

        prim.vel[0] = cons.rhovel[0] / cons.rho;
        prim.vel[1] = cons.rhovel[1] / cons.rho;
        prim.vel[2] = cons.rhovel[2] / cons.rho;

        const auto rhoeint = cons.rhoe - rhoekin(prim.rho, prim.vel);
        prim.press         = (gamma - 1.0) * rhoeint;

        return prim;
    }

    template<class Tvec>
    inline constexpr ConsState<Tvec> hydro_flux_x(
        const ConsState<Tvec> cons, typename ConsState<Tvec>::Tscal gamma) {
        ConsState<Tvec> flux;

        const PrimState<Tvec> prim = cons_to_prim(cons, gamma);

        flux.rho = cons.rhovel[0];

        flux.rhoe = (cons.rhoe + prim.press) * prim.vel[0];

        flux.rhovel[0] = cons.rho * prim.vel[0] * prim.vel[0] + prim.press;
        flux.rhovel[1] = cons.rho * prim.vel[0] * prim.vel[1];
        flux.rhovel[2] = cons.rho * prim.vel[0] * prim.vel[2];

        return flux;
    }

    template<class Tvec>
    inline constexpr shambase::VecComponent<Tvec> sound_speed(
        PrimState<Tvec> prim, shambase::VecComponent<Tvec> gamma) {
        return sycl::sqrt(gamma * prim.press / prim.rho);
    }

    template<class Tcons>
    inline constexpr Tcons y_to_x(const Tcons c) {
        Tcons cprime;
        cprime.rho       = c.rho;
        cprime.rhoe      = c.rhoe;
        cprime.rhovel[0] = c.rhovel[1];
        cprime.rhovel[1] = -c.rhovel[0];
        cprime.rhovel[2] = c.rhovel[2];
        return cprime;
    }

    template<class Tcons>
    inline constexpr Tcons x_to_y(const Tcons c) {
        Tcons cprime;
        cprime.rho       = c.rho;
        cprime.rhoe      = c.rhoe;
        cprime.rhovel[0] = -c.rhovel[1];
        cprime.rhovel[1] = c.rhovel[0];
        cprime.rhovel[2] = c.rhovel[2];
        return cprime;
    }

    template<class Tcons>
    inline constexpr Tcons z_to_x(const Tcons c) {
        Tcons cprime;
        cprime.rho       = c.rho;
        cprime.rhoe      = c.rhoe;
        cprime.rhovel[0] = c.rhovel[2];
        cprime.rhovel[1] = c.rhovel[1];
        cprime.rhovel[2] = -c.rhovel[0];
        return cprime;
    }

    template<class Tcons>
    inline constexpr Tcons x_to_z(const Tcons c) {
        Tcons cprime;
        cprime.rho       = c.rho;
        cprime.rhoe      = c.rhoe;
        cprime.rhovel[0] = -c.rhovel[2];
        cprime.rhovel[1] = c.rhovel[1];
        cprime.rhovel[2] = c.rhovel[0];
        return cprime;
    }

    template<class Tcons>
    inline constexpr Tcons invert_axis(const Tcons c) {
        Tcons cprime;
        cprime.rho       = c.rho;
        cprime.rhoe      = c.rhoe;
        cprime.rhovel[0] = -c.rhovel[0];
        cprime.rhovel[1] = -c.rhovel[1];
        cprime.rhovel[2] = -c.rhovel[2];
        return cprime;
    }

} // namespace shammath
