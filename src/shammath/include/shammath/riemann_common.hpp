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
 * @brief Gas and dust conservative/primitive states and axis-transform helpers
 *        shared by every gas and dust Riemann solver
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
        const PrimState<Tvec> prim, typename PrimState<Tvec>::Tscal gamma) {
        ConsState<Tvec> flux;

        const auto rhoeint = prim.press / (gamma - 1.0);
        const auto rhoe    = rhoeint + rhoekin(prim.rho, prim.vel);

        flux.rho = prim.rho * prim.vel[0];

        flux.rhoe = (rhoe + prim.press) * prim.vel[0];

        flux.rhovel[0] = prim.rho * prim.vel[0] * prim.vel[0] + prim.press;
        flux.rhovel[1] = prim.rho * prim.vel[0] * prim.vel[1];
        flux.rhovel[2] = prim.rho * prim.vel[0] * prim.vel[2];

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

    // Axis-transform helpers for PrimState, mirroring y_to_x/z_to_x/invert_axis above.
    // Riemann solvers take primitive states directly (see riemann_hll.hpp etc.), so these
    // are applied to the inputs; the flux they return is a ConsState and is rotated back
    // with the untransformed x_to_y/x_to_z/invert_axis.
    template<class Tprim>
    inline constexpr Tprim prim_y_to_x(const Tprim p) {
        Tprim pprime;
        pprime.rho    = p.rho;
        pprime.press  = p.press;
        pprime.vel[0] = p.vel[1];
        pprime.vel[1] = -p.vel[0];
        pprime.vel[2] = p.vel[2];
        return pprime;
    }

    template<class Tprim>
    inline constexpr Tprim prim_z_to_x(const Tprim p) {
        Tprim pprime;
        pprime.rho    = p.rho;
        pprime.press  = p.press;
        pprime.vel[0] = p.vel[2];
        pprime.vel[1] = p.vel[1];
        pprime.vel[2] = -p.vel[0];
        return pprime;
    }

    template<class Tprim>
    inline constexpr Tprim prim_invert_axis(const Tprim p) {
        Tprim pprime;
        pprime.rho    = p.rho;
        pprime.press  = p.press;
        pprime.vel[0] = -p.vel[0];
        pprime.vel[1] = -p.vel[1];
        pprime.vel[2] = -p.vel[2];
        return pprime;
    }

    template<class Tvec_>
    struct DustConsState {
        using Tvec  = Tvec_;
        using Tscal = shambase::VecComponent<Tvec>;

        Tscal rho{};
        Tvec rhovel{};

        const DustConsState &operator+=(const DustConsState &);
        const DustConsState &operator-=(const DustConsState &);
        const DustConsState &operator*=(const Tscal);
    };

    template<class Tvec_>
    struct DustPrimState {
        using Tvec  = Tvec_;
        using Tscal = shambase::VecComponent<Tvec>;
        Tscal rho{};
        Tvec vel{};
    };

    template<class Tvec>
    const DustConsState<Tvec> &DustConsState<Tvec>::operator+=(const DustConsState<Tvec> &d_cst) {
        rho += d_cst.rho;
        rhovel += d_cst.rhovel;
        return *this;
    }

    template<class Tvec>
    const DustConsState<Tvec> operator+(
        const DustConsState<Tvec> &lhs, const DustConsState<Tvec> &rhs) {
        return DustConsState<Tvec>(lhs) += rhs;
    }

    template<class Tvec>
    const DustConsState<Tvec> &DustConsState<Tvec>::operator-=(const DustConsState<Tvec> &d_cst) {
        rho -= d_cst.rho;
        rhovel -= d_cst.rhovel;
        return *this;
    }

    template<class Tvec>
    const DustConsState<Tvec> operator-(
        const DustConsState<Tvec> &lhs, const DustConsState<Tvec> &rhs) {
        return DustConsState<Tvec>(lhs) -= rhs;
    }

    template<class Tvec>
    const DustConsState<Tvec> &DustConsState<Tvec>::operator*=(
        const typename DustConsState<Tvec>::Tscal factor) {
        rho *= factor;
        rhovel *= factor;
        return *this;
    }

    template<class Tvec>
    const DustConsState<Tvec> operator*(
        const DustConsState<Tvec> &lhs, const typename DustConsState<Tvec>::Tscal factor) {
        return DustConsState<Tvec>(lhs) *= factor;
    }

    template<class Tvec>
    const DustConsState<Tvec> operator*(
        const typename DustConsState<Tvec>::Tscal factor, const DustConsState<Tvec> &rhs) {
        return DustConsState<Tvec>(rhs) *= factor;
    }

    template<class Tvec_>
    struct DustFluxes {
        using Tvec  = Tvec_;
        using Tscal = shambase::VecComponent<Tvec>;
        std::array<DustConsState<Tvec>, 3> F;
    };

    template<class Tvec>
    inline constexpr DustConsState<Tvec> d_prim_to_cons(const DustPrimState<Tvec> d_prim) {
        DustConsState<Tvec> d_cons;
        d_cons.rho    = d_prim.rho;
        d_cons.rhovel = (d_prim.vel * d_prim.rho);
        return d_cons;
    }

    template<class Tvec>
    inline constexpr DustPrimState<Tvec> d_cons_to_prim(const DustConsState<Tvec> d_cons) {
        DustPrimState<Tvec> d_prim;
        d_prim.rho = d_cons.rho;
        d_prim.vel = (d_cons.rhovel * (1 / d_cons.rho));
        return d_prim;
    }

    template<class Tvec>
    inline constexpr DustConsState<Tvec> d_hydro_flux_x(const DustPrimState<Tvec> d_prim) {
        DustConsState<Tvec> d_flux;
        const typename DustPrimState<Tvec>::Tscal x_vel{d_prim.vel[0]};
        d_flux.rho    = d_prim.rho * x_vel;
        d_flux.rhovel = d_prim.vel * (d_prim.rho * x_vel);
        return d_flux;
    }

    template<class Tcons>
    inline constexpr Tcons d_x_to_y(const Tcons c) {
        Tcons d_cst;
        d_cst.rho       = c.rho;
        d_cst.rhovel[0] = -c.rhovel[1];
        d_cst.rhovel[1] = c.rhovel[0];
        d_cst.rhovel[2] = c.rhovel[2];

        return d_cst;
    }

    template<class Tcons>
    inline constexpr Tcons d_y_to_x(const Tcons c) {
        Tcons d_cst;
        d_cst.rho       = c.rho;
        d_cst.rhovel[0] = c.rhovel[1];
        d_cst.rhovel[1] = -c.rhovel[0];
        d_cst.rhovel[2] = c.rhovel[2];
        return d_cst;
    }

    template<class Tcons>
    inline constexpr Tcons d_x_to_z(const Tcons c) {
        Tcons d_cst;
        d_cst.rho       = c.rho;
        d_cst.rhovel[0] = -c.rhovel[2];
        d_cst.rhovel[1] = c.rhovel[1];
        d_cst.rhovel[2] = c.rhovel[0];
        return d_cst;
    }

    template<class Tcons>
    inline constexpr Tcons d_z_to_x(const Tcons c) {
        Tcons d_cst;
        d_cst.rho       = c.rho;
        d_cst.rhovel[0] = c.rhovel[2];
        d_cst.rhovel[1] = c.rhovel[1];
        d_cst.rhovel[2] = -c.rhovel[0];
        return d_cst;
    }

    template<class Tcons>
    inline constexpr Tcons d_invert_axis(const Tcons c) {
        Tcons d_cst;
        d_cst.rho    = c.rho;
        d_cst.rhovel = -(c.rhovel);
        return d_cst;
    }

    // Axis-transform helpers for DustPrimState, mirroring d_y_to_x/d_z_to_x/d_invert_axis
    // above. Dust Riemann solvers take primitive states directly, so these are applied to
    // the inputs; the flux they return is a DustConsState and is rotated back with the
    // untransformed d_x_to_y/d_x_to_z/d_invert_axis.
    template<class Tprim>
    inline constexpr Tprim d_prim_y_to_x(const Tprim p) {
        Tprim pprime;
        pprime.rho    = p.rho;
        pprime.vel[0] = p.vel[1];
        pprime.vel[1] = -p.vel[0];
        pprime.vel[2] = p.vel[2];
        return pprime;
    }

    template<class Tprim>
    inline constexpr Tprim d_prim_z_to_x(const Tprim p) {
        Tprim pprime;
        pprime.rho    = p.rho;
        pprime.vel[0] = p.vel[2];
        pprime.vel[1] = p.vel[1];
        pprime.vel[2] = -p.vel[0];
        return pprime;
    }

    template<class Tprim>
    inline constexpr Tprim d_prim_invert_axis(const Tprim p) {
        Tprim pprime;
        pprime.rho = p.rho;
        pprime.vel = -(p.vel);
        return pprime;
    }

} // namespace shammath
