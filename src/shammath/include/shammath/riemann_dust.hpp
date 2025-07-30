// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright (c) 2021-2025 Timothée David--Cléris <tim.shamrock@proton.me>
// SPDX-License-Identifier: CeCILL Free Software License Agreement v2.1
// Shamrock is licensed under the CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

#pragma once

/**
 * @file riemann_dust.hpp
 * @author Léodasce Sewanou (leodasce.sewanou@ens-lyon.fr)
 * @author Timothée David--Cléris (tim.shamrock@proton.me)
 * @brief This file contain states and Riemann solvers for dust
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
    const DustConsState<Tvec>
    operator+(const DustConsState<Tvec> &lhs, const DustConsState<Tvec> &rhs) {
        return DustConsState<Tvec>(lhs) += rhs;
    }

    template<class Tvec>
    const DustConsState<Tvec> &DustConsState<Tvec>::operator-=(const DustConsState<Tvec> &d_cst) {
        rho -= d_cst.rho;
        rhovel -= d_cst.rhovel;
        return *this;
    }

    template<class Tvec>
    const DustConsState<Tvec>
    operator-(const DustConsState<Tvec> &lhs, const DustConsState<Tvec> &rhs) {
        return DustConsState<Tvec>(lhs) -= rhs;
    }

    template<class Tvec>
    const DustConsState<Tvec> &
    DustConsState<Tvec>::operator*=(const typename DustConsState<Tvec>::Tscal factor) {
        rho *= factor;
        rhovel *= factor;
        return *this;
    }

    template<class Tvec>
    const DustConsState<Tvec>
    operator*(const DustConsState<Tvec> &lhs, const typename DustConsState<Tvec>::Tscal factor) {
        return DustConsState<Tvec>(lhs) *= factor;
    }

    template<class Tvec>
    const DustConsState<Tvec>
    operator*(const typename DustConsState<Tvec>::Tscal factor, const DustConsState<Tvec> &rhs) {
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
    inline constexpr DustConsState<Tvec> d_hydro_flux_x(const DustConsState<Tvec> d_cons) {
        DustConsState<Tvec> d_flux;
        const DustPrimState<Tvec> d_prim = d_cons_to_prim<Tvec>(d_cons);
        const typename DustConsState<Tvec>::Tscal x_vel{d_prim.vel[0]};
        d_flux.rho    = d_cons.rhovel[0];
        d_flux.rhovel = d_prim.vel * (d_cons.rho * x_vel);
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

    // Krapp et al. 2025, A Fast second-order solver for stiff multifluid dust and gas hydrodynamics
    // Appendice E
    template<class Tcons>
    inline constexpr auto d_hll_flux_x(Tcons cL, Tcons cR) {
        Tcons d_flux;
        const auto d_primL = d_cons_to_prim(cL);
        const auto d_primR = d_cons_to_prim(cR);

        const auto S = sham::max(sham::abs(d_primL.vel[0]), sham::abs(d_primR.vel[0]));

        const auto fL = d_hydro_flux_x(cL);
        const auto fR = d_hydro_flux_x(cR);

        return 0.5 * ((fL + fR) - S * (cR - cL));
    }

    // Huang & Bai, 2022 ,A Multiﬂuid Dust Module in Athena++: Algorithms and Numerical Tests
    // Equation (32)
    template<class Tcons>
    inline constexpr auto huang_bai_flux_x(Tcons cL, Tcons cR) {
        Tcons d_flux;
        const auto d_primL = d_cons_to_prim(cL);
        const auto d_primR = d_cons_to_prim(cR);

        const auto fL = d_hydro_flux_x(cL);
        const auto fR = d_hydro_flux_x(cR);

        if (d_primL.vel[0] > 0 && d_primR.vel[0] > 0)
            d_flux = fL;
        else if (d_primL.vel[0] < 0 && d_primR.vel[0] < 0)
            d_flux = fR;
        else if (d_primL.vel[0] < 0 && d_primR.vel[0] > 0)
            d_flux *= 0;
        else if (d_primL.vel[0] > 0 && d_primR.vel[0] < 0)
            d_flux = (fL + fR);

        return d_flux;
    }

    template<class Tcons>
    inline constexpr Tcons d_hll_flux_y(Tcons cL, Tcons cR) {
        return d_x_to_y(d_hll_flux_x(d_y_to_x(cL), d_y_to_x(cR)));
    }

    template<class Tcons>
    inline constexpr Tcons d_hll_flux_z(Tcons cL, Tcons cR) {
        return d_x_to_z(d_hll_flux_x(d_z_to_x(cL), d_z_to_x(cR)));
    }

    template<class Tcons>
    inline constexpr Tcons d_hll_flux_mx(Tcons cL, Tcons cR) {
        return d_invert_axis(d_hll_flux_x(d_invert_axis(cL), d_invert_axis(cR)));
    }

    template<class Tcons>
    inline constexpr Tcons d_hll_flux_my(Tcons cL, Tcons cR) {
        return d_invert_axis(d_hll_flux_y(d_invert_axis(cL), d_invert_axis(cR)));
    }

    template<class Tcons>
    inline constexpr Tcons d_hll_flux_mz(Tcons cL, Tcons cR) {
        return d_invert_axis(d_hll_flux_z(d_invert_axis(cL), d_invert_axis(cR)));
    }

    template<class Tcons>
    inline constexpr Tcons huang_bai_flux_y(Tcons cL, Tcons cR) {
        return d_x_to_y(huang_bai_flux_x(d_y_to_x(cL), d_y_to_x(cR)));
    }

    template<class Tcons>
    inline constexpr Tcons huang_bai_flux_z(Tcons cL, Tcons cR) {
        return d_x_to_z(huang_bai_flux_x(d_z_to_x(cL), d_z_to_x(cR)));
    }

    template<class Tcons>
    inline constexpr Tcons huang_bai_flux_mx(Tcons cL, Tcons cR) {
        return d_invert_axis(huang_bai_flux_x(d_invert_axis(cL), d_invert_axis(cR)));
    }

    template<class Tcons>
    inline constexpr Tcons huang_bai_flux_my(Tcons cL, Tcons cR) {
        return d_invert_axis(huang_bai_flux_y(d_invert_axis(cL), d_invert_axis(cR)));
    }

    template<class Tcons>
    inline constexpr Tcons huang_bai_flux_mz(Tcons cL, Tcons cR) {
        return d_invert_axis(huang_bai_flux_z(d_invert_axis(cL), d_invert_axis(cR)));
    }
} // namespace shammath
