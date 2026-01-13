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
 * @file metrics.hpp
 * @author Yona Lapeyre (yona.lapeyre@ens-lyon.fr)
 * @brief
 */

#include "shamunits/Constants.hpp"
#include <shambackends/sycl.hpp>

namespace shamphys {

    // The Boyer-Lindquist metric tensor in CARTESIAN-like form
    template<class Tscal, class Tvec>
    inline Tvec get_cartesian_cov_metric(Tvec pos) {

        // metric params
        Tscal a     = 1;
        Tscal mass1 = 1.;
        Tscal rs    = 2. * mass1;

        // pos coords
        Tscal x, y, z = pos;
        Tscal x2       = x * x;
        Tscal y2       = y * y;
        Tscal z2       = z * z;
        Tscal a2       = a * a;
        Tscal r2_spher = x2 + y2 + z2;
        Tscal r2       = 0.5 * (r2_spher - a2)
                   + 0.5 * sycl::sqrt((r2_spher - a2) * (r2_spher - a2) + 4. * a2 * z2);
        Tscal r        = sycl::sqrt(r2);
        Tscal inv_r2   = 1. / r2;
        Tscal rho2     = r2 + a2 * (z2 * inv_r2);
        Tscal inv_rho2 = 1. / rho2;
        Tscal a2pr2    = a2 + r2;

        // metric components
        Tscal delta     = a2pr2 - rs * r;
        Tscal sintheta2 = 1. - z2 * inv_r2;
        Tscal gtt       = -1. + (r * rs) * inv_rho2;
        Tscal gphiphi   = sintheta2 * (a2pr2 + (a2 * r * rs * sintheta2) * inv_rho2);
        Tscal gtphi     = -((a * r * rs * sintheta2) * inv_rho2);
        Tscal omega     = a * r * rs / (rho2 * a2pr2 + rs * r * a2 * sintheta2);

        Tscal inv_rho2delta  = inv_rho2 / delta;
        Tscal gtphi_on_x2py2 = -rs * r * a / (rho2 * a2pr2);

        // let the fun begin
        Tvec g(4, 4);
        g(0, 0) = gtt;
        g(1, 0) = -y * gtphi_on_x2py2;
        g(2, 0) = x * gtphi_on_x2py2;
        g(3, 0) = 0.;
        g(0, 1) = g(1, 0);
        g(1, 1) = 1.
                  + (r * r * r * (a2pr2) *rs * x2 + a2 * delta * r * rs * y2)
                        / (delta * (a2pr2) * (a2pr2) *rho2);
        g(2, 1) = (r * (r * r * r * r + a2 * (-delta + r2)) * rs * x * y)
                  / (delta * (a2pr2) * (a2pr2) *rho2);
        g(3, 1) = (r * rs * x * z) / (delta * rho2);
        g(0, 2) = g(2, 0);
        g(1, 2) = g(2, 1);
        g(2, 2) = 1.
                  + (a2 * delta * r * rs * x2 + r * r * r * (a2pr2) *rs * y2)
                        / (delta * (a2pr2) * (a2pr2) *rho2);
        g(3, 2) = (r * rs * y * z) / (delta * rho2);
        g(0, 3) = g(3, 0);
        g(1, 3) = g(3, 1);
        g(2, 3) = g(3, 2);
        g(3, 3) = 1. + ((a2 + r2) * rs * z2) / (delta * r * rho2);

        return g;
    }

    template<class Tscal, class Tvec>
    inline Tvec get_cartesian_contrav_metric(Tvec pos) {

        // metric params
        Tscal a     = 1;
        Tscal mass1 = 1.;
        Tscal rs    = 2. * mass1;

        // pos coords
        Tscal x, y, z = pos;
        Tscal x2       = x * x;
        Tscal y2       = y * y;
        Tscal z2       = z * z;
        Tscal a2       = a * a;
        Tscal r2_spher = x2 + y2 + z2;
        Tscal r2       = 0.5 * (r2_spher - a2)
                   + 0.5 * sycl::sqrt((r2_spher - a2) * (r2_spher - a2) + 4. * a2 * z2);
        Tscal r        = sycl::sqrt(r2);
        Tscal inv_r2   = 1. / r2;
        Tscal rho2     = r2 + a2 * (z2 * inv_r2);
        Tscal inv_rho2 = 1. / rho2;
        Tscal a2pr2    = a2 + r2;

        // metric components
        Tscal delta     = a2pr2 - rs * r;
        Tscal sintheta2 = 1. - z2 * inv_r2;
        Tscal gtt       = -1. + (r * rs) * inv_rho2;
        Tscal gphiphi   = sintheta2 * (a2pr2 + (a2 * r * rs * sintheta2) * inv_rho2);
        Tscal gtphi     = -((a * r * rs * sintheta2) * inv_rho2);
        Tscal omega     = a * r * rs / (rho2 * a2pr2 + rs * r * a2 * sintheta2);

        Tscal inv_rho2delta  = inv_rho2 / delta;
        Tscal gtphi_on_x2py2 = -rs * r * a / (rho2 * a2pr2);

        // let the fun begin
        Tvec g(4, 4);
        Tscal domegaterm = 1. / (omega * gtphi + gtt);
        g(0, 0)          = domegaterm;
        g(1, 0)          = -y * omega * domegaterm;
        g(2, 0)          = x * omega * domegaterm;
        g(3, 0)          = 0.;
        g(0, 1)          = g(1, 0);
        // NOTE from Phantom: the expressions below are NOT regular at x=y=0 and z=r. Needs fixing!
        // Or use inv4x4 instead...
        g(1, 1) = (delta * r2 * x2) / (a2pr2 * a2pr2 * rho2)
                  + (gtt * y2) / (-gtphi * gtphi + gphiphi * gtt) + (x2 * z2) / (rho2 * (r2 - z2));
        g(2, 1) = -((gtt * x * y) / (-gtphi * gtphi + gphiphi * gtt))
                  + (delta * r2 * x * y) / (a2pr2 * a2pr2 * rho2)
                  + (x * y * z2) / (rho2 * (r2 - z2));
        g(3, 1) = -((x * z) * inv_rho2) + (delta * x * z) / (a2pr2 * rho2);
        g(0, 2) = g(2, 0);
        g(1, 2) = g(2, 1);
        g(2, 2) = (gtt * x2) / (-gtphi * gtphi + gphiphi * gtt)
                  + (delta * r2 * y2) / (a2pr2 * a2pr2 * rho2) + (y2 * z2) / (rho2 * (r2 - z2));
        g(3, 2) = -((y * z) * inv_rho2) + (delta * y * z) / (a2pr2 * rho2);
        g(0, 3) = g(3, 0);
        g(1, 3) = g(3, 1);
        g(2, 3) = g(3, 2);
        g(3, 3) = (r2 - z2) * inv_rho2 + (delta * z2) / (r2 * rho2);

        return g;
    }

    template<class Tscal, class Tvec>
    inline void metric_cartesian_derivatives(
        Tvec pos, Tvec &dgcovdx, Tvec &dgcovdy, Tvec &dgcovdz) {

        // metric params
        Tscal a     = 1;
        Tscal mass1 = 1.;
        Tscal rs    = 2. * mass1;

        // position coords
        Tscal x = pos(0);
        Tscal y = pos(1);
        Tscal z = pos(2);

        Tscal x2 = x * x;
        Tscal y2 = y * y;
        Tscal z2 = z * z;
        Tscal a2 = a * a;

        Tscal r2_spher = x2 + y2 + z2;
        Tscal r2       = 0.5 * (r2_spher - a2)
                   + 0.5 * sycl::sqrt((r2_spher - a2) * (r2_spher - a2) + 4. * a2 * z2);
        Tscal r = sycl::sqrt(r2);

        Tscal rho2  = r2 + a2 * (z2 / r2);
        Tscal a2pr2 = a2 + r2;
        Tscal delta = a2pr2 - rs * r;

        // Metric components
        Tscal sintheta2 = 1. - z2 / r2;
        Tscal gtt       = -1. + (r * rs) / rho2;
        Tscal gphiphi   = sintheta2 * (a2 + r2 + (a2 * r * rs * sintheta2) / rho2);
        Tscal gtphi     = -((a * r * rs * sintheta2) / rho2);

        // inverses
        Tscal r21    = 1. / r2;
        Tscal rho21  = 1. / rho2;
        Tscal delta1 = 1. / delta;
        Tscal r2mz2  = r2 - z2;
        Tscal r2mz21 = 1. / r2mz2;
        Tscal x2py2  = x2 + y2;
        Tscal x2py21 = 1. / x2py2;

        //  terms
        Tscal term1 = a2pr2 * rho21;
        Tscal term2 = (2. * a2pr2 * a2pr2 * rho21 * rho21 * z * z * z) / (r2 * r2);

        // Derivatives
        Tscal dr2dx = (2. * r2 * x) / rho2;
        Tscal dr2dy = (2. * r2 * y) / rho2;
        Tscal dr2dz = (2. * a2pr2 * z) / rho2;

        Tscal drho2dx = 2. * r21 * rho21 * x * (r2 * r2 - a2 * z2);
        Tscal drho2dy = 2. * r21 * rho21 * y * (r2 * r2 - a2 * z2);
        Tscal drho2dz = 2. * r21 * r21 * z * (a2 * r2 + a2pr2 * rho21 * (r2 * r2 - a2 * z2));

        Tscal ddeltadx = r * rho21 * (2. * r - rs) * x;
        Tscal ddeltady = r * rho21 * (2. * r - rs) * y;
        Tscal ddeltadz = (a2pr2 * rho21 * (2. * r - rs) * z) / r;

        // metric components
        Tscal dgtphidx = (a * rho21 * rho21 * rs * (drho2dx * r2mz2 - x * (r2 + z2))) / r;
        Tscal dgtphidy = (a * rho21 * rho21 * rs * (drho2dy * r2mz2 - y * (r2 + z2))) / r;
        Tscal dgtphidz
            = a * drho2dz * r * rho21 * rho21 * rs * sintheta2
              - (a * a2pr2 * rho21 * rho21 * rs * sintheta2 * z) / r
              - a * r * rho21 * rs * (-2. * r21 * z + 2. * a2pr2 * r21 * r21 * rho21 * z * z * z);

        Tscal dsintheta2dx = 2. * r21 * rho21 * x * z2;
        Tscal dsintheta2dy = 2. * r21 * rho21 * y * z2;
        Tscal dsintheta2dz = -2. * r21 * z + 2. * a2pr2 * r21 * r21 * rho21 * z * z * z;

        Tscal dgphiphidx
            = dsintheta2dx * r2 + dr2dx * sintheta2
              + a2 * rho21 * rho21
                    * (dsintheta2dx * rho2 * rho2 - drho2dx * r * rs * sintheta2 * sintheta2
                       + r * rs * sintheta2 * (2. * dsintheta2dx * rho2 + sintheta2 * x));

        Tscal dgphiphidy
            = dsintheta2dy * r2 + dr2dy * sintheta2
              + a2 * rho21 * rho21
                    * (dsintheta2dy * rho2 * rho2 - drho2dy * r * rs * sintheta2 * sintheta2
                       + r * rs * sintheta2 * (2. * dsintheta2dy * rho2 + sintheta2 * y));

        Tscal dgphiphidz
            = dsintheta2dz * r2 + dr2dz * sintheta2
              + (a2 * a2 * rho21 * rho21 * rs * sintheta2 * sintheta2 * z) / r
              + a2 * rho21 * rho21
                    * (dsintheta2dz * rho2 * rho2 - drho2dz * r * rs * sintheta2 * sintheta2
                       + r * rs * sintheta2 * (2. * dsintheta2dz * rho2 + sintheta2 * z));

        // let the fun begin again
        dgcovdx = Tvec(4, 4, 0.0);
        dgcovdy = Tvec(4, 4, 0.0);
        dgcovdz = Tvec(4, 4, 0.0);

        // X
        dgcovdx(0, 0) = r * rho21 * rho21 * rs * (-drho2dx + x);
        dgcovdx(0, 1) = x2py21 * (-dgtphidx + 2. * gtphi * x * x2py21) * y;
        dgcovdx(1, 1)
            = -(ddeltadx * delta1 * delta1 * r2 * rho21 * x2)
              + delta1 * r2 * rho21 * (2. * x + 2. * rho21 * x * x2 - drho2dx * rho21 * x2)
              + x2py21 * x2py21 * (dgphiphidx - 4. * gphiphi * x * x2py21) * y2
              - r2mz21 * rho21
                    * (-2. * x + 2. * r2 * r2mz21 * rho21 * x * x2 + drho2dx * rho21 * x2) * z2;

        dgcovdx(0, 2) = x2py21 * (gtphi + dgtphidx * x - 2. * gtphi * x2 * x2py21);
        dgcovdx(1, 2)
            = -(y
                * (ddeltadx * delta1 * delta1 * r2 * rho21 * x
                   + delta1 * r2 * rho21 * (-1. + drho2dx * rho21 * x - 2. * rho21 * x2)
                   + x2py21 * x2py21 * (gphiphi + dgphiphidx * x - 4. * gphiphi * x2 * x2py21)
                   + r2mz21 * rho21 * (-1. + drho2dx * rho21 * x + 2. * r2 * r2mz21 * rho21 * x2)
                         * z2));

        dgcovdx(2, 2) = dgphiphidx * x2 * x2py21 * x2py21
                        + 2. * gphiphi * x * x2py21 * x2py21 * (1. - 2. * x2 * x2py21)
                        - rho21 * y2
                              * (delta1 * r2 * (ddeltadx * delta1 + rho21 * (drho2dx - 2. * x))
                                 + r2mz21 * rho21 * (drho2dx + 2. * r2 * r2mz21 * x) * z2);

        dgcovdx(0, 3) = 0.;
        dgcovdx(1, 3)
            = -(z
                * (r2mz21 + delta1 * term1 * (-1. + ddeltadx * delta1 * x + drho2dx * rho21 * x)
                   - 2. * delta1 * r2 * rho21 * rho21 * x2
                   + r2mz21
                         * (-2. * rho21 * rho21 * x2
                            + r21 * term1 * (-1. + rho21 * x * (drho2dx + 2. * x)))
                         * z2
                   + 2. * r2 * r2mz21 * r2mz21 * rho21 * x2 * (-1. + r21 * term1 * z2)));

        dgcovdx(2, 3)
            = -(y * z
                * (ddeltadx * delta1 * delta1 * term1
                   + delta1 * rho21 * (drho2dx * term1 - 2. * r2 * rho21 * x)
                   + r2mz21 * rho21
                         * (-2. * r2 * r2mz21 * x - 2. * rho21 * x * z2
                            + r21 * term1 * (drho2dx + 2. * (x + r2 * r2mz21 * x)) * z2)));

        dgcovdx(3, 3)
            = -(a2pr2 * a2pr2 * delta1 * drho2dx * r21 * rho21 * rho21 * z2)
              - a2pr2 * a2pr2 * delta1 * r21 * rho21 * (ddeltadx * delta1 + 2. * rho21 * x) * z2
              - 2. * r2 * r2mz21 * r2mz21 * x * (-1. + r21 * term1 * z2) * (-1. + r21 * term1 * z2)
              - drho2dx * r2mz21 * (-1. + r21 * term1 * z2)
                    * (1. + r21 * (-1. + 2. * rho2 * rho21) * term1 * z2)
              + 4. * rho21 * x * z2
                    * (delta1 * term1
                       + r2mz21 * rho2 * (rho21 - r21 * term1) * (-1. + r21 * term1 * z2));

        // Y
        dgcovdy(0, 0) = r * rho21 * rho21 * rs * (-drho2dy + y);
        dgcovdy(0, 1) = x2py21 * (-(dgtphidy * y) + gtphi * (-1. + 2. * x2py21 * y2));
        dgcovdy(1, 1)
            = -(ddeltady * delta1 * delta1 * r2 * rho21 * x2)
              - delta1 * r2 * rho21 * rho21 * x2 * (drho2dy - 2. * y)
              + x2py21 * x2py21
                    * (2. * gphiphi * y - 4. * gphiphi * x2py21 * y * y2 + dgphiphidy * y2)
              - r2mz21 * rho21 * rho21 * x2 * (drho2dy + 2. * r2 * r2mz21 * y) * z2;

        dgcovdy(0, 2) = x * x2py21 * (dgtphidy - 2. * gtphi * x2py21 * y);
        dgcovdy(1, 2)
            = -(x
                * (ddeltady * delta1 * delta1 * r2 * rho21 * y
                   + delta1 * r2 * rho21 * (-1. + drho2dy * rho21 * y - 2. * rho21 * y2)
                   + x2py21 * x2py21 * (gphiphi + dgphiphidy * y - 4. * gphiphi * x2py21 * y2)
                   + r2mz21 * rho21 * (-1. + drho2dy * rho21 * y + 2. * r2 * r2mz21 * rho21 * y2)
                         * z2));

        dgcovdy(2, 2)
            = dgphiphidy * x2 * x2py21 * x2py21 + 2. * delta1 * r2 * rho21 * y
              - 4. * gphiphi * x2 * x2py21 * x2py21 * x2py21 * y
              - delta1 * r2 * rho21
                    * (-2. * rho21 * y * y2 + ddeltady * delta1 * y2 + drho2dy * rho21 * y2)
              - r2mz21 * rho21
                    * (-2. * y + 2. * r2 * r2mz21 * rho21 * y * y2 + drho2dy * rho21 * y2) * z2;

        dgcovdy(0, 3) = 0.;
        dgcovdy(1, 3)
            = -(x * z
                * (ddeltady * delta1 * delta1 * term1
                   + delta1 * rho21 * (drho2dy * term1 - 2. * r2 * rho21 * y)
                   + r2mz21 * rho21
                         * (-2. * r2 * r2mz21 * y - 2. * rho21 * y * z2
                            + r21 * term1 * (drho2dy + 2. * (y + r2 * r2mz21 * y)) * z2)));

        dgcovdy(2, 3)
            = -(z
                * (r2mz21 + delta1 * term1 * (-1. + ddeltady * delta1 * y + drho2dy * rho21 * y)
                   - 2. * delta1 * r2 * rho21 * rho21 * y2
                   + r2mz21
                         * (-2. * rho21 * rho21 * y2
                            + r21 * term1 * (-1. + rho21 * y * (drho2dy + 2. * y)))
                         * z2
                   + 2. * r2 * r2mz21 * r2mz21 * rho21 * y2 * (-1. + r21 * term1 * z2)));

        dgcovdy(3, 3)
            = -(a2pr2 * a2pr2 * delta1 * drho2dy * r21 * rho21 * rho21 * z2)
              - a2pr2 * a2pr2 * delta1 * r21 * rho21 * (ddeltady * delta1 + 2. * rho21 * y) * z2
              - 2. * r2 * r2mz21 * r2mz21 * y * (-1. + r21 * term1 * z2) * (-1. + r21 * term1 * z2)
              - drho2dy * r2mz21 * (-1. + r21 * term1 * z2)
                    * (1. + r21 * (-1. + 2. * rho2 * rho21) * term1 * z2)
              + 4. * rho21 * y * z2
                    * (delta1 * term1
                       + r2mz21 * rho2 * (rho21 - r21 * term1) * (-1. + r21 * term1 * z2));

        // Z
        dgcovdz(0, 0) = -(drho2dz * r * rho21 * rho21 * rs) + (rho21 * rs * term1 * z) / r;
        dgcovdz(0, 1) = -(dgtphidz * x2py21 * y);
        dgcovdz(1, 1) = -(ddeltadz * delta1 * delta1 * r2 * rho21 * x2)
                        - delta1 * drho2dz * r2 * rho21 * rho21 * x2
                        + dgphiphidz * x2py21 * x2py21 * y2 + 2. * r2mz21 * rho21 * x2 * z
                        + 2. * delta1 * rho21 * term1 * x2 * z
                        - drho2dz * r2mz21 * rho21 * rho21 * x2 * z2
                        - r2mz21 * r2mz21 * rho21 * x2 * (-2. * z + 2. * term1 * z) * z2;

        dgcovdz(0, 2) = dgtphidz * x * x2py21;
        dgcovdz(1, 2) = -(ddeltadz * delta1 * delta1 * r2 * rho21 * x * y)
                        - delta1 * drho2dz * r2 * rho21 * rho21 * x * y
                        - dgphiphidz * x * x2py21 * x2py21 * y + 2. * r2mz21 * rho21 * x * y * z
                        + 2. * delta1 * rho21 * term1 * x * y * z
                        - drho2dz * r2mz21 * rho21 * rho21 * x * y * z2
                        - r2mz21 * r2mz21 * rho21 * x * y * (-2. * z + 2. * term1 * z) * z2;

        dgcovdz(2, 2)
            = dgphiphidz * x2 * x2py21 * x2py21 - ddeltadz * delta1 * delta1 * r2 * rho21 * y2
              - delta1 * drho2dz * r2 * rho21 * rho21 * y2 + 2. * r2mz21 * rho21 * y2 * z
              + 2. * delta1 * rho21 * term1 * y2 * z - drho2dz * r2mz21 * rho21 * rho21 * y2 * z2
              - r2mz21 * r2mz21 * rho21 * y2 * (-2. * z + 2. * term1 * z) * z2;

        dgcovdz(0, 3) = 0.;
        dgcovdz(1, 3)
            = delta1 * term1 * x - ddeltadz * delta1 * delta1 * term1 * x * z
              - delta1 * drho2dz * rho21 * term1 * x * z + 2. * delta1 * rho21 * term1 * x * z2
              - r2mz21 * x * (1. - r21 * term1 * z2)
              + r2mz21 * r2mz21 * x * z * (-2. * z + 2. * term1 * z) * (1. - r21 * term1 * z2)
              - r2mz21 * x * z
                    * (term2 - 2. * r21 * term1 * z - 2. * r21 * rho21 * term1 * z * z2
                       + drho2dz * r21 * rho21 * term1 * z2);

        dgcovdz(2, 3)
            = delta1 * term1 * y - ddeltadz * delta1 * delta1 * term1 * y * z
              - delta1 * drho2dz * rho21 * term1 * y * z + 2. * delta1 * rho21 * term1 * y * z2
              - r2mz21 * y * (1. - r21 * term1 * z2)
              + r2mz21 * r2mz21 * y * z * (-2. * z + 2. * term1 * z) * (1. - r21 * term1 * z2)
              - r2mz21 * y * z
                    * (term2 - 2. * r21 * term1 * z - 2. * r21 * rho21 * term1 * z * z2
                       + drho2dz * r21 * rho21 * term1 * z2);

        dgcovdz(3, 3) = 2. * a2pr2 * a2pr2 * delta1 * r21 * rho21 * z
                        - (2. * a2pr2 * a2pr2 * a2pr2 * delta1 * rho21 * rho21 * z * z2) / (r2 * r2)
                        + 4. * a2pr2 * a2pr2 * delta1 * r21 * rho21 * rho21 * z * z2
                        - a2pr2 * a2pr2 * ddeltadz * delta1 * delta1 * r21 * rho21 * z2
                        - a2pr2 * a2pr2 * delta1 * drho2dz * r21 * rho21 * rho21 * z2
                        + drho2dz * r2mz21 * (1. - r21 * term1 * z2) * (1. - r21 * term1 * z2)
                        - r2mz21 * r2mz21 * rho2 * (-2. * z + 2. * term1 * z)
                              * (1. - r21 * term1 * z2) * (1. - r21 * term1 * z2)
                        + 2. * r2mz21 * rho2 * (1. - r21 * term1 * z2)
                              * (term2 - 2. * r21 * term1 * z - 2. * r21 * rho21 * term1 * z * z2
                                 + drho2dz * r21 * rho21 * term1 * z2);

        // Hopefully that was alright

        dgcovdx(1, 0) = dgcovdx(0, 1);
        dgcovdx(2, 0) = dgcovdx(0, 2);
        dgcovdx(2, 1) = dgcovdx(1, 2);
        dgcovdx(3, 0) = dgcovdx(0, 3);
        dgcovdx(3, 1) = dgcovdx(1, 3);
        dgcovdx(3, 2) = dgcovdx(2, 3);

        dgcovdy(1, 0) = dgcovdy(0, 1);
        dgcovdy(2, 0) = dgcovdy(0, 2);
        dgcovdy(2, 1) = dgcovdy(1, 2);
        dgcovdy(3, 0) = dgcovdy(0, 3);
        dgcovdy(3, 1) = dgcovdy(1, 3);
        dgcovdy(3, 2) = dgcovdy(2, 3);

        dgcovdz(1, 0) = dgcovdz(0, 1);
        dgcovdz(2, 0) = dgcovdz(0, 2);
        dgcovdz(2, 1) = dgcovdz(1, 2);
        dgcovdz(3, 0) = dgcovdz(0, 3);
        dgcovdz(3, 1) = dgcovdz(1, 3);
        dgcovdz(3, 2) = dgcovdz(2, 3);
    }

} // namespace shamphys
