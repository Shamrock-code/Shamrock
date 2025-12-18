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
 * @file sphkernels.hpp
 * @author Guo (guo.yansong@optimind.tech)
 * @author Timothée David--Cléris (tim.shamrock@proton.me)
 * @brief sph kernels
 */

//%Impl status : Good

#include "shambase/constants.hpp"
#include "shambase/type_name_info.hpp"
#include "shambackends/math.hpp"
#include "shammath/integrator.hpp"

namespace shammath::details {

    template<class Tscal>
    class KernelDefM4 {
        public:
        inline static constexpr Tscal Rkern = 2; ///< Compact support radius of the kernel
        /// default hfact to be used for this kernel
        inline static constexpr Tscal hfactd = 1.2;

        /// 1D norm of the kernel
        inline static constexpr Tscal norm_1d = 2. / 3.;
        /// 2D norm of the kernel
        inline static constexpr Tscal norm_2d = 10. / (7. * shambase::constants::pi<Tscal>);
        /// 3D norm of the kernel
        inline static constexpr Tscal norm_3d = 1 / shambase::constants::pi<Tscal>;

        inline static Tscal f(Tscal q) {

            Tscal t1 = 2 - q;
            Tscal t2 = 1 - q;

            t1 = t1 * t1 * t1;
            t2 = t2 * t2 * t2;

            constexpr Tscal div1_4 = (1. / 4.);
            t1 *= div1_4;
            t2 *= -1;

            if (q < 1) {
                return t1 + t2;
            } else if (q < 2) {
                return t1;
            } else
                return 0;
        }

        inline static Tscal df(Tscal q) {

            constexpr Tscal div9_4 = (9. / 4.);
            constexpr Tscal div3_4 = (3. / 4.);

            if (q < 1) {
                return -3 * q + div9_4 * q * q;
            } else if (q < 2) {
                return -3 + 3 * q - div3_4 * q * q;
            } else
                return 0;
        }

        inline static Tscal ddf(Tscal q) {
            if (q < 1) {
                return (9.0 / 2.0) * q - 3;
            } else if (q < 2) {
                return 3 - (3.0 / 2.0) * q;
            } else
                return 0;
        }

        inline static Tscal phi_tilde_3d(Tscal q) {
            Tscal t1 = sham::pow_constexpr<2>(q);
            Tscal t2 = sham::pow_constexpr<3>(q);
            Tscal t3 = sham::pow_constexpr<4>(q);
            Tscal t4 = sham::pow_constexpr<5>(q);
            Tscal t5 = sham::pow_constexpr<6>(q);
            if (q < 1) {
                return (1.0 / 30.0) * shambase::constants::pi<Tscal>
                       * (t1 * (3 * t2 - 9 * t1 + 20) - 42);
            } else if (q < 2) {
                return (1.0 / 30.0) * shambase::constants::pi<Tscal>
                       * (-t5 + 9 * t4 - 30 * t3 + 40 * t2 - 48 * q + 2) / q;
            } else
                return -shambase::constants::pi<Tscal> / q;
        }

        inline static Tscal phi_tilde_3d_prime(Tscal q) {
            Tscal t1 = sham::pow_constexpr<2>(q);
            Tscal t2 = sham::pow_constexpr<3>(q);
            Tscal t3 = sham::pow_constexpr<4>(q);
            Tscal t4 = sham::pow_constexpr<5>(q);
            Tscal t5 = sham::pow_constexpr<6>(q);
            if (q < 1) {
                return (1.0 / 30.0) * shambase::constants::pi<Tscal>
                       * (t1 * (9 * t1 - 18 * q) + 2 * q * (3 * t2 - 9 * t1 + 20));
            } else if (q < 2) {
                return (1.0 / 30.0) * shambase::constants::pi<Tscal>
                           * (-6 * t4 + 45 * t3 - 120 * t2 + 120 * t1 - 48) / q
                       - (1.0 / 30.0) * shambase::constants::pi<Tscal>
                             * (-t5 + 9 * t4 - 30 * t3 + 40 * t2 - 48 * q + 2) / t1;
            } else
                return shambase::constants::pi<Tscal> / t1;
        }
    };

    template<class Tscal>
    class KernelDefM5 {
        public:
        inline static constexpr Tscal Rkern = 5. / 2.; ///< Compact support radius of the kernel
        /// default hfact to be used for this kernel
        inline static constexpr Tscal hfactd = 1.2;

        /// 1D norm of the kernel
        inline static constexpr Tscal norm_1d = 1. / 24.;
        /// 2D norm of the kernel
        inline static constexpr Tscal norm_2d = 96. / (1199 * shambase::constants::pi<Tscal>);
        /// 3D norm of the kernel
        inline static constexpr Tscal norm_3d = 1 / (20 * shambase::constants::pi<Tscal>);

        inline static Tscal f(Tscal q) {

            constexpr Tscal div5_2 = (5. / 2.);
            constexpr Tscal div3_2 = (3. / 2.);
            constexpr Tscal div1_2 = (1. / 2.);

            Tscal t1 = div5_2 - q;
            Tscal t2 = div3_2 - q;
            Tscal t3 = div1_2 - q;

            Tscal t1_2 = t1 * t1;
            Tscal t2_2 = t2 * t2;
            Tscal t3_2 = t3 * t3;

            t1 = t1_2 * t1_2;
            t2 = t2_2 * t2_2;
            t3 = t3_2 * t3_2;

            t1 *= 1;
            t2 *= -5;
            t3 *= 10;

            if (q < div1_2) {
                return t1 + t2 + t3;
            } else if (q < div3_2) {
                return t1 + t2;
            } else if (q < div5_2) {
                return t1;
            } else
                return 0;
        }

        inline static Tscal df(Tscal q) {

            constexpr Tscal div5_2 = (5. / 2.);
            constexpr Tscal div3_2 = (3. / 2.);
            constexpr Tscal div1_2 = (1. / 2.);

            Tscal t1 = div5_2 - q;
            Tscal t2 = div3_2 - q;
            Tscal t3 = div1_2 - q;

            Tscal t1_2 = t1 * t1;
            Tscal t2_2 = t2 * t2;
            Tscal t3_2 = t3 * t3;

            t1 = t1 * t1_2;
            t2 = t2 * t2_2;
            t3 = t3 * t3_2;

            t1 *= (1) * (-4);
            t2 *= (-5) * (-4);
            t3 *= (10) * (-4);

            if (q < div1_2) {
                return t1 + t2 + t3;
            } else if (q < div3_2) {
                return t1 + t2;
            } else if (q < div5_2) {
                return t1;
            } else
                return 0;
        }

        inline static Tscal ddf(Tscal q) {
            Tscal t1 = sham::pow_constexpr<2>(1.0 / 2.0 - q);
            Tscal t2 = sham::pow_constexpr<2>(3.0 / 2.0 - q);
            Tscal t3 = sham::pow_constexpr<2>(5.0 / 2.0 - q);
            if (q < 1.0 / 2.0) {
                return 120 * t1 - 60 * t2 + 12 * t3;
            } else if (q < 3.0 / 2.0) {
                return -60 * t2 + 12 * t3;
            } else if (q < 5.0 / 2.0) {
                return 12 * t3;
            } else
                return 0;
        }

        inline static Tscal phi_tilde_3d(Tscal q) {
            Tscal t1 = sham::pow_constexpr<2>(q);
            Tscal t2 = sham::pow_constexpr<3>(q);
            Tscal t3 = sham::pow_constexpr<4>(q);
            Tscal t4 = sham::pow_constexpr<5>(q);
            Tscal t5 = sham::pow_constexpr<6>(q);
            Tscal t6 = sham::pow_constexpr<7>(q);
            if (q < 1.0 / 2.0) {
                return (1.0 / 336.0) * shambase::constants::pi<Tscal>
                       * (192 * t5 - 1008 * t3 + 3220 * t1 - 8393);
            } else if (q < 3.0 / 2.0) {
                return (1.0 / 336.0) * shambase::constants::pi<Tscal>
                       * (-128 * t6 + 896 * t5 - 2016 * t4 + 560 * t3 + 3080 * t2 - 8386 * q - 1)
                       / q;
            } else if (q < 5.0 / 2.0) {
                return (1.0 / 672.0) * shambase::constants::pi<Tscal>
                       * (64 * t6 - 896 * t5 + 5040 * t4 - 14000 * t3 + 17500 * t2 - 21875 * q
                          + 2185)
                       / q;
            } else
                return -20 * shambase::constants::pi<Tscal> / q;
        }

        inline static Tscal phi_tilde_3d_prime(Tscal q) {
            Tscal t1 = sham::pow_constexpr<2>(q);
            Tscal t2 = sham::pow_constexpr<3>(q);
            Tscal t3 = sham::pow_constexpr<4>(q);
            Tscal t4 = sham::pow_constexpr<5>(q);
            Tscal t5 = sham::pow_constexpr<6>(q);
            Tscal t6 = sham::pow_constexpr<7>(q);
            if (q < 1.0 / 2.0) {
                return (1.0 / 336.0) * shambase::constants::pi<Tscal>
                       * (1152 * t4 - 4032 * t2 + 6440 * q);
            } else if (q < 3.0 / 2.0) {
                return (1.0 / 336.0) * shambase::constants::pi<Tscal>
                           * (-896 * t5 + 5376 * t4 - 10080 * t3 + 2240 * t2 + 9240 * t1 - 8386) / q
                       - (1.0 / 336.0) * shambase::constants::pi<Tscal>
                             * (-128 * t6 + 896 * t5 - 2016 * t4 + 560 * t3 + 3080 * t2 - 8386 * q
                                - 1)
                             / t1;
            } else if (q < 5.0 / 2.0) {
                return (1.0 / 672.0) * shambase::constants::pi<Tscal>
                           * (448 * t5 - 5376 * t4 + 25200 * t3 - 56000 * t2 + 52500 * t1 - 21875)
                           / q
                       - (1.0 / 672.0) * shambase::constants::pi<Tscal>
                             * (64 * t6 - 896 * t5 + 5040 * t4 - 14000 * t3 + 17500 * t2 - 21875 * q
                                + 2185)
                             / t1;
            } else
                return 20 * shambase::constants::pi<Tscal> / t1;
        }
    };

    template<class Tscal>
    class KernelDefM6 {
        public:
        inline static constexpr Tscal Rkern = 3; ///< Compact support radius of the kernel
        /// default hfact to be used for this kernel
        inline static constexpr Tscal hfactd = 1.0;

        /// 1D norm of the kernel
        inline static constexpr Tscal norm_1d = 1. / 120.;
        /// 2D norm of the kernel
        inline static constexpr Tscal norm_2d = 7. / (478 * shambase::constants::pi<Tscal>);
        /// 3D norm of the kernel
        inline static constexpr Tscal norm_3d = 1 / (120 * shambase::constants::pi<Tscal>);

        inline static Tscal f(Tscal q) {

            Tscal t1 = 3 - q;
            Tscal t2 = 2 - q;
            Tscal t3 = 1 - q;

            Tscal t1_2 = t1 * t1;
            Tscal t2_2 = t2 * t2;
            Tscal t3_2 = t3 * t3;

            t1 = t1 * t1_2 * t1_2;
            t2 = t2 * t2_2 * t2_2;
            t3 = t3 * t3_2 * t3_2;

            t1 *= 1;
            t2 *= -6;
            t3 *= 15;

            if (q < 1.) {
                return t1 + t2 + t3;
            } else if (q < 2.) {
                return t1 + t2;
            } else if (q < 3.) {
                return t1;
            } else
                return 0;
        }

        inline static Tscal df(Tscal q) {

            Tscal t1 = 3 - q;
            Tscal t2 = 2 - q;
            Tscal t3 = 1 - q;

            Tscal t1_2 = t1 * t1;
            Tscal t2_2 = t2 * t2;
            Tscal t3_2 = t3 * t3;

            t1 = t1_2 * t1_2;
            t2 = t2_2 * t2_2;
            t3 = t3_2 * t3_2;

            t1 *= (1) * (-5);
            t2 *= (-6) * (-5);
            t3 *= (15) * (-5);

            if (q < 1.) {
                return t1 + t2 + t3;
            } else if (q < 2.) {
                return t1 + t2;
            } else if (q < 3.) {
                return t1;
            } else
                return 0;
        }

        inline static Tscal ddf(Tscal q) {
            Tscal t1 = sham::pow_constexpr<3>(1 - q);
            Tscal t2 = sham::pow_constexpr<3>(2 - q);
            Tscal t3 = sham::pow_constexpr<3>(3 - q);
            if (q < 1) {
                return 300 * t1 - 120 * t2 + 20 * t3;
            } else if (q < 2) {
                return -120 * t2 + 20 * t3;
            } else if (q < 3) {
                return 20 * t3;
            } else
                return 0;
        }

        inline static Tscal phi_tilde_3d(Tscal q) {
            Tscal t1 = sham::pow_constexpr<2>(q);
            Tscal t2 = sham::pow_constexpr<3>(q);
            Tscal t3 = sham::pow_constexpr<4>(q);
            Tscal t4 = sham::pow_constexpr<5>(q);
            Tscal t5 = sham::pow_constexpr<6>(q);
            Tscal t6 = sham::pow_constexpr<7>(q);
            Tscal t7 = sham::pow_constexpr<8>(q);
            if (q < 1) {
                return (1.0 / 7.0) * shambase::constants::pi<Tscal>
                       * (-5 * t6 + 20 * t5 - 84 * t3 + 308 * t1 - 956);
            } else if (q < 2) {
                return (1.0 / 14.0) * shambase::constants::pi<Tscal>
                       * (5 * t7 - 60 * t6 + 280 * t5 - 588 * t4 + 350 * t3 + 476 * t2 - 1892 * q
                          - 5)
                       / q;
            } else if (q < 3) {
                return (1.0 / 14.0) * shambase::constants::pi<Tscal>
                       * (-t7 + 20 * t6 - 168 * t5 + 756 * t4 - 1890 * t3 + 2268 * t2 - 2916 * q
                          + 507)
                       / q;
            } else
                return -120 * shambase::constants::pi<Tscal> / q;
        }

        inline static Tscal phi_tilde_3d_prime(Tscal q) {
            Tscal t1 = sham::pow_constexpr<2>(q);
            Tscal t2 = sham::pow_constexpr<3>(q);
            Tscal t3 = sham::pow_constexpr<4>(q);
            Tscal t4 = sham::pow_constexpr<5>(q);
            Tscal t5 = sham::pow_constexpr<6>(q);
            Tscal t6 = sham::pow_constexpr<7>(q);
            Tscal t7 = sham::pow_constexpr<8>(q);
            if (q < 1) {
                return (1.0 / 7.0) * shambase::constants::pi<Tscal>
                       * (-35 * t5 + 120 * t4 - 336 * t2 + 616 * q);
            } else if (q < 2) {
                return (1.0 / 14.0) * shambase::constants::pi<Tscal>
                           * (40 * t6 - 420 * t5 + 1680 * t4 - 2940 * t3 + 1400 * t2 + 1428 * t1
                              - 1892)
                           / q
                       - (1.0 / 14.0) * shambase::constants::pi<Tscal>
                             * (5 * t7 - 60 * t6 + 280 * t5 - 588 * t4 + 350 * t3 + 476 * t2
                                - 1892 * q - 5)
                             / t1;
            } else if (q < 3) {
                return (1.0 / 14.0) * shambase::constants::pi<Tscal>
                           * (-8 * t6 + 140 * t5 - 1008 * t4 + 3780 * t3 - 7560 * t2 + 6804 * t1
                              - 2916)
                           / q
                       - (1.0 / 14.0) * shambase::constants::pi<Tscal>
                             * (-t7 + 20 * t6 - 168 * t5 + 756 * t4 - 1890 * t3 + 2268 * t2
                                - 2916 * q + 507)
                             / t1;
            } else
                return 120 * shambase::constants::pi<Tscal> / t1;
        }
    };

    template<class Tscal>
    class KernelDefM7 {
        public:
        inline static constexpr Tscal Rkern = 3.5; ///< Compact support radius of the kernel
        /// default hfact to be used for this kernel
        inline static constexpr Tscal hfactd = 1.0;

        /// 1D norm of the kernel
        inline static constexpr Tscal norm_1d = 1. / 720;
        /// 2D norm of the kernel
        inline static constexpr Tscal norm_2d = 256. / (113149 * shambase::constants::pi<Tscal>);
        /// 3D norm of the kernel
        inline static constexpr Tscal norm_3d = 6. / (5040. * shambase::constants::pi<Tscal>);

        inline static Tscal f(Tscal q) {

            constexpr Tscal div7_2 = (7. / 2.);
            constexpr Tscal div5_2 = (5. / 2.);
            constexpr Tscal div3_2 = (3. / 2.);
            constexpr Tscal div1_2 = (1. / 2.);

            Tscal t1 = div7_2 - q;
            Tscal t2 = div5_2 - q;
            Tscal t3 = div3_2 - q;
            Tscal t4 = div1_2 - q;

            // up to order 6
            t1 = sham::pow_constexpr<6>(t1);
            t2 = sham::pow_constexpr<6>(t2);
            t3 = sham::pow_constexpr<6>(t3);
            t4 = sham::pow_constexpr<6>(t4);

            t1 *= 1.;
            t2 *= -7.;
            t3 *= 21.;
            t4 *= -35.;

            if (q < div1_2) {
                return t1 + t2 + t3 + t4;
            } else if (q < div3_2) {
                return t1 + t2 + t3;
            } else if (q < div5_2) {
                return t1 + t2;
            } else if (q < div7_2) {
                return t1;
            } else
                return 0;
        }

        inline static Tscal df(Tscal q) {

            constexpr Tscal div7_2 = (7. / 2.);
            constexpr Tscal div5_2 = (5. / 2.);
            constexpr Tscal div3_2 = (3. / 2.);
            constexpr Tscal div1_2 = (1. / 2.);

            Tscal t1 = div7_2 - q;
            Tscal t2 = div5_2 - q;
            Tscal t3 = div3_2 - q;
            Tscal t4 = div1_2 - q;

            // up to order 6
            t1 = sham::pow_constexpr<5>(t1);
            t2 = sham::pow_constexpr<5>(t2);
            t3 = sham::pow_constexpr<5>(t3);
            t4 = sham::pow_constexpr<5>(t4);

            t1 *= (-6.) * (1.);
            t2 *= (-6.) * (-7.);
            t3 *= (-6.) * (21.);
            t4 *= (-6.) * (-35.);

            if (q < div1_2) {
                return t1 + t2 + t3 + t4;
            } else if (q < div3_2) {
                return t1 + t2 + t3;
            } else if (q < div5_2) {
                return t1 + t2;
            } else if (q < div7_2) {
                return t1;
            } else
                return 0;
        }

        inline static Tscal ddf(Tscal q) {
            Tscal t1 = sham::pow_constexpr<4>(1.0 / 2.0 - q);
            Tscal t2 = sham::pow_constexpr<4>(3.0 / 2.0 - q);
            Tscal t3 = sham::pow_constexpr<4>(5.0 / 2.0 - q);
            Tscal t4 = sham::pow_constexpr<4>(7.0 / 2.0 - q);
            if (q < 1.0 / 2.0) {
                return -1050 * t1 + 630 * t2 - 210 * t3 + 30 * t4;
            } else if (q < 3.0 / 2.0) {
                return 630 * t2 - 210 * t3 + 30 * t4;
            } else if (q < 5.0 / 2.0) {
                return -210 * t3 + 30 * t4;
            } else if (q < 7.0 / 2.0) {
                return 30 * t4;
            } else
                return 0;
        }

        inline static Tscal phi_tilde_3d(Tscal q) {
            Tscal t1 = sham::pow_constexpr<2>(q);
            Tscal t2 = sham::pow_constexpr<3>(q);
            Tscal t3 = sham::pow_constexpr<4>(q);
            Tscal t4 = sham::pow_constexpr<5>(q);
            Tscal t5 = sham::pow_constexpr<6>(q);
            Tscal t6 = sham::pow_constexpr<7>(q);
            Tscal t7 = sham::pow_constexpr<8>(q);
            Tscal t8 = sham::pow_constexpr<9>(q);
            if (q < 1.0 / 2.0) {
                return (1.0 / 1152.0) * shambase::constants::pi<Tscal>
                       * (-1280 * t7 + 11520 * t5 - 66528 * t3 + 282576 * t1 - 1018341);
            } else if (q < 3.0 / 2.0) {
                return (1.0 / 4608.0) * shambase::constants::pi<Tscal>
                       * (3840 * t8 - 34560 * t7 + 103680 * t6 - 53760 * t5 - 235872 * t4
                          - 10080 * t3 + 1131984 * t2 - 4073409 * q + 5)
                       / q;
            } else if (q < 5.0 / 2.0) {
                return (1.0 / 2304.0) * shambase::constants::pi<Tscal>
                       * (-768 * t8 + 13824 * t7 - 103680 * t6 + 408576 * t5 - 852768 * t4
                          + 729792 * t3 + 198576 * t2 - 1948131 * q - 29522)
                       / q;
            } else if (q < 7.0 / 2.0) {
                return (1.0 / 4608.0) * shambase::constants::pi<Tscal>
                       * (256 * t8 - 6912 * t7 + 80640 * t6 - 526848 * t5 + 2074464 * t4
                          - 4840416 * t3 + 5647152 * t2 - 7411887 * q + 1894081)
                       / q;
            } else
                return -840 * shambase::constants::pi<Tscal> / q;
        }

        inline static Tscal phi_tilde_3d_prime(Tscal q) {
            Tscal t1 = sham::pow_constexpr<2>(q);
            Tscal t2 = sham::pow_constexpr<3>(q);
            Tscal t3 = sham::pow_constexpr<4>(q);
            Tscal t4 = sham::pow_constexpr<5>(q);
            Tscal t5 = sham::pow_constexpr<6>(q);
            Tscal t6 = sham::pow_constexpr<7>(q);
            Tscal t7 = sham::pow_constexpr<8>(q);
            Tscal t8 = sham::pow_constexpr<9>(q);
            if (q < 1.0 / 2.0) {
                return (1.0 / 1152.0) * shambase::constants::pi<Tscal>
                       * (-10240 * t6 + 69120 * t4 - 266112 * t2 + 565152 * q);
            } else if (q < 3.0 / 2.0) {
                return (1.0 / 4608.0) * shambase::constants::pi<Tscal>
                           * (34560 * t7 - 276480 * t6 + 725760 * t5 - 322560 * t4 - 1179360 * t3
                              - 40320 * t2 + 3395952 * t1 - 4073409)
                           / q
                       - (1.0 / 4608.0) * shambase::constants::pi<Tscal>
                             * (3840 * t8 - 34560 * t7 + 103680 * t6 - 53760 * t5 - 235872 * t4
                                - 10080 * t3 + 1131984 * t2 - 4073409 * q + 5)
                             / t1;
            } else if (q < 5.0 / 2.0) {
                return (1.0 / 2304.0) * shambase::constants::pi<Tscal>
                           * (-6912 * t7 + 110592 * t6 - 725760 * t5 + 2451456 * t4 - 4263840 * t3
                              + 2919168 * t2 + 595728 * t1 - 1948131)
                           / q
                       - (1.0 / 2304.0) * shambase::constants::pi<Tscal>
                             * (-768 * t8 + 13824 * t7 - 103680 * t6 + 408576 * t5 - 852768 * t4
                                + 729792 * t3 + 198576 * t2 - 1948131 * q - 29522)
                             / t1;
            } else if (q < 7.0 / 2.0) {
                return (1.0 / 4608.0) * shambase::constants::pi<Tscal>
                           * (2304 * t7 - 55296 * t6 + 564480 * t5 - 3161088 * t4 + 10372320 * t3
                              - 19361664 * t2 + 16941456 * t1 - 7411887)
                           / q
                       - (1.0 / 4608.0) * shambase::constants::pi<Tscal>
                             * (256 * t8 - 6912 * t7 + 80640 * t6 - 526848 * t5 + 2074464 * t4
                                - 4840416 * t3 + 5647152 * t2 - 7411887 * q + 1894081)
                             / t1;
            } else
                return 840 * shambase::constants::pi<Tscal> / t1;
        }
    };

    template<class Tscal>
    class KernelDefM8 {
        public:
        inline static constexpr Tscal Rkern = 4.0; ///< Compact support radius of the kernel
        /// default hfact to be used for this kernel
        inline static constexpr Tscal hfactd = 1.0;

        /// 1D norm of the kernel
        inline static constexpr Tscal norm_1d = 1.0 / 5040.0;
        /// 2D norm of the kernel
        inline static constexpr Tscal norm_2d = (9.0 / 29740.0) / shambase::constants::pi<Tscal>;
        /// 3D norm of the kernel
        inline static constexpr Tscal norm_3d = (1.0 / 6720.0) / shambase::constants::pi<Tscal>;

        inline static Tscal f(Tscal q) {

            Tscal t1 = 4 - q;
            Tscal t2 = 3 - q;
            Tscal t3 = 2 - q;
            Tscal t4 = 1 - q;

            t1 = sham::pow_constexpr<7>(t1);
            t2 = sham::pow_constexpr<7>(t2);
            t3 = sham::pow_constexpr<7>(t3);
            t4 = sham::pow_constexpr<7>(t4);

            t1 *= 1.;
            t2 *= -8.;
            t3 *= 28.;
            t4 *= -56.;

            if (q < 1) {
                return t1 + t2 + t3 + t4;
            } else if (q < 2) {
                return t1 + t2 + t3;
            } else if (q < 3) {
                return t1 + t2;
            } else if (q < 4) {
                return t1;
            } else
                return 0;
        }

        inline static Tscal df(Tscal q) {

            Tscal t1 = 4 - q;
            Tscal t2 = 3 - q;
            Tscal t3 = 2 - q;
            Tscal t4 = 1 - q;

            t1 = sham::pow_constexpr<6>(t1);
            t2 = sham::pow_constexpr<6>(t2);
            t3 = sham::pow_constexpr<6>(t3);
            t4 = sham::pow_constexpr<6>(t4);

            t1 *= (-7.) * (1.);
            t2 *= (-7.) * (-8.);
            t3 *= (-7.) * (28.);
            t4 *= (-7.) * (-56.);

            if (q < 1) {
                return t1 + t2 + t3 + t4;
            } else if (q < 2) {
                return t1 + t2 + t3;
            } else if (q < 3) {
                return t1 + t2;
            } else if (q < 4) {
                return t1;
            } else
                return 0;
        }

        inline static Tscal ddf(Tscal q) {
            Tscal t1 = sham::pow_constexpr<5>(1 - q);
            Tscal t2 = sham::pow_constexpr<5>(2 - q);
            Tscal t3 = sham::pow_constexpr<5>(3 - q);
            Tscal t4 = sham::pow_constexpr<5>(4 - q);
            if (q < 1) {
                return -2352 * t1 + 1176 * t2 - 336 * t3 + 42 * t4;
            } else if (q < 2) {
                return 1176 * t2 - 336 * t3 + 42 * t4;
            } else if (q < 3) {
                return -336 * t3 + 42 * t4;
            } else if (q < 4) {
                return 42 * t4;
            } else
                return 0;
        }

        inline static Tscal phi_tilde_3d(Tscal q) {
            Tscal t1 = sham::pow_constexpr<10>(q);
            Tscal t2 = sham::pow_constexpr<2>(q);
            Tscal t3 = sham::pow_constexpr<3>(q);
            Tscal t4 = sham::pow_constexpr<4>(q);
            Tscal t5 = sham::pow_constexpr<5>(q);
            Tscal t6 = sham::pow_constexpr<6>(q);
            Tscal t7 = sham::pow_constexpr<7>(q);
            Tscal t8 = sham::pow_constexpr<8>(q);
            Tscal t9 = sham::pow_constexpr<9>(q);
            if (q < 1) {
                return (2.0 / 9.0) * shambase::constants::pi<Tscal>
                       * (t2 * (7 * t7 - 35 * t6 + 240 * t4 - 1512 * t2 + 7248) - 29740);
            } else if (q < 2) {
                return (2.0 / 45.0) * shambase::constants::pi<Tscal>
                       * (-21 * t1 + 315 * t9 - 1890 * t8 + 5400 * t7 - 5880 * t6 - 2268 * t5
                          - 2940 * t4 + 37080 * t3 - 148770 * q + 14)
                       / q;
            } else if (q < 3) {
                return (2.0 / 45.0) * shambase::constants::pi<Tscal>
                       * (7 * t1 - 175 * t9 + 1890 * t8 - 11400 * t7 + 41160 * t6 - 86940 * t5
                          + 91140 * t4 - 16680 * t3 - 130850 * q - 7154)
                       / q;
            } else if (q < 4) {
                return (2.0 / 45.0) * shambase::constants::pi<Tscal>
                       * (-t1 + 35 * t9 - 540 * t8 + 4800 * t7 - 26880 * t6 + 96768 * t5
                          - 215040 * t4 + 245760 * t3 - 327680 * q + 110944)
                       / q;
            } else
                return -6720 * shambase::constants::pi<Tscal> / q;
        }

        inline static Tscal phi_tilde_3d_prime(Tscal q) {
            Tscal t1 = sham::pow_constexpr<10>(q);
            Tscal t2 = sham::pow_constexpr<2>(q);
            Tscal t3 = sham::pow_constexpr<3>(q);
            Tscal t4 = sham::pow_constexpr<4>(q);
            Tscal t5 = sham::pow_constexpr<5>(q);
            Tscal t6 = sham::pow_constexpr<6>(q);
            Tscal t7 = sham::pow_constexpr<7>(q);
            Tscal t8 = sham::pow_constexpr<8>(q);
            Tscal t9 = sham::pow_constexpr<9>(q);
            if (q < 1) {
                return (2.0 / 9.0) * shambase::constants::pi<Tscal>
                       * (t2 * (49 * t6 - 210 * t5 + 960 * t3 - 3024 * q)
                          + 2 * q * (7 * t7 - 35 * t6 + 240 * t4 - 1512 * t2 + 7248));
            } else if (q < 2) {
                return (2.0 / 45.0) * shambase::constants::pi<Tscal>
                           * (-210 * t9 + 2835 * t8 - 15120 * t7 + 37800 * t6 - 35280 * t5
                              - 11340 * t4 - 11760 * t3 + 111240 * t2 - 148770)
                           / q
                       - (2.0 / 45.0) * shambase::constants::pi<Tscal>
                             * (-21 * t1 + 315 * t9 - 1890 * t8 + 5400 * t7 - 5880 * t6 - 2268 * t5
                                - 2940 * t4 + 37080 * t3 - 148770 * q + 14)
                             / t2;
            } else if (q < 3) {
                return (2.0 / 45.0) * shambase::constants::pi<Tscal>
                           * (70 * t9 - 1575 * t8 + 15120 * t7 - 79800 * t6 + 246960 * t5
                              - 434700 * t4 + 364560 * t3 - 50040 * t2 - 130850)
                           / q
                       - (2.0 / 45.0) * shambase::constants::pi<Tscal>
                             * (7 * t1 - 175 * t9 + 1890 * t8 - 11400 * t7 + 41160 * t6 - 86940 * t5
                                + 91140 * t4 - 16680 * t3 - 130850 * q - 7154)
                             / t2;
            } else if (q < 4) {
                return (2.0 / 45.0) * shambase::constants::pi<Tscal>
                           * (-10 * t9 + 315 * t8 - 4320 * t7 + 33600 * t6 - 161280 * t5
                              + 483840 * t4 - 860160 * t3 + 737280 * t2 - 327680)
                           / q
                       - (2.0 / 45.0) * shambase::constants::pi<Tscal>
                             * (-t1 + 35 * t9 - 540 * t8 + 4800 * t7 - 26880 * t6 + 96768 * t5
                                - 215040 * t4 + 245760 * t3 - 327680 * q + 110944)
                             / t2;
            } else
                return 6720 * shambase::constants::pi<Tscal> / t2;
        }
    };

    template<class Tscal>
    class KernelDefM9 {
        public:
        inline static constexpr Tscal Rkern = 4.5; ///< Compact support radius of the kernel
        /// default hfact to be used for this kernel
        inline static constexpr Tscal hfactd = 1.0;

        /// 1D norm of the kernel
        inline static constexpr Tscal norm_1d = 1. / 40320;
        /// 2D norm of the kernel
        inline static constexpr Tscal norm_2d = 512. / (14345663 * shambase::constants::pi<Tscal>);
        /// 3D norm of the kernel
        inline static constexpr Tscal norm_3d = 6. / (362880. * shambase::constants::pi<Tscal>);

        inline static Tscal f(Tscal q) {

            constexpr Tscal div9_2 = (9. / 2.);
            constexpr Tscal div7_2 = (7. / 2.);
            constexpr Tscal div5_2 = (5. / 2.);
            constexpr Tscal div3_2 = (3. / 2.);
            constexpr Tscal div1_2 = (1. / 2.);

            Tscal t1 = div9_2 - q;
            Tscal t2 = div7_2 - q;
            Tscal t3 = div5_2 - q;
            Tscal t4 = div3_2 - q;
            Tscal t5 = div1_2 - q;

            t1 = sham::pow_constexpr<8>(t1);
            t2 = sham::pow_constexpr<8>(t2);
            t3 = sham::pow_constexpr<8>(t3);
            t4 = sham::pow_constexpr<8>(t4);
            t5 = sham::pow_constexpr<8>(t5);

            t1 *= 1.;
            t2 *= -9.;
            t3 *= 36.;
            t4 *= -84.;
            t5 *= 126.;

            if (q < div1_2) {
                return t1 + t2 + t3 + t4 + t5;
            } else if (q < div3_2) {
                return t1 + t2 + t3 + t4;
            } else if (q < div5_2) {
                return t1 + t2 + t3;
            } else if (q < div7_2) {
                return t1 + t2;
            } else if (q < div9_2) {
                return t1;
            } else
                return 0;
        }

        inline static Tscal df(Tscal q) {

            constexpr Tscal div9_2 = (9. / 2.);
            constexpr Tscal div7_2 = (7. / 2.);
            constexpr Tscal div5_2 = (5. / 2.);
            constexpr Tscal div3_2 = (3. / 2.);
            constexpr Tscal div1_2 = (1. / 2.);

            Tscal t1 = div9_2 - q;
            Tscal t2 = div7_2 - q;
            Tscal t3 = div5_2 - q;
            Tscal t4 = div3_2 - q;
            Tscal t5 = div1_2 - q;

            t1 = sham::pow_constexpr<7>(t1);
            t2 = sham::pow_constexpr<7>(t2);
            t3 = sham::pow_constexpr<7>(t3);
            t4 = sham::pow_constexpr<7>(t4);
            t5 = sham::pow_constexpr<7>(t5);

            t1 *= (-8.) * (1.);
            t2 *= (-8.) * (-9.);
            t3 *= (-8.) * (36.);
            t4 *= (-8.) * (-84.);
            t5 *= (-8.) * (126.);

            if (q < div1_2) {
                return t1 + t2 + t3 + t4 + t5;
            } else if (q < div3_2) {
                return t1 + t2 + t3 + t4;
            } else if (q < div5_2) {
                return t1 + t2 + t3;
            } else if (q < div7_2) {
                return t1 + t2;
            } else if (q < div9_2) {
                return t1;
            } else
                return 0;
        }

        inline static Tscal ddf(Tscal q) {
            Tscal t1 = sham::pow_constexpr<6>(1.0 / 2.0 - q);
            Tscal t2 = sham::pow_constexpr<6>(3.0 / 2.0 - q);
            Tscal t3 = sham::pow_constexpr<6>(5.0 / 2.0 - q);
            Tscal t4 = sham::pow_constexpr<6>(7.0 / 2.0 - q);
            Tscal t5 = sham::pow_constexpr<6>(9.0 / 2.0 - q);
            if (q < 1.0 / 2.0) {
                return 7056 * t1 - 4704 * t2 + 2016 * t3 - 504 * t4 + 56 * t5;
            } else if (q < 3.0 / 2.0) {
                return -4704 * t2 + 2016 * t3 - 504 * t4 + 56 * t5;
            } else if (q < 5.0 / 2.0) {
                return 2016 * t3 - 504 * t4 + 56 * t5;
            } else if (q < 7.0 / 2.0) {
                return -504 * t4 + 56 * t5;
            } else if (q < 9.0 / 2.0) {
                return 56 * t5;
            } else
                return 0;
        }

        inline static Tscal phi_tilde_3d(Tscal q) {
            Tscal t1  = sham::pow_constexpr<10>(q);
            Tscal t2  = sham::pow_constexpr<11>(q);
            Tscal t3  = sham::pow_constexpr<2>(q);
            Tscal t4  = sham::pow_constexpr<3>(q);
            Tscal t5  = sham::pow_constexpr<4>(q);
            Tscal t6  = sham::pow_constexpr<5>(q);
            Tscal t7  = sham::pow_constexpr<6>(q);
            Tscal t8  = sham::pow_constexpr<7>(q);
            Tscal t9  = sham::pow_constexpr<8>(q);
            Tscal t10 = sham::pow_constexpr<9>(q);
            if (q < 1.0 / 2.0) {
                return (1.0 / 2816.0) * shambase::constants::pi<Tscal>
                       * (7168 * t1 - 98560 * t9 + 908160 * t7 - 6408864 * t5 + 34283436 * t3
                          - 157802293);
            } else if (q < 3.0 / 2.0) {
                return (1.0 / 14080.0) * shambase::constants::pi<Tscal>
                       * (-28672 * t2 + 315392 * t1 - 1182720 * t10 + 887040 * t9 + 3801600 * t8
                          + 413952 * t7 - 32199552 * t6 + 36960 * t5 + 171412560 * t4
                          - 789011388 * q - 7)
                       / q;
            } else if (q < 5.0 / 2.0) {
                return (1.0 / 14080.0) * shambase::constants::pi<Tscal>
                       * (14336 * t2 - 315392 * t1 + 2956800 * t10 - 15079680 * t9 + 43718400 * t8
                          - 66646272 * t7 + 43243200 * t6 - 53850720 * t5 + 191620440 * t4
                          - 792042570 * q + 826679)
                       / q;
            } else if (q < 7.0 / 2.0) {
                return (1.0 / 14080.0) * shambase::constants::pi<Tscal>
                       * (-4096 * t2 + 135168 * t1 - 1971200 * t10 + 16600320 * t9 - 88281600 * t8
                          + 302953728 * t7 - 649756800 * t6 + 771149280 * t5 - 324004560 * t4
                          - 577198820 * q - 96829571)
                       / q;
            } else if (q < 9.0 / 2.0) {
                return (1.0 / 28160.0) * shambase::constants::pi<Tscal>
                       * (1024 * t2 - 45056 * t1 + 887040 * t10 - 10264320 * t9 + 76982400 * t8
                          - 387991296 * t7 + 1309470624 * t6 - 2806008480 * t5 + 3156759540 * t4
                          - 4261625379 * q + 1783667601)
                       / q;
            } else
                return -60480 * shambase::constants::pi<Tscal> / q;
        }

        inline static Tscal phi_tilde_3d_prime(Tscal q) {
            Tscal t1  = sham::pow_constexpr<10>(q);
            Tscal t2  = sham::pow_constexpr<11>(q);
            Tscal t3  = sham::pow_constexpr<2>(q);
            Tscal t4  = sham::pow_constexpr<3>(q);
            Tscal t5  = sham::pow_constexpr<4>(q);
            Tscal t6  = sham::pow_constexpr<5>(q);
            Tscal t7  = sham::pow_constexpr<6>(q);
            Tscal t8  = sham::pow_constexpr<7>(q);
            Tscal t9  = sham::pow_constexpr<8>(q);
            Tscal t10 = sham::pow_constexpr<9>(q);
            if (q < 1.0 / 2.0) {
                return (1.0 / 2816.0) * shambase::constants::pi<Tscal>
                       * (71680 * t10 - 788480 * t8 + 5448960 * t6 - 25635456 * t4 + 68566872 * q);
            } else if (q < 3.0 / 2.0) {
                return (1.0 / 14080.0) * shambase::constants::pi<Tscal>
                           * (-315392 * t1 + 3153920 * t10 - 10644480 * t9 + 7096320 * t8
                              + 26611200 * t7 + 2483712 * t6 - 160997760 * t5 + 147840 * t4
                              + 514237680 * t3 - 789011388)
                           / q
                       - (1.0 / 14080.0) * shambase::constants::pi<Tscal>
                             * (-28672 * t2 + 315392 * t1 - 1182720 * t10 + 887040 * t9
                                + 3801600 * t8 + 413952 * t7 - 32199552 * t6 + 36960 * t5
                                + 171412560 * t4 - 789011388 * q - 7)
                             / t3;
            } else if (q < 5.0 / 2.0) {
                return (1.0 / 14080.0) * shambase::constants::pi<Tscal>
                           * (157696 * t1 - 3153920 * t10 + 26611200 * t9 - 120637440 * t8
                              + 306028800 * t7 - 399877632 * t6 + 216216000 * t5 - 215402880 * t4
                              + 574861320 * t3 - 792042570)
                           / q
                       - (1.0 / 14080.0) * shambase::constants::pi<Tscal>
                             * (14336 * t2 - 315392 * t1 + 2956800 * t10 - 15079680 * t9
                                + 43718400 * t8 - 66646272 * t7 + 43243200 * t6 - 53850720 * t5
                                + 191620440 * t4 - 792042570 * q + 826679)
                             / t3;
            } else if (q < 7.0 / 2.0) {
                return (1.0 / 14080.0) * shambase::constants::pi<Tscal>
                           * (-45056 * t1 + 1351680 * t10 - 17740800 * t9 + 132802560 * t8
                              - 617971200 * t7 + 1817722368 * t6 - 3248784000 * t5 + 3084597120 * t4
                              - 972013680 * t3 - 577198820)
                           / q
                       - (1.0 / 14080.0) * shambase::constants::pi<Tscal>
                             * (-4096 * t2 + 135168 * t1 - 1971200 * t10 + 16600320 * t9
                                - 88281600 * t8 + 302953728 * t7 - 649756800 * t6 + 771149280 * t5
                                - 324004560 * t4 - 577198820 * q - 96829571)
                             / t3;
            } else if (q < 9.0 / 2.0) {
                return (1.0 / 28160.0) * shambase::constants::pi<Tscal>
                           * (11264 * t1 - 450560 * t10 + 7983360 * t9 - 82114560 * t8
                              + 538876800 * t7 - 2327947776 * t6 + 6547353120 * t5
                              - 11224033920 * t4 + 9470278620 * t3 - 4261625379)
                           / q
                       - (1.0 / 28160.0) * shambase::constants::pi<Tscal>
                             * (1024 * t2 - 45056 * t1 + 887040 * t10 - 10264320 * t9
                                + 76982400 * t8 - 387991296 * t7 + 1309470624 * t6 - 2806008480 * t5
                                + 3156759540 * t4 - 4261625379 * q + 1783667601)
                             / t3;
            } else
                return 60480 * shambase::constants::pi<Tscal> / t3;
        }
    };

    template<class Tscal>
    class KernelDefM10 {
        public:
        inline static constexpr Tscal Rkern = 5; ///< Compact support radius of the kernel
        /// default hfact to be used for this kernel
        inline static constexpr Tscal hfactd = 1.0;

        /// 1D norm of the kernel
        inline static constexpr Tscal norm_1d = 1. / 362880;
        /// 2D norm of the kernel
        inline static constexpr Tscal norm_2d = 11. / (2922230 * shambase::constants::pi<Tscal>);
        /// 3D norm of the kernel
        inline static constexpr Tscal norm_3d = 6. / (3628800. * shambase::constants::pi<Tscal>);

        inline static Tscal f(Tscal q) {

            Tscal t1 = 5 - q;
            Tscal t2 = 4 - q;
            Tscal t3 = 3 - q;
            Tscal t4 = 2 - q;
            Tscal t5 = 1 - q;

            t1 = sham::pow_constexpr<9>(t1);
            t2 = sham::pow_constexpr<9>(t2);
            t3 = sham::pow_constexpr<9>(t3);
            t4 = sham::pow_constexpr<9>(t4);
            t5 = sham::pow_constexpr<9>(t5);

            t1 *= 1.;
            t2 *= -10.;
            t3 *= 45.;
            t4 *= -120.;
            t5 *= 210.;

            if (q < 1) {
                return t1 + t2 + t3 + t4 + t5;
            } else if (q < 2) {
                return t1 + t2 + t3 + t4;
            } else if (q < 3) {
                return t1 + t2 + t3;
            } else if (q < 4) {
                return t1 + t2;
            } else if (q < 5) {
                return t1;
            } else
                return 0;
        }

        inline static Tscal df(Tscal q) {

            Tscal t1 = 5 - q;
            Tscal t2 = 4 - q;
            Tscal t3 = 3 - q;
            Tscal t4 = 2 - q;
            Tscal t5 = 1 - q;

            t1 = sham::pow_constexpr<8>(t1);
            t2 = sham::pow_constexpr<8>(t2);
            t3 = sham::pow_constexpr<8>(t3);
            t4 = sham::pow_constexpr<8>(t4);
            t5 = sham::pow_constexpr<8>(t5);

            t1 *= (-9.) * (1.);
            t2 *= (-9.) * (-10.);
            t3 *= (-9.) * (45.);
            t4 *= (-9.) * (-120.);
            t5 *= (-9.) * (210.);

            if (q < 1) {
                return t1 + t2 + t3 + t4 + t5;
            } else if (q < 2) {
                return t1 + t2 + t3 + t4;
            } else if (q < 3) {
                return t1 + t2 + t3;
            } else if (q < 4) {
                return t1 + t2;
            } else if (q < 5) {
                return t1;
            } else
                return 0;
        }

        inline static Tscal ddf(Tscal q) {
            Tscal t1 = sham::pow_constexpr<7>(1 - q);
            Tscal t2 = sham::pow_constexpr<7>(2 - q);
            Tscal t3 = sham::pow_constexpr<7>(3 - q);
            Tscal t4 = sham::pow_constexpr<7>(4 - q);
            Tscal t5 = sham::pow_constexpr<7>(5 - q);
            if (q < 1) {
                return 15120 * t1 - 8640 * t2 + 3240 * t3 - 720 * t4 + 72 * t5;
            } else if (q < 2) {
                return -8640 * t2 + 3240 * t3 - 720 * t4 + 72 * t5;
            } else if (q < 3) {
                return 3240 * t3 - 720 * t4 + 72 * t5;
            } else if (q < 4) {
                return -720 * t4 + 72 * t5;
            } else if (q < 5) {
                return 72 * t5;
            } else
                return 0;
        }

        inline static Tscal phi_tilde_3d(Tscal q) {
            Tscal t1  = sham::pow_constexpr<10>(q);
            Tscal t2  = sham::pow_constexpr<11>(q);
            Tscal t3  = sham::pow_constexpr<12>(q);
            Tscal t4  = sham::pow_constexpr<2>(q);
            Tscal t5  = sham::pow_constexpr<3>(q);
            Tscal t6  = sham::pow_constexpr<4>(q);
            Tscal t7  = sham::pow_constexpr<5>(q);
            Tscal t8  = sham::pow_constexpr<6>(q);
            Tscal t9  = sham::pow_constexpr<7>(q);
            Tscal t10 = sham::pow_constexpr<8>(q);
            Tscal t11 = sham::pow_constexpr<9>(q);
            if (q < 1) {
                return (2.0 / 33.0) * shambase::constants::pi<Tscal>
                       * (-63 * t2 + 378 * t1 - 3850 * t10 + 37620 * t8 - 291060 * t6 + 1718090 * t4
                          - 8766690);
            } else if (q < 2) {
                return (2.0 / 33.0) * shambase::constants::pi<Tscal>
                       * (42 * t3 - 756 * t2 + 5544 * t1 - 20020 * t11 + 31185 * t10 - 3960 * t9
                          + 38808 * t8 - 316008 * t7 + 10395 * t6 + 1715780 * t5 - 8766564 * q - 21)
                       / q;
            } else if (q < 3) {
                return (2.0 / 33.0) * shambase::constants::pi<Tscal>
                       * (-18 * t3 + 540 * t2 - 7128 * t1 + 53900 * t11 - 253935 * t10 + 756360 * t9
                          - 1380456 * t8 + 1508760 * t7 - 1510245 * t6 + 2391620 * t5 - 8914020 * q
                          + 49131)
                       / q;
            } else if (q < 4) {
                return (1.0 / 33.0) * shambase::constants::pi<Tscal>
                       * (9 * t3 - 378 * t2 + 7128 * t1 - 79310 * t11 + 574695 * t10 - 2817540 * t9
                          + 9363816 * t8 - 20365884 * t7 + 26208765 * t6 - 14702930 * t5
                          - 8262102 * q - 4684707)
                       / q;
            } else if (q < 5) {
                return (1.0 / 33.0) * shambase::constants::pi<Tscal>
                       * (-t3 + 54 * t2 - 1320 * t1 + 19250 * t11 - 185625 * t10 + 1237500 * t9
                          - 5775000 * t8 + 18562500 * t7 - 38671875 * t6 + 42968750 * t5
                          - 58593750 * q + 28869725)
                       / q;
            } else
                return -604800 * shambase::constants::pi<Tscal> / q;
        }

        inline static Tscal phi_tilde_3d_prime(Tscal q) {
            Tscal t1  = sham::pow_constexpr<10>(q);
            Tscal t2  = sham::pow_constexpr<11>(q);
            Tscal t3  = sham::pow_constexpr<12>(q);
            Tscal t4  = sham::pow_constexpr<2>(q);
            Tscal t5  = sham::pow_constexpr<3>(q);
            Tscal t6  = sham::pow_constexpr<4>(q);
            Tscal t7  = sham::pow_constexpr<5>(q);
            Tscal t8  = sham::pow_constexpr<6>(q);
            Tscal t9  = sham::pow_constexpr<7>(q);
            Tscal t10 = sham::pow_constexpr<8>(q);
            Tscal t11 = sham::pow_constexpr<9>(q);
            if (q < 1) {
                return (2.0 / 33.0) * shambase::constants::pi<Tscal>
                       * (-693 * t1 + 3780 * t11 - 30800 * t9 + 225720 * t7 - 1164240 * t5
                          + 3436180 * q);
            } else if (q < 2) {
                return (2.0 / 33.0) * shambase::constants::pi<Tscal>
                           * (504 * t2 - 8316 * t1 + 55440 * t11 - 180180 * t10 + 249480 * t9
                              - 27720 * t8 + 232848 * t7 - 1580040 * t6 + 41580 * t5 + 5147340 * t4
                              - 8766564)
                           / q
                       - (2.0 / 33.0) * shambase::constants::pi<Tscal>
                             * (42 * t3 - 756 * t2 + 5544 * t1 - 20020 * t11 + 31185 * t10
                                - 3960 * t9 + 38808 * t8 - 316008 * t7 + 10395 * t6 + 1715780 * t5
                                - 8766564 * q - 21)
                             / t4;
            } else if (q < 3) {
                return (2.0 / 33.0) * shambase::constants::pi<Tscal>
                           * (-216 * t2 + 5940 * t1 - 71280 * t11 + 485100 * t10 - 2031480 * t9
                              + 5294520 * t8 - 8282736 * t7 + 7543800 * t6 - 6040980 * t5
                              + 7174860 * t4 - 8914020)
                           / q
                       - (2.0 / 33.0) * shambase::constants::pi<Tscal>
                             * (-18 * t3 + 540 * t2 - 7128 * t1 + 53900 * t11 - 253935 * t10
                                + 756360 * t9 - 1380456 * t8 + 1508760 * t7 - 1510245 * t6
                                + 2391620 * t5 - 8914020 * q + 49131)
                             / t4;
            } else if (q < 4) {
                return (1.0 / 33.0) * shambase::constants::pi<Tscal>
                           * (108 * t2 - 4158 * t1 + 71280 * t11 - 713790 * t10 + 4597560 * t9
                              - 19722780 * t8 + 56182896 * t7 - 101829420 * t6 + 104835060 * t5
                              - 44108790 * t4 - 8262102)
                           / q
                       - (1.0 / 33.0) * shambase::constants::pi<Tscal>
                             * (9 * t3 - 378 * t2 + 7128 * t1 - 79310 * t11 + 574695 * t10
                                - 2817540 * t9 + 9363816 * t8 - 20365884 * t7 + 26208765 * t6
                                - 14702930 * t5 - 8262102 * q - 4684707)
                             / t4;
            } else if (q < 5) {
                return (1.0 / 33.0) * shambase::constants::pi<Tscal>
                           * (-12 * t2 + 594 * t1 - 13200 * t11 + 173250 * t10 - 1485000 * t9
                              + 8662500 * t8 - 34650000 * t7 + 92812500 * t6 - 154687500 * t5
                              + 128906250 * t4 - 58593750)
                           / q
                       - (1.0 / 33.0) * shambase::constants::pi<Tscal>
                             * (-t3 + 54 * t2 - 1320 * t1 + 19250 * t11 - 185625 * t10
                                + 1237500 * t9 - 5775000 * t8 + 18562500 * t7 - 38671875 * t6
                                + 42968750 * t5 - 58593750 * q + 28869725)
                             / t4;
            } else
                return 604800 * shambase::constants::pi<Tscal> / t4;
        }
    };

    template<class Tscal>
    class KernelDefC2 {
        public:
        inline static constexpr Tscal Rkern = 2; ///< Compact support radius of the kernel
        /// default hfact to be used for this kernel
        inline static constexpr Tscal hfactd = 1.0;

        /// 1D norm of the kernel
        inline static constexpr Tscal norm_1d = 3. / 4.;
        /// 2D norm of the kernel
        inline static constexpr Tscal norm_2d = 7. / (4 * shambase::constants::pi<Tscal>);
        /// 3D norm of the kernel
        inline static constexpr Tscal norm_3d = 21 / (16 * shambase::constants::pi<Tscal>);

        inline static Tscal f(Tscal q) {

            constexpr Tscal div1_2 = (1. / 2.);

            Tscal p1 = (1 - q * div1_2);
            Tscal p2 = (1 + q * 2);

            p1 *= p1;
            p1 *= p1;

            if (q < 2.) {
                return p1 * p2;
            } else
                return 0;
        }

        inline static Tscal df(Tscal q) {

            constexpr Tscal div1_2 = (1. / 2.);
            constexpr Tscal div1_8 = (1. / 8.);
            constexpr Tscal div1_4 = (1. / 4.);

            Tscal p1 = (1 - q * div1_2);
            Tscal p2 = (1 + q * 2);

            Tscal p3 = p1 * p1 * p1;

            if (q < 2.) {
                return 2 * (p3 * p1 - p3 * p2);
            } else
                return 0;
        }

        inline static Tscal ddf(Tscal q) {
            Tscal t1 = sham::pow_constexpr<2>(1 - (1.0 / 2.0) * q);
            Tscal t2 = sham::pow_constexpr<3>(1 - (1.0 / 2.0) * q);
            if (q < 2) {
                return -8 * t2 + 3 * t1 * (2 * q + 1);
            } else
                return 0;
        }

        inline static Tscal phi_tilde_3d(Tscal q) {
            Tscal t1 = sham::pow_constexpr<2>(q);
            Tscal t2 = sham::pow_constexpr<3>(q);
            Tscal t3 = sham::pow_constexpr<4>(q);
            Tscal t4 = sham::pow_constexpr<5>(q);
            if (q < 2) {
                return (1.0 / 336.0) * shambase::constants::pi<Tscal>
                       * (t1 * (3 * t4 - 30 * t3 + 112 * t2 - 168 * t1 + 224) - 384);
            } else
                return -(16.0 / 21.0) * shambase::constants::pi<Tscal> / q;
        }

        inline static Tscal phi_tilde_3d_prime(Tscal q) {
            Tscal t1 = sham::pow_constexpr<2>(q);
            Tscal t2 = sham::pow_constexpr<3>(q);
            Tscal t3 = sham::pow_constexpr<4>(q);
            Tscal t4 = sham::pow_constexpr<5>(q);
            if (q < 2) {
                return (1.0 / 336.0) * shambase::constants::pi<Tscal>
                       * (t1 * (15 * t3 - 120 * t2 + 336 * t1 - 336 * q)
                          + 2 * q * (3 * t4 - 30 * t3 + 112 * t2 - 168 * t1 + 224));
            } else
                return (16.0 / 21.0) * shambase::constants::pi<Tscal> / t1;
        }
    };

    template<class Tscal>
    class KernelDefC4 {
        public:
        inline static constexpr Tscal Rkern = 2; ///< Compact support radius of the kernel
        /// default hfact to be used for this kernel
        /// Using same hfact as M4 for comparable neighbor counts
        inline static constexpr Tscal hfactd = 1.2;

        /// 1D norm of the kernel
        inline static constexpr Tscal norm_1d = 27. / 32.;
        /// 2D norm of the kernel
        inline static constexpr Tscal norm_2d = 9. / (4 * shambase::constants::pi<Tscal>);
        /// 3D norm of the kernel
        inline static constexpr Tscal norm_3d = 495. / (256. * shambase::constants::pi<Tscal>);

        inline static Tscal f(Tscal q) {

            constexpr Tscal div1_2   = (1. / 2.);
            constexpr Tscal div35_12 = (35. / 12.);

            Tscal p1 = (1 - q * div1_2);
            Tscal p2 = (1 + q * 3 + div35_12 * q * q);

            p1 *= p1;
            p1 = p1 * p1 * p1;

            if (q < 2.) {
                return p1 * p2;
            } else
                return 0;
        }

        inline static Tscal df(Tscal q) {

            constexpr Tscal div7_96 = (7. / 96.);

            Tscal p1 = (-2 + q);
            Tscal p2 = (2 + 5 * q);

            Tscal p14 = p1 * p1;
            p14 *= p14;

            if (q < 2.) {
                return div7_96 * p14 * p1 * q * p2;
            } else
                return 0;
        }

        inline static Tscal ddf(Tscal q) {
            Tscal t1 = sham::pow_constexpr<2>(q);
            Tscal t2 = sham::pow_constexpr<4>(1 - (1.0 / 2.0) * q);
            Tscal t3 = sham::pow_constexpr<5>(1 - (1.0 / 2.0) * q);
            Tscal t4 = sham::pow_constexpr<6>(1 - (1.0 / 2.0) * q);
            if (q < 2) {
                return (35.0 / 6.0) * t4 - 6 * t3 * ((35.0 / 6.0) * q + 3)
                       + (15.0 / 2.0) * t2 * ((35.0 / 12.0) * t1 + 3 * q + 1);
            } else
                return 0;
        }

        inline static Tscal phi_tilde_3d(Tscal q) {
            Tscal t1 = sham::pow_constexpr<2>(q);
            Tscal t2 = sham::pow_constexpr<4>(q);
            Tscal t3 = sham::pow_constexpr<5>(q);
            Tscal t4 = sham::pow_constexpr<6>(q);
            Tscal t5 = sham::pow_constexpr<7>(q);
            Tscal t6 = sham::pow_constexpr<8>(q);
            if (q < 2) {
                return (1.0 / 63360.0) * shambase::constants::pi<Tscal>
                       * (t1
                              * (105 * t6 - 1408 * t5 + 7700 * t4 - 21120 * t3 + 26400 * t2
                                 - 29568 * t1 + 42240)
                          - 56320);
            } else
                return -(256.0 / 495.0) * shambase::constants::pi<Tscal> / q;
        }

        inline static Tscal phi_tilde_3d_prime(Tscal q) {
            Tscal t1 = sham::pow_constexpr<2>(q);
            Tscal t2 = sham::pow_constexpr<3>(q);
            Tscal t3 = sham::pow_constexpr<4>(q);
            Tscal t4 = sham::pow_constexpr<5>(q);
            Tscal t5 = sham::pow_constexpr<6>(q);
            Tscal t6 = sham::pow_constexpr<7>(q);
            Tscal t7 = sham::pow_constexpr<8>(q);
            if (q < 2) {
                return (1.0 / 63360.0) * shambase::constants::pi<Tscal>
                       * (t1
                              * (840 * t6 - 9856 * t5 + 46200 * t4 - 105600 * t3 + 105600 * t2
                                 - 59136 * q)
                          + 2 * q
                                * (105 * t7 - 1408 * t6 + 7700 * t5 - 21120 * t4 + 26400 * t3
                                   - 29568 * t1 + 42240));
            } else
                return (256.0 / 495.0) * shambase::constants::pi<Tscal> / t1;
        }
    };

    template<class Tscal>
    class KernelDefC6 {
        public:
        inline static constexpr Tscal Rkern = 2; ///< Compact support radius of the kernel
        /// default hfact to be used for this kernel
        inline static constexpr Tscal hfactd = 1.0;

        /// 1D norm of the kernel
        inline static constexpr Tscal norm_1d = 15. / 16.;
        /// 2D norm of the kernel
        inline static constexpr Tscal norm_2d = 39. / (14. * shambase::constants::pi<Tscal>);
        /// 3D norm of the kernel
        inline static constexpr Tscal norm_3d = 1365. / (512. * shambase::constants::pi<Tscal>);

        inline static Tscal f(Tscal q) {

            constexpr Tscal div1_2  = (1. / 2.);
            constexpr Tscal div25_4 = (25. / 4.);

            Tscal p1 = (1 - q * div1_2);
            Tscal p2 = (1 + q * 4 + div25_4 * q * q + 4 * q * q * q);

            p1 *= p1;
            p1 *= p1;
            p1 *= p1;

            if (q < 2.) {
                return p1 * p2;
            } else
                return 0;
        }

        inline static Tscal df(Tscal q) {

            constexpr Tscal div11_512 = (11. / 512.);

            Tscal p1 = (-2 + q);
            Tscal p2 = 2 + 7 * q + 8 * q * q;

            Tscal p12 = p1 * p1;
            Tscal p14 = p12 * p12;
            Tscal p17 = p12 * p14 * p1;

            if (q < 2.) {
                return div11_512 * p17 * q * p2;
            } else
                return 0;
        }

        inline static Tscal ddf(Tscal q) {
            Tscal t1 = sham::pow_constexpr<2>(q);
            Tscal t2 = sham::pow_constexpr<3>(q);
            Tscal t3 = sham::pow_constexpr<6>(1 - (1.0 / 2.0) * q);
            Tscal t4 = sham::pow_constexpr<7>(1 - (1.0 / 2.0) * q);
            Tscal t5 = sham::pow_constexpr<8>(1 - (1.0 / 2.0) * q);
            if (q < 2) {
                return t5 * (24 * q + 25.0 / 2.0) - 8 * t4 * (12 * t1 + (25.0 / 2.0) * q + 4)
                       + 14 * t3 * (4 * t2 + (25.0 / 4.0) * t1 + 4 * q + 1);
            } else
                return 0;
        }

        inline static Tscal phi_tilde_3d(Tscal q) {
            Tscal t1 = sham::pow_constexpr<10>(q);
            Tscal t2 = sham::pow_constexpr<11>(q);
            Tscal t3 = sham::pow_constexpr<2>(q);
            Tscal t4 = sham::pow_constexpr<4>(q);
            Tscal t5 = sham::pow_constexpr<6>(q);
            Tscal t6 = sham::pow_constexpr<7>(q);
            Tscal t7 = sham::pow_constexpr<8>(q);
            Tscal t8 = sham::pow_constexpr<9>(q);
            if (q < 2) {
                return (1.0 / 1397760.0) * shambase::constants::pi<Tscal>
                       * (t3
                              * (480 * t2 - 8085 * t1 + 58240 * t8 - 229320 * t7 + 512512 * t6
                                 - 560560 * t5 + 549120 * t4 - 768768 * t3 + 931840)
                          - 1003520);
            } else
                return -(512.0 / 1365.0) * shambase::constants::pi<Tscal> / q;
        }

        inline static Tscal phi_tilde_3d_prime(Tscal q) {
            Tscal t1  = sham::pow_constexpr<10>(q);
            Tscal t2  = sham::pow_constexpr<11>(q);
            Tscal t3  = sham::pow_constexpr<2>(q);
            Tscal t4  = sham::pow_constexpr<3>(q);
            Tscal t5  = sham::pow_constexpr<4>(q);
            Tscal t6  = sham::pow_constexpr<5>(q);
            Tscal t7  = sham::pow_constexpr<6>(q);
            Tscal t8  = sham::pow_constexpr<7>(q);
            Tscal t9  = sham::pow_constexpr<8>(q);
            Tscal t10 = sham::pow_constexpr<9>(q);
            if (q < 2) {
                return (1.0 / 1397760.0) * shambase::constants::pi<Tscal>
                       * (t3
                              * (5280 * t1 - 80850 * t10 + 524160 * t9 - 1834560 * t8 + 3587584 * t7
                                 - 3363360 * t6 + 2196480 * t4 - 1537536 * q)
                          + 2 * q
                                * (480 * t2 - 8085 * t1 + 58240 * t10 - 229320 * t9 + 512512 * t8
                                   - 560560 * t7 + 549120 * t5 - 768768 * t3 + 931840));
            } else
                return (512.0 / 1365.0) * shambase::constants::pi<Tscal> / t3;
        }
    };

    template<class Tscal>
    class KernelDefM4DoubleHump {
        public:
        inline static constexpr Tscal Rkern = 2; ///< Compact support radius of the kernel
        /// default hfact to be used for this kernel
        inline static constexpr Tscal hfactd = 1.2;

        /// 1D norm of the kernel
        inline static constexpr Tscal norm_1d = 2.;
        /// 2D norm of the kernel
        inline static constexpr Tscal norm_2d
            = (49. / 31.) * 10. / (7. * shambase::constants::pi<Tscal>);
        /// 3D norm of the kernel
        inline static constexpr Tscal norm_3d = (10. / 9.) * 1 / shambase::constants::pi<Tscal>;

        inline static Tscal f(Tscal q) { return KernelDefM4<Tscal>::f(q) * q * q; }

        inline static Tscal df(Tscal q) {
            return KernelDefM4<Tscal>::df(q) * q * q + 2 * KernelDefM4<Tscal>::f(q) * q;
        }

        inline static Tscal ddf(Tscal q) {
            Tscal t1 = sham::pow_constexpr<2>(1 - q);
            Tscal t2 = sham::pow_constexpr<2>(2 - q);
            Tscal t3 = sham::pow_constexpr<2>(q);
            Tscal t4 = sham::pow_constexpr<3>(1 - q);
            Tscal t5 = sham::pow_constexpr<3>(2 - q);
            if (q < 1) {
                return t3 * ((9.0 / 2.0) * q - 3) + 4 * q * (3 * t1 - (3.0 / 4.0) * t2) - 2 * t4
                       + (1.0 / 2.0) * t5;
            } else if (q < 2) {
                return -(3.0 / 4.0) * t3 * (2 * q - 4) - 3 * q * t2 + (1.0 / 2.0) * t5;
            } else
                return 0;
        }

        inline static Tscal phi_tilde_3d(Tscal q) {
            Tscal t1 = sham::pow_constexpr<2>(q);
            Tscal t2 = sham::pow_constexpr<3>(q);
            Tscal t3 = sham::pow_constexpr<4>(q);
            Tscal t4 = sham::pow_constexpr<5>(q);
            Tscal t5 = sham::pow_constexpr<6>(q);
            Tscal t6 = sham::pow_constexpr<7>(q);
            Tscal t7 = sham::pow_constexpr<8>(q);
            if (q < 1) {
                return (1.0 / 280.0) * shambase::constants::pi<Tscal>
                       * (t3 * (15 * t2 - 40 * t1 + 56) - 248);
            } else if (q < 2) {
                return (1.0 / 280.0) * shambase::constants::pi<Tscal>
                       * (-5 * t7 + 40 * t6 - 112 * t5 + 112 * t4 - 256 * q + 4) / q;
            } else
                return -(9.0 / 10.0) * shambase::constants::pi<Tscal> / q;
        }

        inline static Tscal phi_tilde_3d_prime(Tscal q) {
            Tscal t1 = sham::pow_constexpr<2>(q);
            Tscal t2 = sham::pow_constexpr<3>(q);
            Tscal t3 = sham::pow_constexpr<4>(q);
            Tscal t4 = sham::pow_constexpr<5>(q);
            Tscal t5 = sham::pow_constexpr<6>(q);
            Tscal t6 = sham::pow_constexpr<7>(q);
            Tscal t7 = sham::pow_constexpr<8>(q);
            if (q < 1) {
                return (1.0 / 280.0) * shambase::constants::pi<Tscal>
                       * (t3 * (45 * t1 - 80 * q) + 4 * t2 * (15 * t2 - 40 * t1 + 56));
            } else if (q < 2) {
                return (1.0 / 280.0) * shambase::constants::pi<Tscal>
                           * (-40 * t6 + 280 * t5 - 672 * t4 + 560 * t3 - 256) / q
                       - (1.0 / 280.0) * shambase::constants::pi<Tscal>
                             * (-5 * t7 + 40 * t6 - 112 * t5 + 112 * t4 - 256 * q + 4) / t1;
            } else
                return (9.0 / 10.0) * shambase::constants::pi<Tscal> / t1;
        }
    };

    template<class Tscal>
    class KernelDefM4DoubleHump3 {
        public:
        inline static constexpr Tscal Rkern = 2; ///< Compact support radius of the kernel
        /// default hfact to be used for this kernel
        inline static constexpr Tscal hfactd = 1.2;

        /// 1D norm of the kernel
        inline static constexpr Tscal norm_1d = (105. / 31.) * 2. / 3.;
        /// 2D norm of the kernel
        inline static constexpr Tscal norm_2d
            = (14. / 9.) * 10. / (7. * shambase::constants::pi<Tscal>);
        /// 3D norm of the kernel
        inline static constexpr Tscal norm_3d = (126. / 127.) * 1 / shambase::constants::pi<Tscal>;

        inline static Tscal f(Tscal q) { return KernelDefM4<Tscal>::f(q) * q * q * q; }

        inline static Tscal df(Tscal q) {
            return KernelDefM4<Tscal>::df(q) * q * q * q + 3 * KernelDefM4<Tscal>::f(q) * q * q;
        }

        inline static Tscal ddf(Tscal q) {
            Tscal t1 = sham::pow_constexpr<2>(1 - q);
            Tscal t2 = sham::pow_constexpr<2>(2 - q);
            Tscal t3 = sham::pow_constexpr<2>(q);
            Tscal t4 = sham::pow_constexpr<3>(1 - q);
            Tscal t5 = sham::pow_constexpr<3>(2 - q);
            Tscal t6 = sham::pow_constexpr<3>(q);
            if (q < 1) {
                return t6 * ((9.0 / 2.0) * q - 3) + 6 * t3 * (3 * t1 - (3.0 / 4.0) * t2)
                       + 6 * q * (-t4 + (1.0 / 4.0) * t5);
            } else if (q < 2) {
                return -(3.0 / 4.0) * t6 * (2 * q - 4) - (9.0 / 2.0) * t3 * t2
                       + (3.0 / 2.0) * q * t5;
            } else
                return 0;
        }

        inline static Tscal phi_tilde_3d(Tscal q) {
            Tscal t1 = sham::pow_constexpr<2>(q);
            Tscal t2 = sham::pow_constexpr<3>(q);
            Tscal t3 = sham::pow_constexpr<5>(q);
            Tscal t4 = sham::pow_constexpr<6>(q);
            Tscal t5 = sham::pow_constexpr<7>(q);
            Tscal t6 = sham::pow_constexpr<8>(q);
            Tscal t7 = sham::pow_constexpr<9>(q);
            if (q < 1) {
                return (1.0 / 840.0) * shambase::constants::pi<Tscal>
                       * (t3 * (35 * t2 - 90 * t1 + 112) - 756);
            } else if (q < 2) {
                return (1.0 / 2520.0) * shambase::constants::pi<Tscal>
                       * (-35 * t7 + 270 * t6 - 720 * t5 + 672 * t4 - 2304 * q + 20) / q;
            } else
                return -(127.0 / 126.0) * shambase::constants::pi<Tscal> / q;
        }

        inline static Tscal phi_tilde_3d_prime(Tscal q) {
            Tscal t1 = sham::pow_constexpr<2>(q);
            Tscal t2 = sham::pow_constexpr<3>(q);
            Tscal t3 = sham::pow_constexpr<4>(q);
            Tscal t4 = sham::pow_constexpr<5>(q);
            Tscal t5 = sham::pow_constexpr<6>(q);
            Tscal t6 = sham::pow_constexpr<7>(q);
            Tscal t7 = sham::pow_constexpr<8>(q);
            Tscal t8 = sham::pow_constexpr<9>(q);
            if (q < 1) {
                return (1.0 / 840.0) * shambase::constants::pi<Tscal>
                       * (t4 * (105 * t1 - 180 * q) + 5 * t3 * (35 * t2 - 90 * t1 + 112));
            } else if (q < 2) {
                return (1.0 / 2520.0) * shambase::constants::pi<Tscal>
                           * (-315 * t7 + 2160 * t6 - 5040 * t5 + 4032 * t4 - 2304) / q
                       - (1.0 / 2520.0) * shambase::constants::pi<Tscal>
                             * (-35 * t8 + 270 * t7 - 720 * t6 + 672 * t5 - 2304 * q + 20) / t1;
            } else
                return (127.0 / 126.0) * shambase::constants::pi<Tscal> / t1;
        }
    };

    template<class Tscal>
    class KernelDefM4DoubleHump5 {
        public:
        inline static constexpr Tscal Rkern = 2; ///< Compact support radius of the kernel
        /// default hfact to be used for this kernel
        inline static constexpr Tscal hfactd = 1.2;

        /// 1D norm of the kernel
        inline static constexpr Tscal norm_1d = 252.0 / 127.0;
        /// 2D norm of the kernel
        inline static constexpr Tscal norm_2d = (28.0 / 17.0) / shambase::constants::pi<Tscal>;
        /// 3D norm of the kernel
        inline static constexpr Tscal norm_3d = (330.0 / 511.0) / shambase::constants::pi<Tscal>;

        inline static Tscal f(Tscal q) {
            return KernelDefM4<Tscal>::f(q) * sham::pow_constexpr<5>(q);
        }

        inline static Tscal df(Tscal q) {
            return KernelDefM4<Tscal>::df(q) * sham::pow_constexpr<5>(q)
                   + 5 * KernelDefM4<Tscal>::f(q) * sham::pow_constexpr<4>(q);
        }

        inline static Tscal ddf(Tscal q) {
            Tscal t1 = sham::pow_constexpr<2>(1 - q);
            Tscal t2 = sham::pow_constexpr<2>(2 - q);
            Tscal t3 = sham::pow_constexpr<3>(1 - q);
            Tscal t4 = sham::pow_constexpr<3>(2 - q);
            Tscal t5 = sham::pow_constexpr<3>(q);
            Tscal t6 = sham::pow_constexpr<4>(q);
            Tscal t7 = sham::pow_constexpr<5>(q);
            if (q < 1) {
                return t7 * ((9.0 / 2.0) * q - 3) + 10 * t6 * (3 * t1 - (3.0 / 4.0) * t2)
                       + 20 * t5 * (-t3 + (1.0 / 4.0) * t4);
            } else if (q < 2) {
                return -(3.0 / 4.0) * t7 * (2 * q - 4) - (15.0 / 2.0) * t6 * t2 + 5 * t5 * t4;
            } else
                return 0;
        }

        inline static Tscal phi_tilde_3d(Tscal q) {
            Tscal t1 = sham::pow_constexpr<10>(q);
            Tscal t2 = sham::pow_constexpr<11>(q);
            Tscal t3 = sham::pow_constexpr<2>(q);
            Tscal t4 = sham::pow_constexpr<3>(q);
            Tscal t5 = sham::pow_constexpr<7>(q);
            Tscal t6 = sham::pow_constexpr<8>(q);
            Tscal t7 = sham::pow_constexpr<9>(q);
            if (q < 1) {
                return (1.0 / 2310.0) * shambase::constants::pi<Tscal>
                       * (t5 * (63 * t4 - 154 * t3 + 165) - 2805);
            } else if (q < 2) {
                return (1.0 / 2310.0) * shambase::constants::pi<Tscal>
                       * (-21 * t2 + 154 * t1 - 385 * t7 + 330 * t6 - 2816 * q + 7) / q;
            } else
                return -(511.0 / 330.0) * shambase::constants::pi<Tscal> / q;
        }

        inline static Tscal phi_tilde_3d_prime(Tscal q) {
            Tscal t1 = sham::pow_constexpr<10>(q);
            Tscal t2 = sham::pow_constexpr<11>(q);
            Tscal t3 = sham::pow_constexpr<2>(q);
            Tscal t4 = sham::pow_constexpr<3>(q);
            Tscal t5 = sham::pow_constexpr<6>(q);
            Tscal t6 = sham::pow_constexpr<7>(q);
            Tscal t7 = sham::pow_constexpr<8>(q);
            Tscal t8 = sham::pow_constexpr<9>(q);
            if (q < 1) {
                return (1.0 / 2310.0) * shambase::constants::pi<Tscal>
                       * (t6 * (189 * t3 - 308 * q) + 7 * t5 * (63 * t4 - 154 * t3 + 165));
            } else if (q < 2) {
                return (1.0 / 2310.0) * shambase::constants::pi<Tscal>
                           * (-231 * t1 + 1540 * t8 - 3465 * t7 + 2640 * t6 - 2816) / q
                       - (1.0 / 2310.0) * shambase::constants::pi<Tscal>
                             * (-21 * t2 + 154 * t1 - 385 * t8 + 330 * t7 - 2816 * q + 7) / t3;
            } else
                return (511.0 / 330.0) * shambase::constants::pi<Tscal> / t3;
        }
    };

    template<class Tscal>
    class KernelDefM4DoubleHump7 {
        public:
        inline static constexpr Tscal Rkern = 2; ///< Compact support radius of the kernel
        /// default hfact to be used for this kernel
        inline static constexpr Tscal hfactd = 1.2;

        /// 1D norm of the kernel
        inline static constexpr Tscal norm_1d = 660.0 / 511.0;
        /// 2D norm of the kernel
        inline static constexpr Tscal norm_2d = (30.0 / 31.0) / shambase::constants::pi<Tscal>;
        /// 3D norm of the kernel
        inline static constexpr Tscal norm_3d = (715.0 / 2047.0) / shambase::constants::pi<Tscal>;

        inline static Tscal f(Tscal q) {
            return KernelDefM4<Tscal>::f(q) * sham::pow_constexpr<7>(q);
        }

        inline static Tscal df(Tscal q) {
            return KernelDefM4<Tscal>::df(q) * sham::pow_constexpr<7>(q)
                   + 7 * KernelDefM4<Tscal>::f(q) * sham::pow_constexpr<6>(q);
        }

        inline static Tscal ddf(Tscal q) {
            Tscal t1 = sham::pow_constexpr<2>(1 - q);
            Tscal t2 = sham::pow_constexpr<2>(2 - q);
            Tscal t3 = sham::pow_constexpr<3>(1 - q);
            Tscal t4 = sham::pow_constexpr<3>(2 - q);
            Tscal t5 = sham::pow_constexpr<5>(q);
            Tscal t6 = sham::pow_constexpr<6>(q);
            Tscal t7 = sham::pow_constexpr<7>(q);
            if (q < 1) {
                return t7 * ((9.0 / 2.0) * q - 3) + 14 * t6 * (3 * t1 - (3.0 / 4.0) * t2)
                       + 42 * t5 * (-t3 + (1.0 / 4.0) * t4);
            } else if (q < 2) {
                return -(3.0 / 4.0) * t7 * (2 * q - 4) - (21.0 / 2.0) * t6 * t2
                       + (21.0 / 2.0) * t5 * t4;
            } else
                return 0;
        }

        inline static Tscal phi_tilde_3d(Tscal q) {
            Tscal t1 = sham::pow_constexpr<10>(q);
            Tscal t2 = sham::pow_constexpr<11>(q);
            Tscal t3 = sham::pow_constexpr<12>(q);
            Tscal t4 = sham::pow_constexpr<13>(q);
            Tscal t5 = sham::pow_constexpr<2>(q);
            Tscal t6 = sham::pow_constexpr<3>(q);
            Tscal t7 = sham::pow_constexpr<9>(q);
            if (q < 1) {
                return (1.0 / 25740.0) * shambase::constants::pi<Tscal>
                       * (t7 * (495 * t6 - 1170 * t5 + 1144) - 53196);
            } else if (q < 2) {
                return (1.0 / 25740.0) * shambase::constants::pi<Tscal>
                       * (-165 * t4 + 1170 * t3 - 2808 * t2 + 2288 * t1 - 53248 * q + 36) / q;
            } else
                return -(2047.0 / 715.0) * shambase::constants::pi<Tscal> / q;
        }

        inline static Tscal phi_tilde_3d_prime(Tscal q) {
            Tscal t1 = sham::pow_constexpr<10>(q);
            Tscal t2 = sham::pow_constexpr<11>(q);
            Tscal t3 = sham::pow_constexpr<12>(q);
            Tscal t4 = sham::pow_constexpr<13>(q);
            Tscal t5 = sham::pow_constexpr<2>(q);
            Tscal t6 = sham::pow_constexpr<3>(q);
            Tscal t7 = sham::pow_constexpr<8>(q);
            Tscal t8 = sham::pow_constexpr<9>(q);
            if (q < 1) {
                return (1.0 / 25740.0) * shambase::constants::pi<Tscal>
                       * (t8 * (1485 * t5 - 2340 * q) + 9 * t7 * (495 * t6 - 1170 * t5 + 1144));
            } else if (q < 2) {
                return (1.0 / 25740.0) * shambase::constants::pi<Tscal>
                           * (-2145 * t3 + 14040 * t2 - 30888 * t1 + 22880 * t8 - 53248) / q
                       - (1.0 / 25740.0) * shambase::constants::pi<Tscal>
                             * (-165 * t4 + 1170 * t3 - 2808 * t2 + 2288 * t1 - 53248 * q + 36)
                             / t5;
            } else
                return (2047.0 / 715.0) * shambase::constants::pi<Tscal> / t5;
        }
    };

    template<class Tscal>
    class KernelDefM4Shift2 {
        public:
        inline static constexpr Tscal Rkern = 2; ///< Compact support radius of the kernel
        /// default hfact to be used for this kernel
        inline static constexpr Tscal hfactd = 1.2;

        /// 1D norm of the kernel
        inline static constexpr Tscal norm_1d = 4.0 / 11.0;
        /// 2D norm of the kernel
        inline static constexpr Tscal norm_2d = (40.0 / 77.0) / shambase::constants::pi<Tscal>;
        /// 3D norm of the kernel
        inline static constexpr Tscal norm_3d = (120.0 / 439.0) / shambase::constants::pi<Tscal>;

        inline static Tscal f(Tscal q) {
            if (q < 1.) {
                return 1.;
            } else {
                return KernelDefM4<Tscal>::f((q - 1) * 2);
            }
        }

        inline static Tscal df(Tscal q) {
            if (q < 1.) {
                return 0.;
            } else {
                return 2 * KernelDefM4<Tscal>::df((q - 1) * 2);
            }
        }

        inline static Tscal ddf(Tscal q) {
            if (q < 1.) {
                return 0.;
            } else {
                return 4 * KernelDefM4<Tscal>::ddf((q - 1) * 2);
            }
        }

        inline static Tscal phi_tilde_3d(Tscal q) {
            Tscal t1 = sham::pow_constexpr<2>(q);
            Tscal t2 = sham::pow_constexpr<3>(q);
            Tscal t3 = sham::pow_constexpr<4>(q);
            Tscal t4 = sham::pow_constexpr<5>(q);
            Tscal t5 = sham::pow_constexpr<6>(q);
            if (q < 1) {
                return (1.0 / 60.0) * shambase::constants::pi<Tscal> * (40 * t1 - 231);
            } else if (q < 3.0 / 2.0) {
                return (1.0 / 60.0) * shambase::constants::pi<Tscal>
                       * (48 * t5 - 288 * t4 + 600 * t3 - 440 * t2 - 39 * q - 72) / q;
            } else if (q < 2) {
                return (1.0 / 120.0) * shambase::constants::pi<Tscal>
                       * (-32 * t5 + 288 * t4 - 960 * t3 + 1280 * t2 - 1536 * q + 585) / q;
            } else
                return -(439.0 / 120.0) * shambase::constants::pi<Tscal> / q;
        }

        inline static Tscal phi_tilde_3d_prime(Tscal q) {
            Tscal t1 = sham::pow_constexpr<2>(q);
            Tscal t2 = sham::pow_constexpr<3>(q);
            Tscal t3 = sham::pow_constexpr<4>(q);
            Tscal t4 = sham::pow_constexpr<5>(q);
            Tscal t5 = sham::pow_constexpr<6>(q);
            if (q < 1) {
                return (4.0 / 3.0) * shambase::constants::pi<Tscal> * q;
            } else if (q < 3.0 / 2.0) {
                return (1.0 / 60.0) * shambase::constants::pi<Tscal>
                           * (288 * t4 - 1440 * t3 + 2400 * t2 - 1320 * t1 - 39) / q
                       - (1.0 / 60.0) * shambase::constants::pi<Tscal>
                             * (48 * t5 - 288 * t4 + 600 * t3 - 440 * t2 - 39 * q - 72) / t1;
            } else if (q < 2) {
                return (1.0 / 120.0) * shambase::constants::pi<Tscal>
                           * (-192 * t4 + 1440 * t3 - 3840 * t2 + 3840 * t1 - 1536) / q
                       - (1.0 / 120.0) * shambase::constants::pi<Tscal>
                             * (-32 * t5 + 288 * t4 - 960 * t3 + 1280 * t2 - 1536 * q + 585) / t1;
            } else
                return (439.0 / 120.0) * shambase::constants::pi<Tscal> / t1;
        }
    };

    template<class Tscal>
    class KernelDefM4Shift4 {
        public:
        inline static constexpr Tscal Rkern = 2; ///< Compact support radius of the kernel
        /// default hfact to be used for this kernel
        inline static constexpr Tscal hfactd = 1.2;

        /// 1D norm of the kernel
        inline static constexpr Tscal norm_1d = 8.0 / 27.0;
        /// 2D norm of the kernel
        inline static constexpr Tscal norm_2d = (160.0 / 457.0) / shambase::constants::pi<Tscal>;
        /// 3D norm of the kernel
        inline static constexpr Tscal norm_3d = (320.0 / 2069.0) / shambase::constants::pi<Tscal>;

        inline static Tscal f(Tscal q) {
            if (q < 1.5) {
                return 1.;
            } else {
                return KernelDefM4<Tscal>::f((q - 1.5) * 4);
            }
        }

        inline static Tscal df(Tscal q) {
            if (q < 1.5) {
                return 0.;
            } else {
                return 4 * KernelDefM4<Tscal>::df((q - 1.5) * 4);
            }
        }

        inline static Tscal ddf(Tscal q) {
            if (q < 1.5) {
                return 0.;
            } else {
                return 16 * KernelDefM4<Tscal>::ddf((q - 1.5) * 4);
            }
        }

        inline static Tscal phi_tilde_3d(Tscal q) {
            Tscal t1 = sham::pow_constexpr<2>(q);
            Tscal t2 = sham::pow_constexpr<3>(q);
            Tscal t3 = sham::pow_constexpr<4>(q);
            Tscal t4 = sham::pow_constexpr<5>(q);
            Tscal t5 = sham::pow_constexpr<6>(q);
            if (q < 3.0 / 2.0) {
                return (1.0 / 240.0) * shambase::constants::pi<Tscal> * (160 * t1 - 1371);
            } else if (q < 7.0 / 4.0) {
                return (1.0 / 240.0) * shambase::constants::pi<Tscal>
                       * (1536 * t5 - 11520 * t4 + 31680 * t3 - 34400 * t2 + 25845 * q - 14580) / q;
            } else if (q < 2) {
                return (1.0 / 960.0) * shambase::constants::pi<Tscal>
                       * (-2048 * t5 + 18432 * t4 - 61440 * t3 + 81920 * t2 - 98304 * q + 59329)
                       / q;
            } else
                return -(2069.0 / 320.0) * shambase::constants::pi<Tscal> / q;
        }

        inline static Tscal phi_tilde_3d_prime(Tscal q) {
            Tscal t1 = sham::pow_constexpr<2>(q);
            Tscal t2 = sham::pow_constexpr<3>(q);
            Tscal t3 = sham::pow_constexpr<4>(q);
            Tscal t4 = sham::pow_constexpr<5>(q);
            Tscal t5 = sham::pow_constexpr<6>(q);
            if (q < 3.0 / 2.0) {
                return (4.0 / 3.0) * shambase::constants::pi<Tscal> * q;
            } else if (q < 7.0 / 4.0) {
                return (1.0 / 240.0) * shambase::constants::pi<Tscal>
                           * (9216 * t4 - 57600 * t3 + 126720 * t2 - 103200 * t1 + 25845) / q
                       - (1.0 / 240.0) * shambase::constants::pi<Tscal>
                             * (1536 * t5 - 11520 * t4 + 31680 * t3 - 34400 * t2 + 25845 * q
                                - 14580)
                             / t1;
            } else if (q < 2) {
                return (1.0 / 960.0) * shambase::constants::pi<Tscal>
                           * (-12288 * t4 + 92160 * t3 - 245760 * t2 + 245760 * t1 - 98304) / q
                       - (1.0 / 960.0) * shambase::constants::pi<Tscal>
                             * (-2048 * t5 + 18432 * t4 - 61440 * t3 + 81920 * t2 - 98304 * q
                                + 59329)
                             / t1;
            } else
                return (2069.0 / 320.0) * shambase::constants::pi<Tscal> / t1;
        }
    };

    template<class Tscal>
    class KernelDefM4Shift8 {
        public:
        inline static constexpr Tscal Rkern = 2; ///< Compact support radius of the kernel
        /// default hfact to be used for this kernel
        inline static constexpr Tscal hfactd = 1.2;

        /// 1D norm of the kernel
        inline static constexpr Tscal norm_1d = 16.0 / 59.0;
        /// 2D norm of the kernel
        inline static constexpr Tscal norm_2d = (640.0 / 2177.0) / shambase::constants::pi<Tscal>;
        /// 3D norm of the kernel
        inline static constexpr Tscal norm_3d = (7680.0 / 64303.0) / shambase::constants::pi<Tscal>;

        inline static Tscal f(Tscal q) {
            if (q < 1.75) {
                return 1.;
            } else {
                return KernelDefM4<Tscal>::f((q - 1.75) * 8);
            }
        }

        inline static Tscal df(Tscal q) {
            if (q < 1.75) {
                return 0.;
            } else {
                return 8 * KernelDefM4<Tscal>::df((q - 1.75) * 8);
            }
        }

        inline static Tscal ddf(Tscal q) {
            if (q < 1.75) {
                return 0.;
            } else {
                return 64 * KernelDefM4<Tscal>::ddf((q - 1.75) * 8);
            }
        }

        inline static Tscal phi_tilde_3d(Tscal q) {
            Tscal t1 = sham::pow_constexpr<2>(q);
            Tscal t2 = sham::pow_constexpr<3>(q);
            Tscal t3 = sham::pow_constexpr<4>(q);
            Tscal t4 = sham::pow_constexpr<5>(q);
            Tscal t5 = sham::pow_constexpr<6>(q);
            if (q < 7.0 / 4.0) {
                return (1.0 / 960.0) * shambase::constants::pi<Tscal> * (640 * t1 - 6531);
            } else if (q < 15.0 / 8.0) {
                return (1.0 / 960.0) * shambase::constants::pi<Tscal>
                       * (49152 * t5 - 405504 * t4 + 1236480 * t3 - 1504640 * t2 + 1491693 * q
                          - 907578)
                       / q;
            } else if (q < 2) {
                return (1.0 / 7680.0) * shambase::constants::pi<Tscal>
                       * (-131072 * t5 + 1179648 * t4 - 3932160 * t3 + 5242880 * t2 - 6291456 * q
                          + 4130001)
                       / q;
            } else
                return -(64303.0 / 7680.0) * shambase::constants::pi<Tscal> / q;
        }

        inline static Tscal phi_tilde_3d_prime(Tscal q) {
            Tscal t1 = sham::pow_constexpr<2>(q);
            Tscal t2 = sham::pow_constexpr<3>(q);
            Tscal t3 = sham::pow_constexpr<4>(q);
            Tscal t4 = sham::pow_constexpr<5>(q);
            Tscal t5 = sham::pow_constexpr<6>(q);
            if (q < 7.0 / 4.0) {
                return (4.0 / 3.0) * shambase::constants::pi<Tscal> * q;
            } else if (q < 15.0 / 8.0) {
                return (1.0 / 960.0) * shambase::constants::pi<Tscal>
                           * (294912 * t4 - 2027520 * t3 + 4945920 * t2 - 4513920 * t1 + 1491693)
                           / q
                       - (1.0 / 960.0) * shambase::constants::pi<Tscal>
                             * (49152 * t5 - 405504 * t4 + 1236480 * t3 - 1504640 * t2 + 1491693 * q
                                - 907578)
                             / t1;
            } else if (q < 2) {
                return (1.0 / 7680.0) * shambase::constants::pi<Tscal>
                           * (-786432 * t4 + 5898240 * t3 - 15728640 * t2 + 15728640 * t1 - 6291456)
                           / q
                       - (1.0 / 7680.0) * shambase::constants::pi<Tscal>
                             * (-131072 * t5 + 1179648 * t4 - 3932160 * t3 + 5242880 * t2
                                - 6291456 * q + 4130001)
                             / t1;
            } else
                return (64303.0 / 7680.0) * shambase::constants::pi<Tscal> / t1;
        }
    };

    template<class Tscal>
    class KernelDefM4Shift16 {
        public:
        inline static constexpr Tscal Rkern = 2; ///< Compact support radius of the kernel
        /// default hfact to be used for this kernel
        inline static constexpr Tscal hfactd = 1.2;

        /// 1D norm of the kernel
        inline static constexpr Tscal norm_1d = 32.0 / 123.0;
        /// 2D norm of the kernel
        inline static constexpr Tscal norm_2d = (2560.0 / 9457.0) / shambase::constants::pi<Tscal>;
        /// 3D norm of the kernel
        inline static constexpr Tscal norm_3d = (4096.0 / 38785.0) / shambase::constants::pi<Tscal>;

        inline static Tscal f(Tscal q) {
            if (q < 1.875) {
                return 1.;
            } else {
                return KernelDefM4<Tscal>::f((q - 1.875) * 16);
            }
        }

        inline static Tscal df(Tscal q) {
            if (q < 1.875) {
                return 0.;
            } else {
                return 16 * KernelDefM4<Tscal>::df((q - 1.875) * 16);
            }
        }

        inline static Tscal ddf(Tscal q) {
            if (q < 1.875) {
                return 0.;
            } else {
                return 256 * KernelDefM4<Tscal>::ddf((q - 1.875) * 16);
            }
        }

        inline static Tscal phi_tilde_3d(Tscal q) {
            Tscal t1 = sham::pow_constexpr<2>(q);
            Tscal t2 = sham::pow_constexpr<3>(q);
            Tscal t3 = sham::pow_constexpr<4>(q);
            Tscal t4 = sham::pow_constexpr<5>(q);
            Tscal t5 = sham::pow_constexpr<6>(q);
            if (q < 15.0 / 8.0) {
                return (1.0 / 3840.0) * shambase::constants::pi<Tscal> * (2560 * t1 - 28371);
            } else if (q < 31.0 / 16.0) {
                return (1.0 / 3840.0) * shambase::constants::pi<Tscal>
                       * (1572864 * t5 - 13565952 * t4 + 43315200 * t3 - 55293440 * t2
                          + 60721629 * q - 38728125)
                       / q;
            } else if (q < 2) {
                return (1.0 / 61440.0) * shambase::constants::pi<Tscal>
                       * (-8388608 * t5 + 75497472 * t4 - 251658240 * t3 + 335544320 * t2
                          - 402653184 * q + 267853681)
                       / q;
            } else
                return -(38785.0 / 4096.0) * shambase::constants::pi<Tscal> / q;
        }

        inline static Tscal phi_tilde_3d_prime(Tscal q) {
            Tscal t1 = sham::pow_constexpr<2>(q);
            Tscal t2 = sham::pow_constexpr<3>(q);
            Tscal t3 = sham::pow_constexpr<4>(q);
            Tscal t4 = sham::pow_constexpr<5>(q);
            Tscal t5 = sham::pow_constexpr<6>(q);
            if (q < 15.0 / 8.0) {
                return (4.0 / 3.0) * shambase::constants::pi<Tscal> * q;
            } else if (q < 31.0 / 16.0) {
                return (1.0 / 3840.0) * shambase::constants::pi<Tscal>
                           * (9437184 * t4 - 67829760 * t3 + 173260800 * t2 - 165880320 * t1
                              + 60721629)
                           / q
                       - (1.0 / 3840.0) * shambase::constants::pi<Tscal>
                             * (1572864 * t5 - 13565952 * t4 + 43315200 * t3 - 55293440 * t2
                                + 60721629 * q - 38728125)
                             / t1;
            } else if (q < 2) {
                return (1.0 / 61440.0) * shambase::constants::pi<Tscal>
                           * (-50331648 * t4 + 377487360 * t3 - 1006632960 * t2 + 1006632960 * t1
                              - 402653184)
                           / q
                       - (1.0 / 61440.0) * shambase::constants::pi<Tscal>
                             * (-8388608 * t5 + 75497472 * t4 - 251658240 * t3 + 335544320 * t2
                                - 402653184 * q + 267853681)
                             / t1;
            } else
                return (38785.0 / 4096.0) * shambase::constants::pi<Tscal> / t1;
        }
    };
} // namespace shammath::details

namespace shammath {

    // Type trait to detect if BaseKernel has phi_tilde_3d method
    template<typename T, typename Tscal, typename = void>
    struct has_phi_tilde_3d : std::false_type {};

    template<typename T, typename Tscal>
    struct has_phi_tilde_3d<T, Tscal, std::void_t<decltype(T::phi_tilde_3d(std::declval<Tscal>()))>>
        : std::true_type {};

    // Type trait to detect if BaseKernel has phi_tilde_3d_prime method
    template<typename T, typename Tscal, typename = void>
    struct has_phi_tilde_3d_prime : std::false_type {};

    template<typename T, typename Tscal>
    struct has_phi_tilde_3d_prime<
        T,
        Tscal,
        std::void_t<decltype(T::phi_tilde_3d_prime(std::declval<Tscal>()))>> : std::true_type {};

    template<class Tscal_, class BaseKernel>
    class SPHKernelGen {
        public:
        using Generator                     = BaseKernel;
        using Tscal                         = Tscal_;
        inline static constexpr Tscal Rkern = BaseKernel::Rkern; /*!< Radius of the support */
        inline static constexpr Tscal hfactd
            = BaseKernel::hfactd; /*!< default $h_{\rm fact}$ for this kernel*/

        /**
         * @brief the base function for this SPH kernel
         *
         * Such that : \f$\int f(\vert\vert q\vert\vert) {\rm d}^n \mathbf{q} = 1\f$
         * Also : \f$ f(q > Rkern) = 0\f$
         * @param q parameter of the function
         * @return Tscal the value of \f$ f(q) \f$
         */
        inline static Tscal f(Tscal q) { return BaseKernel::f(q); }

        /**
         * @brief derivative of \ref M4.f
         *
         * @param q parameter of the function
         * @return Tscal the value of \f$ f'(q) \f$
         */
        inline static Tscal df(Tscal q) { return BaseKernel::df(q); }

        inline static Tscal ddf(Tscal q) { return BaseKernel::ddf(q); }

        inline static Tscal W_1d(Tscal r, Tscal h) { return BaseKernel::norm_1d * f(r / h) / (h); }

        inline static Tscal W_2d(Tscal r, Tscal h) {
            return BaseKernel::norm_2d * f(r / h) / (h * h);
        }

        /**
         * @brief compute the normed & resized version of the kernel :
         * \f[
         *  W(r,h) = C_{\rm norm} \frac{1}{h^3} f(\frac{r}{h})
         * \f]
         * @param r
         * @param h
         * @return Tscal
         */
        inline static Tscal W_3d(Tscal r, Tscal h) {
            return BaseKernel::norm_3d * f(r / h) / (h * h * h);
        }

        inline static Tscal dW_3d(Tscal r, Tscal h) {
            return BaseKernel::norm_3d * df(r / h) / (h * h * h * h);
        }

        inline static Tscal ddW_3d(Tscal r, Tscal h) {
            return BaseKernel::norm_3d * ddf(r / h) / (h * h * h * h * h);
        }

        inline static Tscal dhW_3d(Tscal r, Tscal h) {
            return -(BaseKernel::norm_3d) * (3 * f(r / h) + (r / h) * df(r / h)) / (h * h * h * h);
        }

        inline static Tscal f3d_integ_z(Tscal x, int np = 32) {
            return integ_riemann_sum<Tscal>(-Rkern, Rkern, Rkern / np, [&](Tscal z) {
                return f(sqrt(x * x + z * z));
            });
        }

        inline static Tscal Y_3d(Tscal r, Tscal h, int np = 32) {
            return BaseKernel::norm_3d * f3d_integ_z(r / h, np) / (h * h);
        }

        static constexpr bool has_3d_phi_soft
            = ::shammath::has_phi_tilde_3d<BaseKernel, Tscal>::value;

        inline static Tscal phi_tilde_3d(Tscal q) {
            if constexpr (has_3d_phi_soft) {
                return BaseKernel::phi_tilde_3d(q);
            } else {
                return std::numeric_limits<Tscal>::quiet_NaN();
            }
        }

        static constexpr bool has_3d_phi_soft_prime
            = ::shammath::has_phi_tilde_3d_prime<BaseKernel, Tscal>::value;

        inline static Tscal phi_tilde_3d_prime(Tscal q) {
            if constexpr (has_3d_phi_soft_prime) {
                return BaseKernel::phi_tilde_3d_prime(q);
            } else {
                return std::numeric_limits<Tscal>::quiet_NaN();
            }
        }

        inline static Tscal phi_3D(Tscal r, Tscal h) {
            return BaseKernel::norm_3d * phi_tilde_3d(r / h) / h;
        }

        inline static Tscal phi_3D_prime(Tscal r, Tscal h) {
            return BaseKernel::norm_3d * phi_tilde_3d_prime(r / h) / (h * h);
        }
    };

    /**
     * @brief The M4 SPH kernel
     * \todo add graph
     *
     * @tparam flt_type the flating point representation to use
     */
    template<class flt_type>
    using M4 = SPHKernelGen<flt_type, details::KernelDefM4<flt_type>>;

    /**
     * @brief The M5 SPH kernel
     * \todo add graph
     *
     * @tparam flt_type the flating point representation to use
     */
    template<class flt_type>
    using M5 = SPHKernelGen<flt_type, details::KernelDefM5<flt_type>>;

    /**
     * @brief The M6 SPH kernel
     * \todo add graph
     *
     * @tparam flt_type the flating point representation to use
     */
    template<class flt_type>
    using M6 = SPHKernelGen<flt_type, details::KernelDefM6<flt_type>>;

    /**
     * @brief The M7 SPH kernel
     * \todo add graph
     *
     * @tparam flt_type the flating point representation to use
     */
    template<class flt_type>
    using M7 = SPHKernelGen<flt_type, details::KernelDefM7<flt_type>>;

    /**
     * @brief The M8 SPH kernel
     * \todo add graph
     *
     * @tparam flt_type the flating point representation to use
     */
    template<class flt_type>
    using M8 = SPHKernelGen<flt_type, details::KernelDefM8<flt_type>>;

    /**
     * @brief The M8 SPH kernel
     * \todo add graph
     *
     * @tparam flt_type the flating point representation to use
     */
    template<class flt_type>
    using M9 = SPHKernelGen<flt_type, details::KernelDefM9<flt_type>>;

    /**
     * @brief The M8 SPH kernel
     * \todo add graph
     *
     * @tparam flt_type the flating point representation to use
     */
    template<class flt_type>
    using M10 = SPHKernelGen<flt_type, details::KernelDefM10<flt_type>>;

    /**
     * @brief The C2 SPH kernel
     * \todo add graph
     *
     * @tparam flt_type the flating point representation to use
     */
    template<class flt_type>
    using C2 = SPHKernelGen<flt_type, details::KernelDefC2<flt_type>>;

    /**
     * @brief The C4 SPH kernel
     * \todo add graph
     *
     * @tparam flt_type the flating point representation to use
     */
    template<class flt_type>
    using C4 = SPHKernelGen<flt_type, details::KernelDefC4<flt_type>>;

    /**
     * @brief The C6 SPH kernel
     * \todo add graph
     *
     * @tparam flt_type the flating point representation to use
     */
    template<class flt_type>
    using C6 = SPHKernelGen<flt_type, details::KernelDefC6<flt_type>>;

    /**
     * @brief The M4DoubleHump SPH kernel
     * \todo add graph
     *
     * @tparam flt_type the flating point representation to use
     */
    template<class flt_type>
    using M4DH = SPHKernelGen<flt_type, details::KernelDefM4DoubleHump<flt_type>>;

    /**
     * @brief The M4DoubleHump3 SPH kernel
     * \todo add graph
     *
     * @tparam flt_type the flating point representation to use
     */
    template<class flt_type>
    using M4DH3 = SPHKernelGen<flt_type, details::KernelDefM4DoubleHump3<flt_type>>;

    /**
     * @brief The M4DoubleHump5 SPH kernel
     * \todo add graph
     *
     * @tparam flt_type the flating point representation to use
     */
    template<class flt_type>
    using M4DH5 = SPHKernelGen<flt_type, details::KernelDefM4DoubleHump5<flt_type>>;

    /**
     * @brief The M4DoubleHump7 SPH kernel
     * \todo add graph
     *
     * @tparam flt_type the flating point representation to use
     */
    template<class flt_type>
    using M4DH7 = SPHKernelGen<flt_type, details::KernelDefM4DoubleHump7<flt_type>>;

    /**
     * @brief The M4Shift2 SPH kernel
     * \todo add graph
     *
     * @tparam flt_type the flating point representation to use
     */
    template<class flt_type>
    using M4Shift2 = SPHKernelGen<flt_type, details::KernelDefM4Shift2<flt_type>>;

    /**
     * @brief The M4Shift4 SPH kernel
     * \todo add graph
     *
     * @tparam flt_type the flating point representation to use
     */
    template<class flt_type>
    using M4Shift4 = SPHKernelGen<flt_type, details::KernelDefM4Shift4<flt_type>>;

    /**
     * @brief The M4Shift8 SPH kernel
     * \todo add graph
     *
     * @tparam flt_type the flating point representation to use
     */
    template<class flt_type>
    using M4Shift8 = SPHKernelGen<flt_type, details::KernelDefM4Shift8<flt_type>>;

    /**
     * @brief The M4Shift16 SPH kernel
     * \todo add graph
     *
     * @tparam flt_type the flating point representation to use
     */
    template<class flt_type>
    using M4Shift16 = SPHKernelGen<flt_type, details::KernelDefM4Shift16<flt_type>>;

} // namespace shammath

namespace shambase {

    template<class flt_type>
    struct TypeNameInfo<shammath::M4<flt_type>> {
        inline static const std::string name = "M4<" + get_type_name<flt_type>() + ">";
    };
    template<class flt_type>
    struct TypeNameInfo<shammath::M5<flt_type>> {
        inline static const std::string name = "M5<" + get_type_name<flt_type>() + ">";
    };
    template<class flt_type>
    struct TypeNameInfo<shammath::M6<flt_type>> {
        inline static const std::string name = "M6<" + get_type_name<flt_type>() + ">";
    };
    template<class flt_type>
    struct TypeNameInfo<shammath::M7<flt_type>> {
        inline static const std::string name = "M7<" + get_type_name<flt_type>() + ">";
    };
    template<class flt_type>
    struct TypeNameInfo<shammath::M8<flt_type>> {
        inline static const std::string name = "M8<" + get_type_name<flt_type>() + ">";
    };
    template<class flt_type>
    struct TypeNameInfo<shammath::M9<flt_type>> {
        inline static const std::string name = "M9<" + get_type_name<flt_type>() + ">";
    };
    template<class flt_type>
    struct TypeNameInfo<shammath::M10<flt_type>> {
        inline static const std::string name = "M10<" + get_type_name<flt_type>() + ">";
    };

    template<class flt_type>
    struct TypeNameInfo<shammath::C2<flt_type>> {
        inline static const std::string name = "C2<" + get_type_name<flt_type>() + ">";
    };
    template<class flt_type>
    struct TypeNameInfo<shammath::C4<flt_type>> {
        inline static const std::string name = "C4<" + get_type_name<flt_type>() + ">";
    };
    template<class flt_type>
    struct TypeNameInfo<shammath::C6<flt_type>> {
        inline static const std::string name = "C6<" + get_type_name<flt_type>() + ">";
    };

    template<class flt_type>
    struct TypeNameInfo<shammath::M4DH<flt_type>> {
        inline static const std::string name = "M4DH<" + get_type_name<flt_type>() + ">";
    };

    template<class flt_type>
    struct TypeNameInfo<shammath::M4DH3<flt_type>> {
        inline static const std::string name = "M4DH3<" + get_type_name<flt_type>() + ">";
    };

    template<class flt_type>
    struct TypeNameInfo<shammath::M4DH5<flt_type>> {
        inline static const std::string name = "M4DH5<" + get_type_name<flt_type>() + ">";
    };

    template<class flt_type>
    struct TypeNameInfo<shammath::M4DH7<flt_type>> {
        inline static const std::string name = "M4DH7<" + get_type_name<flt_type>() + ">";
    };

    template<class flt_type>
    struct TypeNameInfo<shammath::M4Shift2<flt_type>> {
        inline static const std::string name = "M4Shift2<" + get_type_name<flt_type>() + ">";
    };

    template<class flt_type>
    struct TypeNameInfo<shammath::M4Shift4<flt_type>> {
        inline static const std::string name = "M4Shift4<" + get_type_name<flt_type>() + ">";
    };

    template<class flt_type>
    struct TypeNameInfo<shammath::M4Shift8<flt_type>> {
        inline static const std::string name = "M4Shift8<" + get_type_name<flt_type>() + ">";
    };

    template<class flt_type>
    struct TypeNameInfo<shammath::M4Shift16<flt_type>> {
        inline static const std::string name = "M4Shift16<" + get_type_name<flt_type>() + ">";
    };

} // namespace shambase
