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
            if (q < 1 / 2) {
                return 120 * t1 - 60 * t2 + 12 * t3;
            } else if (q < 3 / 2) {
                return -60 * t2 + 12 * t3;
            } else if (q < 5 / 2) {
                return 12 * t3;
            } else
                return 0;
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
            if (q < 1 / 2) {
                return -1050 * t1 + 630 * t2 - 210 * t3 + 30 * t4;
            } else if (q < 3 / 2) {
                return 630 * t2 - 210 * t3 + 30 * t4;
            } else if (q < 5 / 2) {
                return -210 * t3 + 30 * t4;
            } else if (q < 7 / 2) {
                return 30 * t4;
            } else
                return 0;
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
            if (q < 1 / 2) {
                return 7056 * t1 - 4704 * t2 + 2016 * t3 - 504 * t4 + 56 * t5;
            } else if (q < 3 / 2) {
                return -4704 * t2 + 2016 * t3 - 504 * t4 + 56 * t5;
            } else if (q < 5 / 2) {
                return 2016 * t3 - 504 * t4 + 56 * t5;
            } else if (q < 7 / 2) {
                return -504 * t4 + 56 * t5;
            } else if (q < 9 / 2) {
                return 56 * t5;
            } else
                return 0;
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
    };

    template<class Tscal>
    class KernelDefC4 {
        public:
        inline static constexpr Tscal Rkern = 2; ///< Compact support radius of the kernel
        /// default hfact to be used for this kernel
        inline static constexpr Tscal hfactd = 1.0;

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
    };

    template<class Tscal>
    class KernelDefM4DoubleHump5 {
        public:
        inline static constexpr Tscal Rkern = 2; ///< Compact support radius of the kernel
        /// default hfact to be used for this kernel
        inline static constexpr Tscal hfactd = 1.2;

        /// 1D norm of the kernel
        inline static constexpr Tscal norm_1d = 1. / 0.5039682539663191;
        /// 2D norm of the kernel
        inline static constexpr Tscal norm_2d = 1. / 1.9073955396705662;
        /// 3D norm of the kernel
        inline static constexpr Tscal norm_3d = 1. / 4.864708624169304;

        inline static Tscal f(Tscal q) {
            return KernelDefM4<Tscal>::f(q) * sham::pow_constexpr<5>(q);
        }

        inline static Tscal df(Tscal q) {
            return KernelDefM4<Tscal>::df(q) * sham::pow_constexpr<5>(q)
                   + 5 * KernelDefM4<Tscal>::f(q) * sham::pow_constexpr<4>(q);
        }
    };

    template<class Tscal>
    class KernelDefM4DoubleHump7 {
        public:
        inline static constexpr Tscal Rkern = 2; ///< Compact support radius of the kernel
        /// default hfact to be used for this kernel
        inline static constexpr Tscal hfactd = 1.2;

        /// 1D norm of the kernel
        inline static constexpr Tscal norm_1d = 1. / 0.7742424242383008;
        /// 2D norm of the kernel
        inline static constexpr Tscal norm_2d = 1. / 3.246312408690764;
        /// 3D norm of the kernel
        inline static constexpr Tscal norm_3d = 1. / 8.99418204455883;

        inline static Tscal f(Tscal q) {
            return KernelDefM4<Tscal>::f(q) * sham::pow_constexpr<7>(q);
        }

        inline static Tscal df(Tscal q) {
            return KernelDefM4<Tscal>::df(q) * sham::pow_constexpr<7>(q)
                   + 7 * KernelDefM4<Tscal>::f(q) * sham::pow_constexpr<6>(q);
        }
    };

    template<class Tscal>
    class KernelDefM4Shift2 {
        public:
        inline static constexpr Tscal Rkern = 2; ///< Compact support radius of the kernel
        /// default hfact to be used for this kernel
        inline static constexpr Tscal hfactd = 1.2;

        /// 1D norm of the kernel
        inline static constexpr Tscal norm_1d = 1. / 2.750009999995089;
        /// 2D norm of the kernel
        inline static constexpr Tscal norm_2d = 1. / 6.047565858099014;
        /// 3D norm of the kernel
        inline static constexpr Tscal norm_3d = 1. / 11.49299312435096;

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
    };

    template<class Tscal>
    class KernelDefM4Shift4 {
        public:
        inline static constexpr Tscal Rkern = 2; ///< Compact support radius of the kernel
        /// default hfact to be used for this kernel
        inline static constexpr Tscal hfactd = 1.2;

        /// 1D norm of the kernel
        inline static constexpr Tscal norm_1d = 1. / 3.3750099999975305;
        /// 2D norm of the kernel
        inline static constexpr Tscal norm_2d = 1. / 8.973174016735282;
        /// 3D norm of the kernel
        inline static constexpr Tscal norm_3d = 1. / 20.312360000777;

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
    };

    template<class Tscal>
    class KernelDefM4Shift8 {
        public:
        inline static constexpr Tscal Rkern = 2; ///< Compact support radius of the kernel
        /// default hfact to be used for this kernel
        inline static constexpr Tscal hfactd = 1.2;

        /// 1D norm of the kernel
        inline static constexpr Tscal norm_1d = 1. / 3.687509999998779;
        /// 2D norm of the kernel
        inline static constexpr Tscal norm_2d = 1. / 10.686323760634679;
        /// 3D norm of the kernel
        inline static constexpr Tscal norm_3d = 1. / 26.303884427447596;

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
    };

    template<class Tscal>
    class KernelDefM4Shift16 {
        public:
        inline static constexpr Tscal Rkern = 2; ///< Compact support radius of the kernel
        /// default hfact to be used for this kernel
        inline static constexpr Tscal hfactd = 1.2;

        /// 1D norm of the kernel
        inline static constexpr Tscal norm_1d = 1. / 3.8437599999993775;
        /// 2D norm of the kernel
        inline static constexpr Tscal norm_2d = 1. / 11.60548504872985;
        /// 3D norm of the kernel
        inline static constexpr Tscal norm_3d = 1. / 29.747722428921268;

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
    };
} // namespace shammath::details

namespace shammath {

    // Type trait to detect if BaseKernel has phi_tilde_3d method
    template<typename T, typename Tscal, typename = void>
    struct has_phi_tilde_3d : std::false_type {};

    template<typename T, typename Tscal>
    struct has_phi_tilde_3d<T, Tscal, std::void_t<decltype(T::phi_tilde_3d(std::declval<Tscal>()))>>
        : std::true_type {};

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

        static constexpr bool has_3d_softening
            = ::shammath::has_phi_tilde_3d<BaseKernel, Tscal>::value;

        inline static Tscal phi_tilde_3d(Tscal q) {
            if constexpr (has_3d_softening) {
                return BaseKernel::phi_tilde_3d(q);
            } else {
                return std::numeric_limits<Tscal>::quiet_NaN();
            }
        }

        inline static Tscal phi_3D(Tscal r, Tscal h) {
            return BaseKernel::norm_3d * phi_tilde_3d(r / h) / h;
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
