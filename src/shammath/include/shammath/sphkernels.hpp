// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright(C) 2021-2023 Timothée David--Cléris <timothee.david--cleris@ens-lyon.fr>
// Licensed under CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

#pragma once

/**
 * @file sphkernels.hpp
 * @author Timothée David--Cléris (timothee.david--cleris@ens-lyon.fr)
 * @brief sph kernels
 */

//%Impl status : Good

#include "shambase/constants.hpp"

namespace shammath::details {

    template<class Tscal>
    class KernelDefM4 {
        public:
        inline static constexpr Tscal Rkern  = 2;
        inline static constexpr Tscal hfactd = 1.2;

        inline static constexpr Tscal norm_1d = 2. / 3.;
        inline static constexpr Tscal norm_2d = 10. / (7. * shambase::constants::pi<Tscal>);
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
    };

    template<class Tscal>
    class KernelDefM5 {
        public:
        inline static constexpr Tscal Rkern  = 5. / 2.;
        inline static constexpr Tscal hfactd = 1.2;

        inline static constexpr Tscal norm_1d = 1. / 24.;
        inline static constexpr Tscal norm_2d = 96. / (1199 * shambase::constants::pi<Tscal>);
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
    };

    template<class Tscal>
    class KernelDefM6 {
        public:
        inline static constexpr Tscal Rkern  = 3;
        inline static constexpr Tscal hfactd = 1.0;

        inline static constexpr Tscal norm_1d = 1. / 120.;
        inline static constexpr Tscal norm_2d = 7. / (478 * shambase::constants::pi<Tscal>);
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
    };



    template<class Tscal>
    class KernelDefM7 {
        public:
        inline static constexpr Tscal Rkern  = 3.5;
        inline static constexpr Tscal hfactd = 1.0;

        inline static constexpr Tscal norm_1d = 1. / 120.;
        inline static constexpr Tscal norm_2d = 7. / (478 * shambase::constants::pi<Tscal>);
        inline static constexpr Tscal norm_3d = 1 / (120 * shambase::constants::pi<Tscal>);

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
            t1 = t1 * (t1*t1);
            t2 = t2 * (t2*t2);
            t3 = t3 * (t3*t3);
            t4 = t4 * (t4*t4);

            t1 *= t1;
            t2 *= t2;
            t3 *= t3;
            t4 *= t4;

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
    };

    template<class Tscal>
    class KernelDefC2 {
        public:
        inline static constexpr Tscal Rkern  = 2;
        inline static constexpr Tscal hfactd = 1.0;

        inline static constexpr Tscal norm_1d = 3. / 4.;
        inline static constexpr Tscal norm_2d = 7. / (4 * shambase::constants::pi<Tscal>);
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
    };

    template<class Tscal>
    class KernelDefC4 {
        public:
        inline static constexpr Tscal Rkern  = 2;
        inline static constexpr Tscal hfactd = 1.0;

        inline static constexpr Tscal norm_1d = 27. / 32.;
        inline static constexpr Tscal norm_2d = 9. / (4 * shambase::constants::pi<Tscal>);
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
    };

    template<class Tscal>
    class KernelDefC6 {
        public:
        inline static constexpr Tscal Rkern  = 2;
        inline static constexpr Tscal hfactd = 1.0;

        inline static constexpr Tscal norm_1d = 15. / 16.;
        inline static constexpr Tscal norm_2d = 39. / (14. * shambase::constants::pi<Tscal>);
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
    };
} // namespace shammath::details

namespace shammath {

    template<class Tscal_, class BaseKernel>
    class SPHKernelGen {
        public:
        using Generator                     = BaseKernel;
        using Tscal                         = Tscal_;
        inline static constexpr Tscal Rkern = BaseKernel::Rkern; /*!< Radius of the support */
        inline static constexpr Tscal hfactd =
            BaseKernel::hfactd; /*!< default $h_{\rm fact}$ for this kernel*/

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

} // namespace shammath