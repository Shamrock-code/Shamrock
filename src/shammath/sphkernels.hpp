// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright(C) 2021-2023 Timothée David--Cléris <timothee.david--cleris@ens-lyon.fr>
// Licensed under CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

#pragma once

/**
 * @file kernels.hpp
 * @author Timothée David--Cléris (timothee.david--cleris@ens-lyon.fr)
 * @brief sph kernels
 * @date 2023-05-10
 */

//%Impl status : Good

#include "aliases.hpp"
#include "shambase/Constants.hpp"

namespace shammath::details {

    template<class Tscal>
    class KernelDefM4 {
        public:
        inline static constexpr Tscal Rkern  = 2;   
        inline static constexpr Tscal hfactd = 1.2; 

        inline static constexpr Tscal norm_3d = 1 / shambase::Constants<Tscal>::pi;

        inline static Tscal f(Tscal q) {

            constexpr Tscal div3_4 = (3. / 4.);
            constexpr Tscal div3_2 = (3. / 2.);
            constexpr Tscal div1_4 = (1. / 4.);

            if (q < 1) {
                return 1 + q * q * (div3_4 * q - div3_2);
            } else if (q < 2) {
                return div1_4 * (2 - q) * (2 - q) * (2 - q);
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
    class KernelDefM6 {
        public:
        inline static constexpr Tscal Rkern  = 3;  
        inline static constexpr Tscal hfactd = 1.0; 

        inline static constexpr Tscal norm_3d = 1 / (120 * shambase::Constants<Tscal>::pi);

        inline static Tscal f(Tscal q) {

            if (q < 1.) {
                return -10. * q * q * q * q * q + 30. * q * q * q * q - 60. * q * q + 66.;
            } else if (q < 2.) {
                return -(q - 3.) * (q - 3.) * (q - 3.) * (q - 3.) * (q - 3.) +
                       6. * (q - 2.) * (q - 2.) * (q - 2.) * (q - 2.) * (q - 2.);
            } else if (q < 3.) {
                return -(q - 3.) * (q - 3.) * (q - 3.) * (q - 3.) * (q - 3.);
            } else
                return 0;
        }

        inline static Tscal df(Tscal q) {

            if (q < 1.) {
                return q * (-50. * q * q * q + 120. * q * q - 120.);
            } else if (q < 2.) {
                return -5. * (q - 3.) * (q - 3.) * (q - 3.) * (q - 3.) +
                       30. * (q - 2.) * (q - 2.) * (q - 2.) * (q - 2.);
            } else if (q < 3.) {
                return -5. * (q - 3.) * (q - 3.) * (q - 3.) * (q - 3.);
            } else
                return 0.;
        }
    };
} // namespace shammath::details

namespace shammath {

    template<class Tscal_, class BaseKernel>
    class SPHKernelGen {
        public:
        using Generator = BaseKernel;
        using Tscal = Tscal_;
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
     * @brief The M6 SPH kernel
     * \todo add graph
     *
     * @tparam flt_type the flating point representation to use
     */
    template<class flt_type>
    using M6 = SPHKernelGen<flt_type, details::KernelDefM6<flt_type>>;

} // namespace shammath