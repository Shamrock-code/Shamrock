// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright(C) 2021-2023 Timothée David--Cléris <timothee.david--cleris@ens-lyon.fr>
// Licensed under CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

#pragma once

//%Impl status : Good

#include "aliases.hpp"
#include "shambase/Constants.hpp"



namespace shamrock::sph::kernels {

    // 3d kernels only

    template<class flt_type>
    class M4 {
        public:
        inline static constexpr flt_type Rkern = 2;
        inline static constexpr flt_type hfactd = 1.2;

        inline static constexpr flt_type norm = 1 / shambase::Constants<flt_type>::pi;

        inline static flt_type f(flt_type q) {

            constexpr flt_type div3_4 = (3. / 4.);
            constexpr flt_type div3_2 = (3. / 2.);
            constexpr flt_type div1_4 = (1. / 4.);

            if (q < 1) {
                return 1 + q * q * (div3_4 * q - div3_2);
            } else if (q < 2) {
                return div1_4 * (2 - q) * (2 - q) * (2 - q);
            } else
                return 0;
        }

        inline static flt_type df(flt_type q) {

            constexpr flt_type div9_4 = (9. / 4.);
            constexpr flt_type div3_4 = (3. / 4.);

            if (q < 1) {
                return -3 * q + div9_4 * q * q;
            } else if (q < 2) {
                return -3 + 3 * q - div3_4 * q * q;
            } else
                return 0;
        }

        inline static flt_type W(flt_type r, flt_type h) { return norm * f(r / h) / (h * h * h); }

        inline static flt_type dW(flt_type r, flt_type h) {
            return norm * df(r / h) / (h * h * h * h);
        }

        inline static flt_type dhW(flt_type r, flt_type h) {
            return -(norm) * (3 * f(r / h) + (r / h) * df(r / h)) / (h * h * h * h);
        }
    };

    template<class flt_type>
    class M6 {
        public:
        using flt                              = flt_type;
        inline static constexpr flt_type Rkern = 3;
        inline static constexpr flt_type hfactd = 1.0;


        inline static constexpr flt_type norm = 1 / (120 * shambase::Constants<flt_type>::pi);

        inline static flt_type f(flt_type q) {

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

        inline static flt_type df(flt_type q) {

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

        inline static flt_type W(flt_type r, flt_type h) { return norm * f(r / h) / (h * h * h); }

        inline static flt_type dW(flt_type r, flt_type h) {
            return norm * df(r / h) / (h * h * h * h);
        }

        inline static flt_type dhW(flt_type r, flt_type h) {
            return -(norm) * (3 * f(r / h) + (r / h) * df(r / h)) / (h * h * h * h);
        }
    };

} // namespace shamrock::sph::kernels