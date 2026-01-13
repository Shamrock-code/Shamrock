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
 * @file hilbert.hpp
 * @author Timothée David--Cléris (tim.shamrock@proton.me)
 * @brief hilbert curve implementation from killing J., 2004
 *
 * modified from :
 * Programming the Hilbert curve
 * killing J., 2004, AIPC, 707, 381. doi:10.1063/1.1751381
 *
 *
 */

#include "bmi.hpp"
#include "shambackends/sycl.hpp"
#include "shambackends/typeAliasVec.hpp"

// modified from :
// Programming the Hilbert curve
// killing J., 2004, AIPC, 707, 381. doi:10.1063/1.1751381

namespace shamrock::sfc {

    namespace details {

        template<int bits>
        inline u64 compute_hilbert_index_3d(u64 x, u64 y, u64 z) {

            const int n = 3;
            u64 X[3]    = {x, y, z};

            u64 M = 1 << (bits - 1), P, Q, t;
            int i;
            // Inverse undo
            for (Q = M; Q > 1; Q >>= 1) {
                P = Q - 1;
                for (i = 0; i < n; i++)
                    if (X[i] & Q)
                        X[0] ^= P; // invert
                    else {
                        t = (X[0] ^ X[i]) & P;
                        X[0] ^= t;
                        X[i] ^= t;
                    }
            } // exchange

            // Gray encode
            for (i = 1; i < n; i++)
                X[i] ^= X[i - 1];
            t = 0;
            for (Q = M; Q > 1; Q >>= 1)
                if (X[n - 1] & Q)
                    t ^= Q - 1;
            for (i = 0; i < n; i++)
                X[i] ^= t;

            X[0] = shamrock::sfc::bmi::expand_bits<u64, 2>(X[0]) << 2;
            X[1] = shamrock::sfc::bmi::expand_bits<u64, 2>(X[1]) << 1;
            X[2] = shamrock::sfc::bmi::expand_bits<u64, 2>(X[2]);

            return X[0] + X[1] + X[2];
        }
    } // namespace details

    template<class hilbert_repr, u32 dim>
    class HilbertCurve {};

    template<>
    class HilbertCurve<u64, 3> {
        public:
        using int_vec_repr_base                    = u32;
        using int_vec_repr                         = u32_3;
        static constexpr int_vec_repr_base max_val = 2097152 - 1;

        inline static u64 icoord_to_hilbert(u64 x, u64 y, u64 z) {
            return details::compute_hilbert_index_3d<21>(x, y, z);
        }

        template<class flt>
        inline static u64 coord_to_hilbert(flt x, flt y, flt z) {

            constexpr bool ok_type = std::is_same<flt, f32>::value || std::is_same<flt, f64>::value;
            static_assert(ok_type, "unknown input type");

            if constexpr (std::is_same<flt, f32>::value) {

                x = sycl::fmin(sycl::fmax(x * 2097152.F, 0.F), 2097152.F - 1.F);
                y = sycl::fmin(sycl::fmax(y * 2097152.F, 0.F), 2097152.F - 1.F);
                z = sycl::fmin(sycl::fmax(z * 2097152.F, 0.F), 2097152.F - 1.F);

                return icoord_to_hilbert(x, y, z);

            } else if constexpr (std::is_same<flt, f64>::value) {

                x = sycl::fmin(sycl::fmax(x * 2097152., 0.), 2097152. - 1.);
                y = sycl::fmin(sycl::fmax(y * 2097152., 0.), 2097152. - 1.);
                z = sycl::fmin(sycl::fmax(z * 2097152., 0.), 2097152. - 1.);

                return icoord_to_hilbert(x, y, z);
            }
        }
    };

    using quad_hilbert_num = std::pair<u64, u64>;

    template<>
    class HilbertCurve<quad_hilbert_num, 3> {

        static constexpr u64 divisor = HilbertCurve<u64, 3>::max_val + 1;

        public:
        using int_vec_repr_base                    = u64;
        using int_vec_repr                         = u64_3;
        static constexpr int_vec_repr_base max_val = divisor * divisor - 1;

        inline static quad_hilbert_num icoord_to_hilbert(u64 x, u64 y, u64 z) {

            u64 upper_val_x = x / divisor;
            u64 upper_val_y = y / divisor;
            u64 upper_val_z = z / divisor;

            u64 lower_val_x = x % divisor;
            u64 lower_val_y = y % divisor;
            u64 lower_val_z = z % divisor;

            return {
                details::compute_hilbert_index_3d<21>(upper_val_x, upper_val_y, upper_val_z),
                details::compute_hilbert_index_3d<21>(lower_val_x, lower_val_y, lower_val_z),
            };
        }
    };

} // namespace shamrock::sfc

[[deprecated]]
constexpr u64 hilbert_box21_sz
    = 2097152 - 1;

template<int bits>
[[deprecated]]
inline u64 compute_hilbert_index_3d(u64 x, u64 y, u64 z) {

    const int n = 3;
    u64 X[3]    = {x, y, z};

    u64 M = 1 << (bits - 1), P, Q, t;
    int i;
    // Inverse undo
    for (Q = M; Q > 1; Q >>= 1) {
        P = Q - 1;
        for (i = 0; i < n; i++)
            if (X[i] & Q)
                X[0] ^= P; // invert
            else {
                t = (X[0] ^ X[i]) & P;
                X[0] ^= t;
                X[i] ^= t;
            }
    } // exchange

    // Gray encode
    for (i = 1; i < n; i++)
        X[i] ^= X[i - 1];
    t = 0;
    for (Q = M; Q > 1; Q >>= 1)
        if (X[n - 1] & Q)
            t ^= Q - 1;
    for (i = 0; i < n; i++)
        X[i] ^= t;

    X[0] = shamrock::sfc::bmi::expand_bits<u64, 2>(X[0]) << 2;
    X[1] = shamrock::sfc::bmi::expand_bits<u64, 2>(X[1]) << 1;
    X[2] = shamrock::sfc::bmi::expand_bits<u64, 2>(X[2]);

    return X[0] + X[1] + X[2];
}
