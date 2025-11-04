// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright (c) 2021-2025 Timothée David--Cléris <tim.shamrock@proton.me>
// SPDX-License-Identifier: CeCILL Free Software License Agreement v2.1
// Shamrock is licensed under the CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

/**
 * @file reduction.cpp
 * @author Timothée David--Cléris (tim.shamrock@proton.me)
 * @brief
 *
 */

#include "shamalgs/details/reduction/reduction.hpp"
#include "shambase/floats.hpp"
#include "shamalgs/details/reduction/fallbackReduction.hpp"
#include "shamalgs/details/reduction/fallbackReduction_usm.hpp"
#include "shamalgs/details/reduction/groupReduction.hpp"
#include "shamalgs/details/reduction/groupReduction_usm.hpp"
#include "shamalgs/details/reduction/sycl2020reduction.hpp"
#include "shamalgs/memory.hpp"
#include "shambackends/math.hpp"
#include "shambackends/vec.hpp"

namespace shamalgs::reduction {

    template<class T>
    T sum(sycl::queue &q, sycl::buffer<T> &buf1, u32 start_id, u32 end_id) {
#ifdef SYCL2020_FEATURE_GROUP_REDUCTION
        return details::GroupReduction<T, 32>::sum(q, buf1, start_id, end_id);
#else
        return details::FallbackReduction<T>::sum(q, buf1, start_id, end_id);
#endif
    }

    template<class T>
    T max(sycl::queue &q, sycl::buffer<T> &buf1, u32 start_id, u32 end_id) {
#ifdef SYCL2020_FEATURE_GROUP_REDUCTION
        return details::GroupReduction<T, 32>::max(q, buf1, start_id, end_id);
#else
        return details::FallbackReduction<T>::max(q, buf1, start_id, end_id);
#endif
    }

    template<class T>
    T min(sycl::queue &q, sycl::buffer<T> &buf1, u32 start_id, u32 end_id) {
#ifdef SYCL2020_FEATURE_GROUP_REDUCTION
        return details::GroupReduction<T, 32>::min(q, buf1, start_id, end_id);
#else
        return details::FallbackReduction<T>::min(q, buf1, start_id, end_id);
#endif
    }

    template<class T>
    shambase::VecComponent<T> dot_sum(
        sycl::queue &q, sycl::buffer<T> &buf1, u32 start_id, u32 end_id) {
        sycl::buffer<shambase::VecComponent<T>> ret_data_base(end_id - start_id);

        q.submit([&](sycl::handler &cgh) {
            sycl::accessor acc_dot{ret_data_base, cgh, sycl::write_only, sycl::no_init};
            sycl::accessor acc{buf1, cgh, sycl::read_only};

            cgh.parallel_for(sycl::range<1>{end_id - start_id}, [=](sycl::item<1> it) {
                const T tmp = acc[it];
                acc_dot[it] = sham::dot(tmp, tmp);
            });
        });

        return sum(q, ret_data_base, 0, end_id - start_id);
    }

    bool is_all_true(sycl::buffer<u8> &buf, u32 cnt) {

        // TODO do it on GPU pleeeaze

        bool res = true;
        {
            sycl::host_accessor acc{buf, sycl::read_only};

            for (u32 i = 0; i < cnt; i++) {
                res = res && (acc[i] != 0);
            }
        }

        return res;
    }

    template<class T>
    bool has_nan(sycl::queue &q, sycl::buffer<T> &buf, u64 cnt) {
        if constexpr (shambase::VectorProperties<T>::is_float_based) {
            // res is filled with 1 if no nan 0 otherwise
            sycl::buffer<u8> res(cnt);
            q.submit([&](sycl::handler &cgh) {
                sycl::accessor acc1{buf, cgh, sycl::read_only};

                sycl::accessor out{res, cgh, sycl::write_only, sycl::no_init};

                cgh.parallel_for(sycl::range{cnt}, [=](sycl::item<1> item) {
                    out[item] = !sham::has_nan(acc1[item]);
                });
            });

            return !is_all_true(res, cnt);
        } else {
            return false;
        }
    }

    template<class T>
    bool has_inf(sycl::queue &q, sycl::buffer<T> &buf, u64 cnt) {
        if constexpr (shambase::VectorProperties<T>::is_float_based) {
            // res is filled with 1 if no inf 0 otherwise
            sycl::buffer<u8> res(cnt);
            q.submit([&](sycl::handler &cgh) {
                sycl::accessor acc1{buf, cgh, sycl::read_only};

                sycl::accessor out{res, cgh, sycl::write_only, sycl::no_init};

                cgh.parallel_for(sycl::range{cnt}, [=](sycl::item<1> item) {
                    out[item] = !sham::has_inf(acc1[item]);
                });
            });

            return !is_all_true(res, cnt);
        } else {
            return false;
        }
    }

    template<class T>
    bool has_nan_or_inf(sycl::queue &q, sycl::buffer<T> &buf, u64 cnt) {
        if constexpr (shambase::VectorProperties<T>::is_float_based) {
            // res is filled with 1 if no nan or inf 0 otherwise
            sycl::buffer<u8> res(cnt);
            q.submit([&](sycl::handler &cgh) {
                sycl::accessor acc1{buf, cgh, sycl::read_only};

                sycl::accessor out{res, cgh, sycl::write_only, sycl::no_init};

                cgh.parallel_for(sycl::range{cnt}, [=](sycl::item<1> item) {
                    out[item] = !sham::has_nan_or_inf(acc1[item]);
                });
            });

            return !is_all_true(res, cnt);
        } else {
            return false;
        }
    }

#ifndef DOXYGEN
    #define XMAC_TYPES                                                                             \
        X(f32)                                                                                     \
        X(f32_2)                                                                                   \
        X(f32_3)                                                                                   \
        X(f32_4)                                                                                   \
        X(f32_8)                                                                                   \
        X(f32_16)                                                                                  \
        X(f64)                                                                                     \
        X(f64_2)                                                                                   \
        X(f64_3)                                                                                   \
        X(f64_4)                                                                                   \
        X(f64_8)                                                                                   \
        X(f64_16)                                                                                  \
        X(u32)                                                                                     \
        X(u64)                                                                                     \
        X(i32)                                                                                     \
        X(i64)                                                                                     \
        X(u32_3)                                                                                   \
        X(u64_3)                                                                                   \
        X(i64_3)                                                                                   \
        X(i32_3)

    #define X(_arg_)                                                                               \
        template _arg_ sum(sycl::queue &q, sycl::buffer<_arg_> &buf1, u32 start_id, u32 end_id);   \
        template shambase::VecComponent<_arg_> dot_sum(                                            \
            sycl::queue &q, sycl::buffer<_arg_> &buf1, u32 start_id, u32 end_id);                  \
        template _arg_ max(sycl::queue &q, sycl::buffer<_arg_> &buf1, u32 start_id, u32 end_id);   \
        template _arg_ min(sycl::queue &q, sycl::buffer<_arg_> &buf1, u32 start_id, u32 end_id);   \
        template bool has_nan(sycl::queue &q, sycl::buffer<_arg_> &buf1, u64 cnt);                 \
        template bool has_inf(sycl::queue &q, sycl::buffer<_arg_> &buf1, u64 cnt);                 \
        template bool has_nan_or_inf(sycl::queue &q, sycl::buffer<_arg_> &buf1, u64 cnt);

    XMAC_TYPES
    #undef X
#endif

} // namespace shamalgs::reduction
