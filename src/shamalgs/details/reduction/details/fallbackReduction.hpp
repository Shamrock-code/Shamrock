// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright(C) 2021-2023 Timothée David--Cléris <timothee.david--cleris@ens-lyon.fr>
// Licensed under CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

#pragma once

#include "aliases.hpp"
#include "shambase/sycl.hpp"

namespace shamalgs::reduction::details {

    template<class T>
    struct FallbackReduction{

        static T sum(sycl::queue &q, sycl::buffer<T> &buf1, u32 start_id, u32 end_id);

        static T min(sycl::queue &q, sycl::buffer<T> &buf1, u32 start_id, u32 end_id);

        static T max(sycl::queue &q, sycl::buffer<T> &buf1, u32 start_id, u32 end_id);

    };

} // namespace shamalgs::reduction::details