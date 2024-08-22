// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright(C) 2021-2023 Timothée David--Cléris <timothee.david--cleris@ens-lyon.fr>
// Licensed under CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

#pragma once

#include "shambackends/vec.hpp"
#include "shammath/CoordRange.hpp"

template<class vec>
shammath::CoordRange<vec> get_test_coord_ranges();

template<>
inline shammath::CoordRange<f32_3> get_test_coord_ranges() {
    return {f32_3{-1, -1, -1}, f32_3{1, 1, 1}};
}
template<>
inline shammath::CoordRange<f64_3> get_test_coord_ranges() {
    return {f64_3{-1, -1, -1}, f64_3{1, 1, 1}};
}

template<>
inline shammath::CoordRange<u32_3> get_test_coord_ranges() {
    using Prop = shambase::VectorProperties<u32_3>;
    return {u32_3{0, 0, 0}, Prop::get_max() / 2 + 1};
}
template<>
inline shammath::CoordRange<u64_3> get_test_coord_ranges() {
    using Prop = shambase::VectorProperties<u64_3>;
    return {u64_3{0, 0, 0}, Prop::get_max() / 2 + 1};
}
