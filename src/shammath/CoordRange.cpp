// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright(C) 2021-2023 Timothée David--Cléris <timothee.david--cleris@ens-lyon.fr>
// Licensed under CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

/**
 * @file CoordRange.cpp
 * @author Timothée David--Cléris (timothee.david--cleris@ens-lyon.fr)
 * @brief 
 * 
 */
 
#include "CoordRange.hpp"
#include "shambase/exception.hpp"


namespace shammath {

    template<>
    void CoordRange<f32_3>::check_throw_ranges(SourceLocation loc){
        if(lower.x() >= upper.x()){
            throw shambase::throw_with_loc<std::runtime_error>("this range is ill formed normally upper > lower");
        }
        if(lower.y() >= upper.y()){
            throw shambase::throw_with_loc<std::runtime_error>("this range is ill formed normally upper > lower");
        }
        if(lower.z() >= upper.z()){
            throw shambase::throw_with_loc<std::runtime_error>("this range is ill formed normally upper > lower");
        }
    }

    template<>
    void CoordRange<f64_3>::check_throw_ranges(SourceLocation loc){
        if(lower.x() >= upper.x()){
            throw shambase::throw_with_loc<std::runtime_error>("this range is ill formed normally upper > lower");
        }
        if(lower.y() >= upper.y()){
            throw shambase::throw_with_loc<std::runtime_error>("this range is ill formed normally upper > lower");
        }
        if(lower.z() >= upper.z()){
            throw shambase::throw_with_loc<std::runtime_error>("this range is ill formed normally upper > lower");
        }
    }

    template<>
    void CoordRange<u16_3>::check_throw_ranges(SourceLocation loc){
        if(lower.x() >= upper.x()){
            throw shambase::throw_with_loc<std::runtime_error>("this range is ill formed normally upper > lower");
        }
        if(lower.y() >= upper.y()){
            throw shambase::throw_with_loc<std::runtime_error>("this range is ill formed normally upper > lower");
        }
        if(lower.z() >= upper.z()){
            throw shambase::throw_with_loc<std::runtime_error>("this range is ill formed normally upper > lower");
        }
    }

    template<>
    void CoordRange<u32_3>::check_throw_ranges(SourceLocation loc){
        if(lower.x() >= upper.x()){
            throw shambase::throw_with_loc<std::runtime_error>("this range is ill formed normally upper > lower");
        }
        if(lower.y() >= upper.y()){
            throw shambase::throw_with_loc<std::runtime_error>("this range is ill formed normally upper > lower");
        }
        if(lower.z() >= upper.z()){
            throw shambase::throw_with_loc<std::runtime_error>("this range is ill formed normally upper > lower");
        }
    }

    template<>
    void CoordRange<u64_3>::check_throw_ranges(SourceLocation loc){
        if(lower.x() >= upper.x()){
            throw shambase::throw_with_loc<std::runtime_error>("this range is ill formed normally upper > lower");
        }
        if(lower.y() >= upper.y()){
            throw shambase::throw_with_loc<std::runtime_error>("this range is ill formed normally upper > lower");
        }
        if(lower.z() >= upper.z()){
            throw shambase::throw_with_loc<std::runtime_error>("this range is ill formed normally upper > lower");
        }
    }
    
    template<>
    void CoordRange<i32_3>::check_throw_ranges(SourceLocation loc){
        if(lower.x() >= upper.x()){
            throw shambase::throw_with_loc<std::runtime_error>("this range is ill formed normally upper > lower");
        }
        if(lower.y() >= upper.y()){
            throw shambase::throw_with_loc<std::runtime_error>("this range is ill formed normally upper > lower");
        }
        if(lower.z() >= upper.z()){
            throw shambase::throw_with_loc<std::runtime_error>("this range is ill formed normally upper > lower");
        }
    }

    template<>
    void CoordRange<i64_3>::check_throw_ranges(SourceLocation loc){
        if(lower.x() >= upper.x()){
            throw shambase::throw_with_loc<std::runtime_error>("this range is ill formed normally upper > lower");
        }
        if(lower.y() >= upper.y()){
            throw shambase::throw_with_loc<std::runtime_error>("this range is ill formed normally upper > lower");
        }
        if(lower.z() >= upper.z()){
            throw shambase::throw_with_loc<std::runtime_error>("this range is ill formed normally upper > lower");
        }
    }

}