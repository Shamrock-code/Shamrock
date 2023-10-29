// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright(C) 2021-2023 Timothée David--Cléris <timothee.david--cleris@ens-lyon.fr>
// Licensed under CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

#include "shamrock/sfc/morton.hpp"
#include "shamtest/shamtest.hpp"

#include "aliases.hpp"

TestStart(Unittest, "shamrock/sfc/morton/min-max", mortonminmaxval, 1){


    using Morton32 = shamrock::sfc::MortonCodes<u32,3>;
    using Morton64 = shamrock::sfc::MortonCodes<u64,3>;


    f32 zerof32 = 0;
    f32 onef32 = 1;

    u32 m_0_32 = Morton32::coord_to_morton(zerof32, zerof32, zerof32);
    u32 m_max_32 = Morton32::coord_to_morton(onef32, onef32, onef32);

    u64 m_0_64 = Morton64::coord_to_morton(zerof32, zerof32, zerof32);
    u64 m_max_64 = Morton64::coord_to_morton(onef32, onef32, onef32);


    shamtest::asserts().assert_bool("min morton 64 == b0", m_0_64 == 0x0);    
    shamtest::asserts().assert_bool("max morton 64 == b63x1", m_max_64 == 0x7fffffffffffffff);
    shamtest::asserts().assert_bool("min morton 32 == b0x0", m_0_32 == 0x0);    
    shamtest::asserts().assert_bool("max morton 32 == b30x1", m_max_32 == 0x3fffffff);
}



