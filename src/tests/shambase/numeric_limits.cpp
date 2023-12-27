// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright(C) 2021-2023 Timothée David--Cléris <timothee.david--cleris@ens-lyon.fr>
// Licensed under CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

#include "shamtest/shamtest.hpp"

#include "shambase/numeric_limits.hpp"

TestStart(Unittest, "shambase/numeric_limits", testnumericlimits, 1){
    
    {
        using T = f64;
        _Assert(shambase::get_max<T>() == std::numeric_limits<T>::max())
        _Assert(shambase::get_min<T>() == std::numeric_limits<T>::lowest())
        _Assert(shambase::get_epsilon<T>() == std::numeric_limits<T>::epsilon())
        _Assert(shambase::get_infty<T>() == std::numeric_limits<T>::infinity())
    }
    {
        using T = f32;
        _Assert(shambase::get_max<T>() == std::numeric_limits<T>::max())
        _Assert(shambase::get_min<T>() == std::numeric_limits<T>::lowest())
        _Assert(shambase::get_epsilon<T>() == std::numeric_limits<T>::epsilon())
        _Assert(shambase::get_infty<T>() == std::numeric_limits<T>::infinity())
    }
    {
        using T = u32;
        _Assert(shambase::get_max<T>() == std::numeric_limits<T>::max())
        _Assert(shambase::get_min<T>() == std::numeric_limits<T>::lowest())
    }
    {
        using T = i32;
        _Assert(shambase::get_max<T>() == std::numeric_limits<T>::max())
        _Assert(shambase::get_min<T>() == std::numeric_limits<T>::lowest())
    }
}