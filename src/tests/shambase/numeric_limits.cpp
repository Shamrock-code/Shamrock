// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright (c) 2021-2024 Timothée David--Cléris <tim.shamrock@proton.me>
// SPDX-License-Identifier: CeCILL Free Software License Agreement v2.1
// Shamrock is licensed under the CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

#include "shambase/numeric_limits.hpp"
#include "shamtest/shamtest.hpp"

TestStart(Unittest, "shambase/numeric_limits", testnumericlimits, 1) {

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
