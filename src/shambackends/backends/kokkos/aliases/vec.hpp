// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright(C) 2021-2023 Timothée David--Cléris <timothee.david--cleris@ens-lyon.fr>
// Licensed under CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

#pragma once

#include "basetypes.hpp"
#include "half.hpp"
#include <array>

namespace sham {

    template<typename Type, int NumElements>
    class vec {
        std::array<Type, NumElements> container;
    };

} // namespace sham