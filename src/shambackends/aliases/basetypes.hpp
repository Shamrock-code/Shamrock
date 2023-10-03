// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright(C) 2021-2023 Timothée David--Cléris <timothee.david--cleris@ens-lyon.fr>
// Licensed under CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

#pragma once

#ifdef SHAMROCK_ENABLE_BACKEND_SYCL
#include <shambackends/backends/sycl/aliases/basetypes.hpp>
#endif

#ifdef SHAMROCK_ENABLE_BACKEND_KOKKOS
#include <shambackends/backends/kokkos/aliases/basetypes.hpp>
#endif